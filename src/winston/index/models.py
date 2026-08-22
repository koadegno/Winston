"""Winston-owned visual index domain models and validation."""

from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import PurePosixPath
import re

import numpy as np
from numpy.typing import NDArray

from winston.embeddings.models import EmbeddingIdentity
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind

type VisualVector = NDArray[np.float32]

ASSET_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class VisualIndexError(RuntimeError):
    """Base exception for Winston visual index failures."""


class VisualIndexConfigurationError(VisualIndexError):
    """Raised when Winston index configuration or input data is invalid."""


class IncompatibleVisualIndexError(VisualIndexError):
    """Raised when an existing visual index is incompatible with Winston."""


class SampleKind(StrEnum):
    """Temporal sample kind represented by one indexed visual candidate."""

    IMAGE = "image"
    KEYFRAME = "keyframe"


@dataclass(frozen=True, slots=True)
class RegionGeometry:
    """Exact pixel rectangle and source-relative scale for one visual region."""

    x: int
    y: int
    width: int
    height: int
    scale: float

    def __post_init__(self) -> None:
        """Validate deterministic region geometry before indexing."""
        coordinates = (self.x, self.y, self.width, self.height)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in coordinates):
            raise VisualIndexConfigurationError(
                "region coordinates and extents must be integers"
            )
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise VisualIndexConfigurationError(
                "region must have a non-negative origin and positive width/height"
            )
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)):
            raise VisualIndexConfigurationError("region scale must be numeric")

        scale = float(self.scale)
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise VisualIndexConfigurationError(
                "region scale must be finite, positive, and <= 1.0"
            )


@dataclass(frozen=True, slots=True)
class IndexedVisual:
    """One already-embedded visual candidate with complete source provenance."""

    asset_id: str
    source_path: str
    media_type: MediaType
    sample_kind: SampleKind
    timestamp_seconds: float | None
    region_kind: RegionKind
    region: RegionGeometry
    vector: VisualVector
    embedding_identity: EmbeddingIdentity

    def __post_init__(self) -> None:
        """Validate provenance and vector invariants before any storage I/O."""
        _validate_asset_id(self.asset_id)
        _validate_source_path(self.source_path)
        _validate_sample(self)
        _validate_region(self)
        _validate_vector(self)


def _validate_asset_id(asset_id: str) -> None:
    """Validate the canonical lowercase SHA-256 asset identifier."""
    if not isinstance(asset_id, str) or ASSET_ID_PATTERN.fullmatch(asset_id) is None:
        raise VisualIndexConfigurationError(
            "asset_id must be a 64-character lowercase SHA-256 hexadecimal string"
        )


def _validate_source_path(source_path: str) -> None:
    """Validate a portable path relative to the explicit indexing root."""
    if not isinstance(source_path, str) or not source_path or "\\" in source_path:
        raise VisualIndexConfigurationError(
            "source_path must be a non-empty relative POSIX path"
        )
    if source_path.startswith("/") or source_path.endswith("/") or "//" in source_path:
        raise VisualIndexConfigurationError(
            "source_path must be a normalized relative POSIX path"
        )

    path = PurePosixPath(source_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise VisualIndexConfigurationError(
            "source_path must be a normalized relative POSIX path without '..'"
        )


def _validate_sample(visual: IndexedVisual) -> None:
    """Validate media/sample-kind pairing and temporal provenance."""
    if visual.media_type is MediaType.IMAGE:
        if visual.sample_kind is not SampleKind.IMAGE:
            raise VisualIndexConfigurationError(
                "image media must use sample_kind=image"
            )
        if visual.timestamp_seconds is not None:
            raise VisualIndexConfigurationError(
                "image samples must not have a timestamp"
            )
        return

    if visual.media_type is MediaType.VIDEO:
        if visual.sample_kind is not SampleKind.KEYFRAME:
            raise VisualIndexConfigurationError(
                "video media must use sample_kind=keyframe"
            )
        if visual.timestamp_seconds is None:
            raise VisualIndexConfigurationError(
                "keyframe samples require a timestamp"
            )
    else:
        raise VisualIndexConfigurationError("media_type must be image or video")

    timestamp = visual.timestamp_seconds
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise VisualIndexConfigurationError("timestamp must be a finite non-negative number")
    if not math.isfinite(float(timestamp)) or float(timestamp) < 0.0:
        raise VisualIndexConfigurationError("timestamp must be finite and non-negative")


def _validate_region(visual: IndexedVisual) -> None:
    """Validate region kind invariants not expressible by geometry alone."""
    if not isinstance(visual.region, RegionGeometry):
        raise VisualIndexConfigurationError("region must be RegionGeometry")

    if visual.region_kind is RegionKind.FULL:
        if (
            visual.region.x != 0
            or visual.region.y != 0
            or float(visual.region.scale) != 1.0
        ):
            raise VisualIndexConfigurationError(
                "full-frame regions must start at x=0, y=0 with scale=1.0"
            )
    elif visual.region_kind is not RegionKind.TILE:
        raise VisualIndexConfigurationError("region_kind must be full or tile")


def _validate_vector(visual: IndexedVisual) -> None:
    """Validate one finite float32 vector against its embedding identity."""
    vector = visual.vector
    if not isinstance(vector, np.ndarray):
        raise VisualIndexConfigurationError("vector must be a NumPy array")
    if vector.ndim != 1:
        raise VisualIndexConfigurationError("vector must be one-dimensional")
    if vector.dtype != np.float32:
        raise VisualIndexConfigurationError("vector dtype must be float32")

    dimension = visual.embedding_identity.dimension
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise VisualIndexConfigurationError(
            "embedding identity dimension must be a positive integer"
        )
    if vector.shape[0] != dimension:
        raise VisualIndexConfigurationError(
            f"vector dimension {vector.shape[0]} does not match embedding identity dimension {dimension}"
        )
    if not bool(np.isfinite(vector).all()):
        raise VisualIndexConfigurationError("vector values must all be finite")
