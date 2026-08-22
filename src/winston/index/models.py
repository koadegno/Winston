"""Winston-owned visual index domain models and validation."""

from enum import StrEnum
import math
from pathlib import PurePosixPath
import re
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from winston.embeddings.models import EmbeddingIdentity
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind
from winston.utils.canonical import timestamp_to_microseconds

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


class RegionGeometry(BaseModel):
    """Exact pixel rectangle and source-relative scale for one visual region."""

    model_config = ConfigDict(frozen=True, strict=True)

    x: int
    y: int
    width: int
    height: int
    scale: float

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Validate deterministic region geometry after strict type validation."""
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(
                "region must have a non-negative origin and positive width/height"
            )
        if not math.isfinite(self.scale) or not 0.0 < self.scale <= 1.0:
            raise ValueError("region scale must be finite, positive, and <= 1.0")
        return self


class IndexedVisual(BaseModel):
    """One already-embedded visual candidate with complete source provenance."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        arbitrary_types_allowed=True,
    )

    asset_id: str
    source_path: str
    media_type: MediaType
    sample_kind: SampleKind
    timestamp_seconds: float | None
    region_kind: RegionKind
    region: RegionGeometry
    vector: VisualVector = Field(exclude=True)
    embedding_identity: EmbeddingIdentity = Field(exclude=True)

    @computed_field
    @property
    def timestamp_us(self) -> int | None:
        """Return the canonical integer timestamp persisted in Qdrant payloads."""
        return timestamp_to_microseconds(self.timestamp_seconds)

    @computed_field
    @property
    def model_id(self) -> str:
        """Expose the embedding model identity directly in serialized payloads."""
        return self.embedding_identity.model_id

    @computed_field
    @property
    def dimension(self) -> int:
        """Expose the embedding dimension directly in serialized payloads."""
        return self.embedding_identity.dimension

    @computed_field
    @property
    def preprocessing_version(self) -> int:
        """Expose preprocessing identity directly in serialized payloads."""
        return self.embedding_identity.preprocessing_version

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, asset_id: str) -> str:
        """Require the canonical lowercase SHA-256 asset identifier."""
        if ASSET_ID_PATTERN.fullmatch(asset_id) is None:
            raise ValueError(
                "asset_id must be a 64-character lowercase SHA-256 hexadecimal string"
            )
        return asset_id

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, source_path: str) -> str:
        """Require a normalized portable path relative to the indexing root."""
        if not source_path or "\\" in source_path:
            raise ValueError("source_path must be a non-empty relative POSIX path")
        if source_path.startswith("/") or source_path.endswith("/") or "//" in source_path:
            raise ValueError("source_path must be a normalized relative POSIX path")

        # Inspect raw components before PurePosixPath can normalize away '.' segments.
        if any(part in {"", ".", ".."} for part in source_path.split("/")):
            raise ValueError(
                "source_path must be a normalized relative POSIX path without '.' or '..'"
            )
        if PurePosixPath(source_path).is_absolute():
            raise ValueError("source_path must be a normalized relative POSIX path")
        return source_path

    @field_validator("timestamp_seconds")
    @classmethod
    def validate_timestamp(cls, timestamp_seconds: float | None) -> float | None:
        """Reject non-finite or negative video timestamps."""
        if timestamp_seconds is None:
            return None
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        return timestamp_seconds

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Validate cross-field media, region, and vector invariants."""
        self._validate_sample()
        self._validate_region()
        self._validate_vector()
        return self

    def _validate_sample(self) -> None:
        """Validate media/sample-kind pairing and temporal provenance."""
        if self.media_type is MediaType.IMAGE:
            if self.sample_kind is not SampleKind.IMAGE:
                raise ValueError("image media must use sample_kind=image")
            if self.timestamp_seconds is not None:
                raise ValueError("image samples must not have a timestamp")
            return

        if self.sample_kind is not SampleKind.KEYFRAME:
            raise ValueError("video media must use sample_kind=keyframe")
        if self.timestamp_seconds is None:
            raise ValueError("keyframe samples require a timestamp")

    def _validate_region(self) -> None:
        """Validate region-kind invariants not expressible by geometry alone."""
        if self.region_kind is RegionKind.FULL and (
            self.region.x != 0
            or self.region.y != 0
            or self.region.scale != 1.0
        ):
            raise ValueError(
                "full-frame regions must start at x=0, y=0 with scale=1.0"
            )

    def _validate_vector(self) -> None:
        """Validate one finite float32 vector against its embedding identity."""
        if self.vector.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if self.vector.dtype != np.float32:
            raise ValueError("vector dtype must be float32")

        dimension = self.embedding_identity.dimension
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("embedding identity dimension must be a positive integer")
        if self.vector.shape[0] != dimension:
            raise ValueError(
                f"vector dimension {self.vector.shape[0]} does not match "
                f"embedding identity dimension {dimension}"
            )
        if not bool(np.isfinite(self.vector).all()):
            raise ValueError("vector values must all be finite")
