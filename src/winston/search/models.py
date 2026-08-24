"""User-facing models for semantic search results."""

import math
from pathlib import PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from winston.index.models import RegionGeometry
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind


class SearchRunError(RuntimeError):
    """Raised when one semantic-search invocation cannot be completed safely."""


class SearchResult(BaseModel):
    """One ranked semantic match returned to a Winston user.

    ``raw_score`` is the original cosine similarity for the representative visual.
    It is deliberately not converted into a percentage or confidence value.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    source_path: str
    media_type: MediaType
    start_timestamp_seconds: float | None
    end_timestamp_seconds: float | None
    representative_timestamp_seconds: float | None
    region_kind: RegionKind
    region: RegionGeometry
    raw_score: float

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, source_path: str) -> str:
        """Require the same normalized relative POSIX path used by indexed visuals."""
        if not source_path or "\\" in source_path:
            raise ValueError("source_path must be a non-empty relative POSIX path")
        if source_path.startswith("/") or source_path.endswith("/") or "//" in source_path:
            raise ValueError("source_path must be a normalized relative POSIX path")
        if any(part in {"", ".", ".."} for part in source_path.split("/")):
            raise ValueError("source_path must not contain empty, current, or parent path segments")
        if PurePosixPath(source_path).is_absolute():
            raise ValueError("source_path must be relative")
        return source_path

    @field_validator(
        "start_timestamp_seconds",
        "end_timestamp_seconds",
        "representative_timestamp_seconds",
    )
    @classmethod
    def validate_timestamp(cls, value: float | None) -> float | None:
        """Require present timestamps to be finite non-negative video positions."""
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError("timestamps must be finite and non-negative")
        return value

    @field_validator("raw_score")
    @classmethod
    def validate_raw_score(cls, raw_score: float) -> float:
        """Reject NaN and infinities while preserving the unmodified cosine score."""
        if not math.isfinite(raw_score):
            raise ValueError("raw_score must be finite")
        return raw_score

    @model_validator(mode="after")
    def validate_temporal_shape(self) -> Self:
        """Keep image and video timestamp shapes unambiguous and internally ordered."""
        start = self.start_timestamp_seconds
        end = self.end_timestamp_seconds
        representative = self.representative_timestamp_seconds

        if self.media_type is MediaType.IMAGE:
            if start is not None or end is not None or representative is not None:
                raise ValueError("image results must not contain timestamps")
            return self

        if start is None or end is None or representative is None:
            raise ValueError("video results require start, end, and representative timestamps")
        if not start <= representative <= end:
            raise ValueError("video timestamps must satisfy start <= representative <= end")
        return self
