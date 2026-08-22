from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SampledFrame:
    """One decoded video sample with exact source-time provenance."""

    source_path: Path
    timestamp_seconds: float
    width: int
    height: int
    rgb24: bytes
