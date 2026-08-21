from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class MediaType(StrEnum):
    VIDEO = "video"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class MediaFile:
    path: Path
    media_type: MediaType


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    codec: str
    duration_seconds: float
    fps: float


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    path: Path
    width: int
    height: int


MediaMetadata = VideoMetadata | ImageMetadata
