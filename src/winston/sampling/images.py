"""Decode still images into Winston's RGB24 sampling contract."""

from PIL import Image

from winston.ingest.models import ImageMetadata
from winston.sampling.models import SampledImage


class ImageSamplingError(RuntimeError):
    """Raised when a still image cannot satisfy Winston's sampled-image contract."""


def load_image(metadata: ImageMetadata) -> SampledImage:
    """Decode one image, convert to RGB, and validate it against probe metadata."""
    try:
        with Image.open(metadata.path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            if (width, height) != (metadata.width, metadata.height):
                raise ImageSamplingError(
                    f"Decoded dimensions for {metadata.path} are {width}x{height}; "
                    f"probe reported {metadata.width}x{metadata.height}"
                )
            rgb24 = rgb.tobytes()
    except ImageSamplingError:
        raise
    except OSError as exc:
        raise ImageSamplingError(f"Failed to decode image {metadata.path}: {exc}") from exc

    expected_bytes = width * height * 3
    if len(rgb24) != expected_bytes:
        raise ImageSamplingError(
            f"Decoded RGB24 buffer for {metadata.path} has {len(rgb24)} bytes; "
            f"expected {expected_bytes}"
        )

    return SampledImage(
        source_path=metadata.path,
        width=width,
        height=height,
        rgb24=rgb24,
    )
