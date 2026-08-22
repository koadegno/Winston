from pathlib import Path

import pytest
from PIL import Image

from winston.ingest.models import ImageMetadata
from winston.sampling.images import ImageSamplingError, load_image


def test_load_image_decodes_rgb24(tmp_path: Path) -> None:
    """A JPEG becomes contiguous RGB24 with the probed geometry."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 3), (10, 20, 30)).save(path, format="JPEG")

    sampled = load_image(ImageMetadata(path=path, width=4, height=3))

    assert sampled.source_path == path
    assert (sampled.width, sampled.height) == (4, 3)
    assert len(sampled.rgb24) == 4 * 3 * 3


def test_load_image_converts_non_rgb_source(tmp_path: Path) -> None:
    """Pillow modes other than RGB are converted explicitly before bytes leave sampling."""
    path = tmp_path / "gray.jpg"
    Image.new("L", (2, 2), 127).save(path, format="JPEG")

    sampled = load_image(ImageMetadata(path=path, width=2, height=2))

    assert len(sampled.rgb24) == 12


def test_load_image_rejects_probe_dimension_mismatch(tmp_path: Path) -> None:
    """Decoded and probed dimensions must describe the same image."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 3)).save(path, format="JPEG")

    with pytest.raises(ImageSamplingError, match=r"photo\.jpg.*4x3.*5x3"):
        load_image(ImageMetadata(path=path, width=5, height=3))


def test_load_image_wraps_corrupt_media_with_path(tmp_path: Path) -> None:
    """A corrupt image error names the asset and preserves the Pillow cause."""
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a jpeg")

    with pytest.raises(ImageSamplingError, match="broken.jpg") as caught:
        load_image(ImageMetadata(path=path, width=1, height=1))

    assert caught.value.__cause__ is not None
