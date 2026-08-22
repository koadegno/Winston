from dataclasses import dataclass

import pytest
from PIL import Image

from winston.embeddings.jina.preprocessing import prepare_jina_api_image, rgb_image_to_pil


@dataclass(frozen=True, slots=True)
class TinyImage:
    """Small RGB24 fixture implementing Winston's structural image contract."""

    width: int
    height: int
    rgb24: bytes


def test_rgb_image_to_pil_preserves_exact_pixels() -> None:
    """RGB24 bytes must be converted without channel swaps or geometry changes."""
    source = TinyImage(2, 1, bytes((255, 0, 0, 0, 255, 0)))
    image = rgb_image_to_pil(source)
    assert image.mode == "RGB"
    assert image.size == (2, 1)
    assert image.tobytes() == source.rgb24


def test_rgb_image_to_pil_rejects_invalid_buffer_length() -> None:
    """Malformed RGB24 images must fail before reaching an embedding provider."""
    with pytest.raises(ValueError, match="RGB24 buffer length"):
        rgb_image_to_pil(TinyImage(2, 2, b"broken"))


def test_prepare_jina_api_image_produces_one_canonical_224_square() -> None:
    """API images must be pre-sized to one Jina CLIP v1 224x224 token tile."""
    source = Image.new("RGB", (400, 200), (10, 20, 30))
    prepared = prepare_jina_api_image(source)
    assert prepared.mode == "RGB"
    assert prepared.size == (224, 224)
