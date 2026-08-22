"""RGB24 conversion and canonical image preparation for Jina CLIP v1."""

from PIL import Image

from winston.embeddings.base import RGBImage
from winston.embeddings.jina.identity import JINA_CLIP_V1_IMAGE_SIZE

RGB24_CHANNELS = 3


def rgb_image_to_pil(image: RGBImage) -> Image.Image:
    """Validate a Winston RGB24 image and convert it to an RGB Pillow image."""
    # Validate geometry before constructing a Pillow image so malformed provider input fails locally.
    if isinstance(image.width, bool) or not isinstance(image.width, int) or image.width <= 0:
        raise ValueError("image width must be a positive integer")
    if isinstance(image.height, bool) or not isinstance(image.height, int) or image.height <= 0:
        raise ValueError("image height must be a positive integer")

    expected_length = image.width * image.height * RGB24_CHANNELS
    if len(image.rgb24) != expected_length:
        raise ValueError(
            f"RGB24 buffer length must be exactly {expected_length} bytes for "
            f"{image.width}x{image.height}, got {len(image.rgb24)}"
        )

    return Image.frombytes("RGB", (image.width, image.height), image.rgb24)


def prepare_jina_api_image(image: Image.Image) -> Image.Image:
    """Apply Jina CLIP v1 shortest-edge resize and center crop to 224x224 for API transport."""
    # Jina CLIP v1's official inference transform resizes the shortest edge to 224
    # with bicubic interpolation, then center-crops a 224x224 square.
    rgb = image.convert("RGB")
    width, height = rgb.size
    target = JINA_CLIP_V1_IMAGE_SIZE

    if width <= height:
        resized_width = target
        resized_height = max(target, round(height * target / width))
    else:
        resized_height = target
        resized_width = max(target, round(width * target / height))

    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    left = (resized_width - target) // 2
    top = (resized_height - target) // 2
    return resized.crop((left, top, left + target, top + target))
