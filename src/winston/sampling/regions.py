"""Generate deterministic class-agnostic spatial regions from RGB24 images."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

RGB24_CHANNELS = 3
DEFAULT_TILE_SCALES = (0.5,)
DEFAULT_OVERLAP = 0.25


class RegionKind(StrEnum):
    """Kind of visual candidate produced from a source image."""

    FULL = "full"
    TILE = "tile"


@dataclass(frozen=True, slots=True)
class VisualRegion:
    """One full-image or tiled visual candidate with pixel geometry and RGB24 data."""

    region_kind: RegionKind
    x: int
    y: int
    width: int
    height: int
    scale: float
    rgb24: bytes


def _validate_inputs(
    width: int,
    height: int,
    rgb24: bytes,
    tile_scales: tuple[float, ...],
    overlap: float,
) -> None:
    """Validate source geometry, RGB24 byte count, tile scales, and overlap."""
    # Reject invalid source geometry before computing byte offsets for any crop.
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")

    expected_length = width * height * RGB24_CHANNELS
    if len(rgb24) != expected_length:
        raise ValueError(
            f"RGB24 buffer length must be exactly {expected_length} bytes for {width}x{height}, "
            f"got {len(rgb24)}"
        )

    # Scales stay below 1.0 because the full-frame candidate is already emitted separately.
    for scale in tile_scales:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0.0 < float(scale) < 1.0:
            raise ValueError(f"tile scale must be greater than 0 and less than 1, got {scale!r}")

    # An overlap of 1.0 would produce a zero stride and therefore cannot advance across the image.
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or not 0.0 <= float(overlap) < 1.0:
        raise ValueError(f"overlap must be greater than or equal to 0 and less than 1, got {overlap!r}")


def _scaled_length(full_length: int, scale: float) -> int:
    """Convert a relative tile scale to a deterministic pixel length."""
    # Round halves upward instead of relying on Python's banker rounding, then clamp tiny images to one pixel.
    scaled = int(full_length * scale + 0.5)
    return min(full_length, max(1, scaled))


def _axis_positions(full_length: int, tile_length: int, overlap: float) -> tuple[int, ...]:
    """Return deterministic tile starts that cover one axis from edge to edge."""
    # If a scaled tile already spans this axis, only the zero origin is meaningful.
    if tile_length >= full_length:
        return (0,)

    nominal_stride = max(1, int(tile_length * (1.0 - overlap) + 0.5))
    last_start = full_length - tile_length
    positions = list(range(0, last_start + 1, nominal_stride))

    # Force the final tile flush with the far edge; this may increase only the final overlap but prevents gaps.
    if positions[-1] != last_start:
        positions.append(last_start)
    return tuple(positions)


def _crop_rgb24(
    rgb24: bytes,
    source_width: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> bytes:
    """Copy one rectangular RGB24 crop from a row-major source image."""
    # Copy complete RGB rows so channel triplets are never split or reordered.
    source = memoryview(rgb24)
    source_row_bytes = source_width * RGB24_CHANNELS
    crop_row_bytes = width * RGB24_CHANNELS
    cropped = bytearray(crop_row_bytes * height)

    for row in range(height):
        source_start = (y + row) * source_row_bytes + x * RGB24_CHANNELS
        target_start = row * crop_row_bytes
        cropped[target_start : target_start + crop_row_bytes] = source[
            source_start : source_start + crop_row_bytes
        ]

    return bytes(cropped)


def generate_regions(
    *,
    width: int,
    height: int,
    rgb24: bytes,
    tile_scales: tuple[float, ...] = DEFAULT_TILE_SCALES,
    overlap: float = DEFAULT_OVERLAP,
) -> Iterator[VisualRegion]:
    """Yield the full image first, then deterministic row-major overlapping tiles."""
    # Validate once before yielding any candidate so callers never receive a partial result from invalid input.
    _validate_inputs(width, height, rgb24, tile_scales, overlap)

    yield VisualRegion(
        region_kind=RegionKind.FULL,
        x=0,
        y=0,
        width=width,
        height=height,
        scale=1.0,
        rgb24=rgb24,
    )

    for raw_scale in tile_scales:
        scale = float(raw_scale)
        tile_width = _scaled_length(width, scale)
        tile_height = _scaled_length(height, scale)

        # Tiny images can round a tile to the complete source; skip that duplicate of the full-frame candidate.
        if tile_width == width and tile_height == height:
            continue

        x_positions = _axis_positions(width, tile_width, float(overlap))
        y_positions = _axis_positions(height, tile_height, float(overlap))

        # Y-major then X-major iteration gives every run the same stable region ordering.
        for y in y_positions:
            for x in x_positions:
                yield VisualRegion(
                    region_kind=RegionKind.TILE,
                    x=x,
                    y=y,
                    width=tile_width,
                    height=tile_height,
                    scale=scale,
                    rgb24=_crop_rgb24(rgb24, width, x, y, tile_width, tile_height),
                )
