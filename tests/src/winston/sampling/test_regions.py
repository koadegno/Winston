from __future__ import annotations

from itertools import product

import pytest

from winston.sampling.regions import RegionKind, VisualRegion, generate_regions


def make_rgb24(width: int, height: int) -> bytes:
    """Build a deterministic RGB24 image whose pixels are easy to identify."""
    # Encode x/y into the channels so crop assertions can identify source pixels.
    return bytes(channel for y in range(height) for x in range(width) for channel in (x, y, (x + y) % 256))


def test_generate_regions_includes_full_frame_first() -> None:
    """The original image must always be the first visual candidate."""
    # Keeping the full image first gives downstream consumers a stable baseline candidate.
    rgb24 = make_rgb24(8, 8)

    regions = list(generate_regions(width=8, height=8, rgb24=rgb24))

    assert regions[0] == VisualRegion(
        region_kind=RegionKind.FULL,
        x=0,
        y=0,
        width=8,
        height=8,
        scale=1.0,
        rgb24=rgb24,
    )


def test_default_regions_are_deterministic_three_by_three_tiles() -> None:
    """The default 50% scale and 25% nominal overlap must have stable ordering."""
    # For an 8x8 image, 4x4 tiles use stride 3 and a final edge-aligned start at 4.
    regions = list(generate_regions(width=8, height=8, rgb24=make_rgb24(8, 8)))

    tiles = regions[1:]
    assert len(tiles) == 9
    assert [(tile.x, tile.y, tile.width, tile.height, tile.scale) for tile in tiles] == [
        (x, y, 4, 4, 0.5)
        for y, x in product((0, 3, 4), (0, 3, 4))
    ]


def test_tiles_cover_every_source_pixel_and_stay_inside_bounds() -> None:
    """Tile geometry must cover border pixels without exceeding the image."""
    # Mark every pixel covered by at least one tile, excluding the full-frame candidate.
    width, height = 11, 7
    tiles = list(generate_regions(width=width, height=height, rgb24=make_rgb24(width, height)))[1:]
    covered = [[False for _ in range(width)] for _ in range(height)]

    for tile in tiles:
        assert 0 <= tile.x < width
        assert 0 <= tile.y < height
        assert tile.x + tile.width <= width
        assert tile.y + tile.height <= height
        for y in range(tile.y, tile.y + tile.height):
            for x in range(tile.x, tile.x + tile.width):
                covered[y][x] = True

    assert all(all(row) for row in covered)


def test_tile_rgb24_contains_the_exact_source_crop() -> None:
    """A region's pixel bytes must correspond exactly to its declared geometry."""
    # Disable overlap here so the bottom-right tile has a simple, unambiguous 2x2 crop.
    rgb24 = make_rgb24(4, 4)
    regions = list(generate_regions(width=4, height=4, rgb24=rgb24, overlap=0.0))
    bottom_right = next(region for region in regions if (region.x, region.y) == (2, 2))

    expected = bytes(
        channel
        for y in (2, 3)
        for x in (2, 3)
        for channel in (x, y, (x + y) % 256)
    )
    assert bottom_right.rgb24 == expected


def test_generate_regions_rejects_invalid_rgb24_length() -> None:
    """Invalid raw image buffers must fail before any region is emitted."""
    # RGB24 requires exactly three bytes per pixel.
    with pytest.raises(ValueError, match="RGB24 buffer length"):
        list(generate_regions(width=4, height=4, rgb24=b"too short"))


def test_generate_regions_rejects_invalid_configuration() -> None:
    """Scale and overlap configuration must stay within meaningful ranges."""
    # Reject configurations that would create duplicate/full-size tiles or zero stride.
    with pytest.raises(ValueError, match="tile scale"):
        list(generate_regions(width=4, height=4, rgb24=make_rgb24(4, 4), tile_scales=(1.0,)))
    with pytest.raises(ValueError, match="overlap"):
        list(generate_regions(width=4, height=4, rgb24=make_rgb24(4, 4), overlap=1.0))


def test_real_camera_geometry_produces_full_frame_plus_nine_tiles() -> None:
    """The agreed 2688x1512 baseline must produce one full frame and a 3x3 tile grid."""
    # Use zeroed RGB24 bytes because this test checks the exact geometry agreed for the real camera resolution.
    width, height = 2688, 1512
    regions = list(generate_regions(width=width, height=height, rgb24=bytes(width * height * 3)))

    assert len(regions) == 10
    assert [(region.x, region.y) for region in regions[1:]] == [
        (0, 0),
        (1008, 0),
        (1344, 0),
        (0, 567),
        (1008, 567),
        (1344, 567),
        (0, 756),
        (1008, 756),
        (1344, 756),
    ]
    assert all((region.width, region.height, region.scale) == (1344, 756, 0.5) for region in regions[1:])


def test_region_types_are_exported_from_sampling_package() -> None:
    """Downstream stages must consume regions through the public sampling package API."""
    # Import here so this test fails until the package explicitly exposes the 1B API.
    from winston.sampling import RegionKind as ExportedRegionKind
    from winston.sampling import VisualRegion as ExportedVisualRegion
    from winston.sampling import generate_regions as exported_generate_regions

    assert ExportedRegionKind is RegionKind
    assert ExportedVisualRegion is VisualRegion
    assert exported_generate_regions is generate_regions
