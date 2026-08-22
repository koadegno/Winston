import numpy as np

from winston.embeddings.models import EmbeddingIdentity
from winston.index.identity import timestamp_to_microseconds, visual_point_id
from winston.index.models import IndexedVisual, RegionGeometry, SampleKind
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind

ASSET_ID = "b" * 64
IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)


def make_visual(
    *,
    timestamp_seconds: float = 123.456,
    region: RegionGeometry | None = None,
    region_kind: RegionKind = RegionKind.TILE,
    embedding_identity: EmbeddingIdentity = IDENTITY,
) -> IndexedVisual:
    """Build a valid visual candidate for deterministic identity tests."""
    geometry = region or RegionGeometry(x=10, y=20, width=100, height=80, scale=0.5)
    return IndexedVisual(
        asset_id=ASSET_ID,
        source_path="cameras/cam01.mkv",
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=timestamp_seconds,
        region_kind=region_kind,
        region=geometry,
        vector=np.ones(embedding_identity.dimension, dtype=np.float32),
        embedding_identity=embedding_identity,
    )


def test_timestamp_to_microseconds_uses_half_up_rounding() -> None:
    """Point identity must not depend on binary float formatting or banker's rounding."""
    assert timestamp_to_microseconds(0.0000005) == 1
    assert timestamp_to_microseconds(0.0000015) == 2
    assert timestamp_to_microseconds(123.456) == 123_456_000
    assert timestamp_to_microseconds(None) is None


def test_visual_point_id_is_deterministic_uuid5() -> None:
    """The same logical visual candidate must always map to the same UUIDv5."""
    visual = make_visual()

    first = visual_point_id(visual)
    second = visual_point_id(visual)

    assert first == second
    assert first.version == 5


def test_visual_point_id_changes_with_timestamp() -> None:
    """Temporal sample identity must affect the Qdrant point ID."""
    assert visual_point_id(make_visual()) != visual_point_id(make_visual(timestamp_seconds=124.0))


def test_visual_point_id_changes_with_region_geometry() -> None:
    """Exact pixel rectangle must affect spatial point identity."""
    base = make_visual()
    moved = make_visual(region=RegionGeometry(x=11, y=20, width=100, height=80, scale=0.5))
    resized = make_visual(region=RegionGeometry(x=10, y=20, width=101, height=80, scale=0.5))

    assert visual_point_id(base) != visual_point_id(moved)
    assert visual_point_id(base) != visual_point_id(resized)


def test_visual_point_id_changes_with_embedding_identity() -> None:
    """Model, dimension, and preprocessing identity must participate in point identity."""
    base = make_visual()
    other_model = make_visual(embedding_identity=EmbeddingIdentity("other-model", 768, 1))
    other_dimension = make_visual(embedding_identity=EmbeddingIdentity("jinaai/jina-clip-v1", 512, 1))
    other_preprocessing = make_visual(embedding_identity=EmbeddingIdentity("jinaai/jina-clip-v1", 768, 2))

    assert visual_point_id(base) != visual_point_id(other_model)
    assert visual_point_id(base) != visual_point_id(other_dimension)
    assert visual_point_id(base) != visual_point_id(other_preprocessing)


def test_visual_point_id_changes_with_region_kind() -> None:
    """Full-frame and tile candidates must remain distinct semantic candidates."""
    tile = make_visual(region=RegionGeometry(x=0, y=0, width=100, height=80, scale=1.0))
    full = make_visual(
        region=RegionGeometry(x=0, y=0, width=100, height=80, scale=1.0),
        region_kind=RegionKind.FULL,
    )

    assert visual_point_id(tile) != visual_point_id(full)


def test_visual_point_id_ignores_scale_when_pixel_rectangle_is_identical() -> None:
    """Scale is provenance, while exact pixel geometry defines spatial identity."""
    first = make_visual(region=RegionGeometry(x=10, y=20, width=100, height=80, scale=0.5))
    second = make_visual(region=RegionGeometry(x=10, y=20, width=100, height=80, scale=0.6))

    assert visual_point_id(first) == visual_point_id(second)
