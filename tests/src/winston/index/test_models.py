from dataclasses import replace

import numpy as np
import pytest

from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import (
    IndexedVisual,
    RegionGeometry,
    SampleKind,
    VisualIndexConfigurationError,
)
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind

ASSET_ID = "a" * 64
IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=768,
    preprocessing_version=1,
)


def make_visual(**overrides: object) -> IndexedVisual:
    """Build one valid video tile and apply explicit test overrides."""
    values: dict[str, object] = {
        "asset_id": ASSET_ID,
        "source_path": "cameras/brussels/cam01.mkv",
        "media_type": MediaType.VIDEO,
        "sample_kind": SampleKind.KEYFRAME,
        "timestamp_seconds": 123.456,
        "region_kind": RegionKind.TILE,
        "region": RegionGeometry(x=1008, y=567, width=1344, height=756, scale=0.5),
        "vector": np.ones(768, dtype=np.float32),
        "embedding_identity": IDENTITY,
    }
    values.update(overrides)
    return IndexedVisual(**values)  # type: ignore[arg-type]


def make_image_visual(**overrides: object) -> IndexedVisual:
    """Build one valid still-image full-frame visual and apply explicit test overrides."""
    values: dict[str, object] = {
        "asset_id": ASSET_ID,
        "source_path": "photos/front-door.jpg",
        "media_type": MediaType.IMAGE,
        "sample_kind": SampleKind.IMAGE,
        "timestamp_seconds": None,
        "region_kind": RegionKind.FULL,
        "region": RegionGeometry(x=0, y=0, width=1920, height=1080, scale=1.0),
        "vector": np.ones(768, dtype=np.float32),
        "embedding_identity": IDENTITY,
    }
    values.update(overrides)
    return IndexedVisual(**values)  # type: ignore[arg-type]


def test_indexed_visual_accepts_valid_video_tile() -> None:
    """A valid keyframe tile must retain exact provenance and a 1-D float32 vector."""
    visual = make_visual()

    assert visual.sample_kind is SampleKind.KEYFRAME
    assert visual.region_kind is RegionKind.TILE
    assert visual.vector.shape == (768,)
    assert visual.vector.dtype == np.float32


def test_indexed_visual_rejects_wrong_vector_dimension() -> None:
    """Vector size must match the embedding identity before any Qdrant I/O."""
    with pytest.raises(VisualIndexConfigurationError, match="vector dimension"):
        make_visual(vector=np.ones(767, dtype=np.float32))


def test_indexed_visual_rejects_non_float32_vector() -> None:
    """Stored visual vectors must already use Winston's float32 embedding representation."""
    with pytest.raises(VisualIndexConfigurationError, match="float32"):
        make_visual(vector=np.ones(768, dtype=np.float64))


def test_indexed_visual_rejects_non_vector_matrix() -> None:
    """One indexed visual maps to one vector, never a batch matrix."""
    with pytest.raises(VisualIndexConfigurationError, match="one-dimensional"):
        make_visual(vector=np.ones((1, 768), dtype=np.float32))


def test_indexed_visual_rejects_non_finite_vector() -> None:
    """NaN or infinite vector values must be rejected before persistence."""
    vector = np.ones(768, dtype=np.float32)
    vector[20] = np.nan
    with pytest.raises(VisualIndexConfigurationError, match="finite"):
        make_visual(vector=vector)


def test_indexed_visual_rejects_image_with_timestamp() -> None:
    """Photos must use sample_kind=image and no timestamp."""
    with pytest.raises(VisualIndexConfigurationError, match="image samples must not have a timestamp"):
        make_image_visual(timestamp_seconds=1.0)


def test_indexed_visual_rejects_video_keyframe_without_timestamp() -> None:
    """Video keyframes must preserve a concrete source timestamp."""
    with pytest.raises(VisualIndexConfigurationError, match="keyframe samples require a timestamp"):
        make_visual(timestamp_seconds=None)


def test_indexed_visual_rejects_media_sample_kind_mismatch() -> None:
    """Image samples and video keyframes cannot be mislabeled across media types."""
    with pytest.raises(VisualIndexConfigurationError, match="image media must use sample_kind=image"):
        make_image_visual(sample_kind=SampleKind.KEYFRAME, timestamp_seconds=1.0)
    with pytest.raises(VisualIndexConfigurationError, match="video media must use sample_kind=keyframe"):
        make_visual(sample_kind=SampleKind.IMAGE, timestamp_seconds=None)


def test_indexed_visual_rejects_invalid_asset_id() -> None:
    """Asset IDs crossing the index boundary must be canonical lowercase SHA-256 hex strings."""
    for invalid in ("a" * 63, "A" * 64, "g" * 64):
        with pytest.raises(VisualIndexConfigurationError, match="asset_id"):
            make_visual(asset_id=invalid)


def test_indexed_visual_rejects_non_portable_source_path() -> None:
    """Qdrant payload paths must remain normalized relative POSIX paths under the indexing root."""
    for invalid in (
        "/absolute/cam.mkv",
        "../escape.mkv",
        "cams/../escape.mkv",
        "cams/./cam.mkv",
        ".",
        r"cams\cam.mkv",
        "",
    ):
        with pytest.raises(VisualIndexConfigurationError, match="source_path"):
            make_visual(source_path=invalid)


def test_indexed_visual_rejects_invalid_timestamp() -> None:
    """Video timestamps must be finite and non-negative."""
    for invalid in (-0.1, float("nan"), float("inf")):
        with pytest.raises(VisualIndexConfigurationError, match="timestamp"):
            make_visual(timestamp_seconds=invalid)


def test_region_geometry_rejects_negative_origin_or_non_positive_extent() -> None:
    """Region geometry must describe a non-empty rectangle inside positive pixel coordinates."""
    invalid_regions = (
        (-1, 0, 10, 10, 0.5),
        (0, -1, 10, 10, 0.5),
        (0, 0, 0, 10, 0.5),
        (0, 0, 10, 0, 0.5),
    )
    for x, y, width, height, scale in invalid_regions:
        with pytest.raises(VisualIndexConfigurationError, match="region"):
            RegionGeometry(x=x, y=y, width=width, height=height, scale=scale)


def test_region_geometry_rejects_invalid_scale() -> None:
    """Region scale must be finite, positive, and no greater than the complete source image."""
    for scale in (0.0, -0.5, 1.1, float("nan"), float("inf")):
        with pytest.raises(VisualIndexConfigurationError, match="scale"):
            RegionGeometry(x=0, y=0, width=10, height=10, scale=scale)


def test_indexed_visual_rejects_invalid_full_frame_geometry() -> None:
    """Full-frame candidates must start at the origin and carry scale 1.0."""
    with pytest.raises(VisualIndexConfigurationError, match="full-frame"):
        make_image_visual(region=RegionGeometry(x=1, y=0, width=1920, height=1080, scale=1.0))
    with pytest.raises(VisualIndexConfigurationError, match="full-frame"):
        make_image_visual(region=RegionGeometry(x=0, y=0, width=1920, height=1080, scale=0.5))


def test_frozen_visual_does_not_copy_vector() -> None:
    """Validation must not double vector memory merely because the domain model is frozen."""
    vector = np.ones(768, dtype=np.float32)
    visual = make_visual(vector=vector)

    assert visual.vector is vector
    assert replace(visual, timestamp_seconds=123.456).vector is vector
