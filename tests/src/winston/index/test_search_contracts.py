import math

import numpy as np
import pytest

import winston.index as index_api
from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import IndexedVisual, RegionGeometry, SampleKind
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind


IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=768,
    preprocessing_version=1,
)


def _visual() -> IndexedVisual:
    """Build one strict indexed visual used by storage-agnostic search contract tests."""
    return IndexedVisual(
        asset_id="a" * 64,
        source_path="cameras/brussels/cam01.mkv",
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=12.0,
        region_kind=RegionKind.TILE,
        region=RegionGeometry(x=100, y=50, width=640, height=360, scale=0.5),
        vector=np.ones(768, dtype=np.float32),
        embedding_identity=IDENTITY,
    )


def _search_session_type() -> type:
    """Return the public read-session type or fail on the missing Phase 1F contract."""
    session_type = getattr(index_api, "VisualSearchSession", None)
    assert isinstance(session_type, type), "winston.index must export VisualSearchSession"
    return session_type


def _scored_visual_type() -> type:
    """Return the public scored-visual type or fail on the missing Phase 1F contract."""
    scored_type = getattr(index_api, "ScoredVisual", None)
    assert isinstance(scored_type, type), "winston.index must export ScoredVisual"
    return scored_type


def test_visual_search_session_preserves_valid_collection_uuids() -> None:
    """A read session carries both dataset and index UUID identities from collection metadata."""
    session_type = _search_session_type()

    session = session_type(
        dataset_instance_id="52cc120e-91c1-4114-92f1-007afb735f97",
        index_instance_id="8b97abb3-35a4-4724-86af-f5dadc88c5d9",
    )

    assert session.dataset_instance_id == "52cc120e-91c1-4114-92f1-007afb735f97"
    assert session.index_instance_id == "8b97abb3-35a4-4724-86af-f5dadc88c5d9"


@pytest.mark.parametrize("field", ["dataset_instance_id", "index_instance_id"])
def test_visual_search_session_rejects_invalid_uuid(field: str) -> None:
    """Malformed stored collection identities must not cross the index boundary."""
    session_type = _search_session_type()
    values = {
        "dataset_instance_id": "52cc120e-91c1-4114-92f1-007afb735f97",
        "index_instance_id": "8b97abb3-35a4-4724-86af-f5dadc88c5d9",
    }
    values[field] = "not-a-uuid"

    with pytest.raises(ValueError, match="UUID"):
        session_type(**values)


def test_scored_visual_preserves_raw_finite_score() -> None:
    """ANN results pair one Winston visual with its unmodified finite cosine score."""
    scored_type = _scored_visual_type()
    visual = _visual()

    match = scored_type(visual=visual, score=0.6125)

    assert match.visual is visual
    assert match.score == pytest.approx(0.6125)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_scored_visual_rejects_non_finite_score(score: float) -> None:
    """Provider NaN or infinite scores are rejected before search orchestration sees them."""
    scored_type = _scored_visual_type()

    with pytest.raises(ValueError, match="finite"):
        scored_type(visual=_visual(), score=score)


def test_visual_index_protocol_exposes_read_only_search_methods() -> None:
    """Storage-neutral search depends only on Winston-owned read contracts."""
    for method_name in ("open_search", "search_visuals", "iter_visuals"):
        method = getattr(index_api.VisualIndex, method_name, None)
        assert callable(method), f"VisualIndex must define {method_name}"
