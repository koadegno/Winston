from uuid import uuid4

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import IndexedVisual, RegionGeometry, SampleKind
from winston.index.qdrant import QdrantVisualIndex
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind
from winston.search.temporal import cosine_similarity

QDRANT_URL = "http://localhost:6333"
IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)
DATASET_ID = "52cc120e-91c1-4114-92f1-007afb735f97"


def _unit_vector(axis: int, *, secondary_axis: int | None = None, secondary: float = 0.0) -> np.ndarray:
    """Build one deterministic finite float32 vector for real cosine retrieval tests."""
    vector = np.zeros(768, dtype=np.float32)
    vector[axis] = 1.0
    if secondary_axis is not None:
        vector[secondary_axis] = secondary
    return vector


def _image(
    *,
    asset_character: str,
    source_path: str,
    vector: np.ndarray,
) -> IndexedVisual:
    """Build one full-frame still image for real Qdrant semantic retrieval."""
    return IndexedVisual(
        asset_id=asset_character * 64,
        source_path=source_path,
        media_type=MediaType.IMAGE,
        sample_kind=SampleKind.IMAGE,
        timestamp_seconds=None,
        region_kind=RegionKind.FULL,
        region=RegionGeometry(x=0, y=0, width=1280, height=720, scale=1.0),
        vector=vector,
        embedding_identity=IDENTITY,
    )


def _video_region(
    *,
    timestamp_seconds: float,
    vector: np.ndarray,
    region_kind: RegionKind = RegionKind.FULL,
    tile_x: int = 0,
) -> IndexedVisual:
    """Build one timestamped video region belonging to a shared integration-test asset."""
    if region_kind is RegionKind.FULL:
        region = RegionGeometry(x=0, y=0, width=1280, height=720, scale=1.0)
    else:
        region = RegionGeometry(x=tile_x, y=100, width=640, height=360, scale=0.5)
    return IndexedVisual(
        asset_id="c" * 64,
        source_path="videos/integration/crossing.mkv",
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=timestamp_seconds,
        region_kind=region_kind,
        region=region,
        vector=vector,
        embedding_identity=IDENTITY,
    )


@pytest.mark.asyncio
async def test_real_qdrant_semantic_ann_and_bounded_timeline_reads() -> None:
    """Qdrant 1.18.2 must support Winston's named-vector ANN and complete filtered timeline reads."""
    collection = f"winston_semantic_search_{uuid4().hex}"
    client = AsyncQdrantClient(url=QDRANT_URL)
    index = QdrantVisualIndex(
        QdrantSettings(url=QDRANT_URL, collection=collection),
        timeline_page_size=2,
        client=client,
    )
    query = _unit_vector(0)
    red_image = _image(
        asset_character="a",
        source_path="photos/red.jpg",
        vector=_unit_vector(0, secondary_axis=1, secondary=0.05),
    )
    blue_image = _image(
        asset_character="b",
        source_path="photos/blue.jpg",
        vector=_unit_vector(1),
    )
    video_visuals = (
        _video_region(
            timestamp_seconds=8.0,
            vector=_unit_vector(0, secondary_axis=1, secondary=0.8),
        ),
        _video_region(
            timestamp_seconds=12.0,
            vector=_unit_vector(0, secondary_axis=1, secondary=0.6),
        ),
        _video_region(
            timestamp_seconds=12.0,
            vector=_unit_vector(0, secondary_axis=1, secondary=0.2),
            region_kind=RegionKind.TILE,
            tile_x=320,
        ),
        _video_region(
            timestamp_seconds=16.0,
            vector=_unit_vector(1),
        ),
    )

    try:
        await index.ensure_compatible(IDENTITY, DATASET_ID)
        await index.upsert([red_image, blue_image, *video_visuals])
        session = await index.open_search(IDENTITY)

        assert session.dataset_instance_id == DATASET_ID
        matches = await index.search_visuals(query, limit=10)
        assert len(matches) == 6
        assert matches[0].visual.source_path == "photos/red.jpg"
        assert matches[0].score > matches[-1].score
        assert matches[-1].visual.source_path in {
            "photos/blue.jpg",
            "videos/integration/crossing.mkv",
        }

        for match in matches:
            assert match.score == pytest.approx(
                cosine_similarity(query, match.visual.vector),
                abs=1e-5,
            )

        window_visuals = [
            visual
            async for visual in index.iter_visuals(
                asset_id="c" * 64,
                start_timestamp_us=10_000_000,
                end_timestamp_us=14_000_000,
            )
        ]
        assert len(window_visuals) == 2
        assert [visual.timestamp_us for visual in window_visuals] == [
            12_000_000,
            12_000_000,
        ]
        assert {visual.region_kind for visual in window_visuals} == {
            RegionKind.FULL,
            RegionKind.TILE,
        }
        assert all(visual.asset_id == "c" * 64 for visual in window_visuals)
        assert all(
            visual.source_path == "videos/integration/crossing.mkv"
            for visual in window_visuals
        )
    finally:
        if await client.collection_exists(collection):
            await client.delete_collection(collection)
        await index.close()
