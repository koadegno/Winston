from uuid import UUID, uuid4

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient, models

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.identity import visual_point_id
from winston.index.models import (
    IncompatibleVisualIndexError,
    IndexedVisual,
    RegionGeometry,
    SampleKind,
)
from winston.index.qdrant import QdrantVisualIndex
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind

QDRANT_URL = "http://localhost:6333"
IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)
DATASET_ID = "8f7ad0c0-7ab7-4ec0-9025-22f40dd70c4a"
INCOMPATIBLE_INSTANCE_ID = "0f27f24e-7436-4c46-9260-4a37f419b3a6"


def make_visual(
    *,
    asset_id: str = "e" * 64,
    source_path: str = "cameras/integration/cam01.mkv",
    timestamp_seconds: float = 42.125,
) -> IndexedVisual:
    """Build one deterministic keyframe tile for real Qdrant acceptance testing."""
    return IndexedVisual(
        asset_id=asset_id,
        source_path=source_path,
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=timestamp_seconds,
        region_kind=RegionKind.TILE,
        region=RegionGeometry(x=100, y=200, width=640, height=360, scale=0.5),
        vector=np.linspace(-1.0, 1.0, 768, dtype=np.float32),
        embedding_identity=IDENTITY,
    )


@pytest.mark.asyncio
async def test_real_qdrant_collection_and_idempotent_upsert() -> None:
    """Real Qdrant 1.18.2 must preserve Winston ownership and idempotent point semantics."""
    collection = "winston_visual_integration"
    client = AsyncQdrantClient(url=QDRANT_URL)
    index = QdrantVisualIndex(
        QdrantSettings(url=QDRANT_URL, collection=collection),
        client=client,
    )

    session = await index.ensure_compatible(IDENTITY, DATASET_ID)

    info = await client.get_collection(collection)
    assert isinstance(info.config.params.vectors, dict)
    vector = info.config.params.vectors["visual"]
    assert vector.size == 768
    assert vector.distance is models.Distance.COSINE
    assert info.config.metadata == {
        "winston_schema_version": 2,
        "dataset_instance_id": DATASET_ID,
        "index_instance_id": session.index_instance_id,
        "model_id": "jinaai/jina-clip-v1",
        "dimension": 768,
        "preprocessing_version": 1,
        "vector_name": "visual",
        "distance": "cosine",
    }
    assert str(UUID(session.index_instance_id)) == session.index_instance_id

    visual = make_visual()
    await index.upsert([visual])
    await index.upsert([visual])

    count = await client.count(collection, exact=True)
    assert count.count == 1

    records = await client.retrieve(
        collection,
        ids=[str(visual_point_id(visual))],
        with_payload=True,
        with_vectors=False,
    )
    assert len(records) == 1
    assert records[0].payload == {
        "asset_id": "e" * 64,
        "source_path": "cameras/integration/cam01.mkv",
        "media_type": "video",
        "sample_kind": "keyframe",
        "timestamp_seconds": 42.125,
        "timestamp_us": 42_125_000,
        "region_kind": "tile",
        "region": {
            "x": 100,
            "y": 200,
            "width": 640,
            "height": 360,
            "scale": 0.5,
        },
        "model_id": "jinaai/jina-clip-v1",
        "dimension": 768,
        "preprocessing_version": 1,
    }

    await index.close()


@pytest.mark.asyncio
async def test_real_qdrant_delete_old_revisions_keeps_current_asset() -> None:
    """The real Qdrant filter must remove older asset IDs without deleting the current retry ID."""
    collection = f"winston_visual_revision_{uuid4().hex}"
    client = AsyncQdrantClient(url=QDRANT_URL)
    index = QdrantVisualIndex(
        QdrantSettings(url=QDRANT_URL, collection=collection),
        client=client,
    )
    try:
        await index.ensure_compatible(IDENTITY, DATASET_ID)
        old_visual = make_visual(asset_id="a" * 64, timestamp_seconds=10.0)
        current_visual = make_visual(asset_id="f" * 64, timestamp_seconds=20.0)
        await index.upsert([old_visual, current_visual])

        await index.delete_old_revisions(
            source_path=current_visual.source_path,
            current_asset_id=current_visual.asset_id,
        )

        records = await client.retrieve(
            collection,
            ids=[
                str(visual_point_id(old_visual)),
                str(visual_point_id(current_visual)),
            ],
            with_payload=True,
            with_vectors=False,
        )
        assert len(records) == 1
        assert records[0].payload is not None
        assert records[0].payload["asset_id"] == "f" * 64
    finally:
        if await client.collection_exists(collection):
            await client.delete_collection(collection)
        await index.close()


@pytest.mark.asyncio
async def test_real_qdrant_rejects_incompatible_collection_without_mutation() -> None:
    """A real incompatible collection must be rejected and keep its original vector configuration."""
    collection = "winston_visual_incompatible_integration"
    client = AsyncQdrantClient(url=QDRANT_URL)
    await client.create_collection(
        collection,
        vectors_config={
            "visual": models.VectorParams(size=512, distance=models.Distance.COSINE)
        },
        metadata={
            "winston_schema_version": 2,
            "dataset_instance_id": DATASET_ID,
            "index_instance_id": INCOMPATIBLE_INSTANCE_ID,
            "model_id": "jinaai/jina-clip-v1",
            "dimension": 768,
            "preprocessing_version": 1,
            "vector_name": "visual",
            "distance": "cosine",
        },
    )
    index = QdrantVisualIndex(
        QdrantSettings(url=QDRANT_URL, collection=collection),
        client=client,
    )

    with pytest.raises(IncompatibleVisualIndexError, match="size=768.*size=512"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)

    info = await client.get_collection(collection)
    assert isinstance(info.config.params.vectors, dict)
    assert info.config.params.vectors["visual"].size == 512
    assert info.config.metadata is not None
    assert info.config.metadata["dimension"] == 768

    await index.close()
