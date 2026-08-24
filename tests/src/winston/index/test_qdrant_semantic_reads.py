from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import JsonValue
from qdrant_client import models

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import (
    IncompatibleVisualIndexError,
    SampleKind,
    VisualIndexConfigurationError,
)
from winston.index.qdrant import QdrantVisualIndex
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind

IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=768,
    preprocessing_version=1,
)
DATASET_ID = "52cc120e-91c1-4114-92f1-007afb735f97"
INDEX_INSTANCE_ID = "8b97abb3-35a4-4724-86af-f5dadc88c5d9"
ASSET_ID = "a" * 64


type MetadataValue = str | int
type Metadata = dict[str, MetadataValue]
type DenseVectorOutput = list[float] | dict[str, list[float]] | None


def _metadata() -> Metadata:
    """Return the exact schema-v2 metadata required by read-only semantic search."""
    return {
        "winston_schema_version": 2,
        "dataset_instance_id": DATASET_ID,
        "index_instance_id": INDEX_INSTANCE_ID,
        "model_id": IDENTITY.model_id,
        "dimension": IDENTITY.dimension,
        "preprocessing_version": IDENTITY.preprocessing_version,
        "vector_name": "visual",
        "distance": "cosine",
    }


def _payload(**overrides: JsonValue) -> dict[str, JsonValue]:
    """Build one complete stored visual payload and apply explicit corruption overrides."""
    payload: dict[str, JsonValue] = {
        "asset_id": ASSET_ID,
        "source_path": "cameras/brussels/cam01.mkv",
        "media_type": "video",
        "sample_kind": "keyframe",
        "timestamp_seconds": 12.5,
        "timestamp_us": 12_500_000,
        "region_kind": "tile",
        "region": {"x": 100, "y": 50, "width": 640, "height": 360, "scale": 0.5},
        "model_id": IDENTITY.model_id,
        "dimension": IDENTITY.dimension,
        "preprocessing_version": IDENTITY.preprocessing_version,
    }
    payload.update(overrides)
    return payload


@dataclass(slots=True)
class SearchCollectionParams:
    """Minimal Qdrant collection params consumed by read-only compatibility checks."""

    vectors: dict[str, models.VectorParams]


@dataclass(slots=True)
class SearchCollectionConfig:
    """Minimal read-only Qdrant collection config."""

    params: SearchCollectionParams
    metadata: Metadata | None


@dataclass(slots=True)
class SearchCollectionInfo:
    """Minimal read-only Qdrant collection info."""

    config: SearchCollectionConfig


@dataclass(frozen=True, slots=True)
class FakeScoredPoint:
    """One ANN point returned by the fake Qdrant client."""

    score: float
    payload: dict[str, JsonValue] | None
    vector: DenseVectorOutput


@dataclass(frozen=True, slots=True)
class FakeQueryResponse:
    """Minimal query response matching the adapter surface used by Winston."""

    points: list[FakeScoredPoint]


@dataclass(frozen=True, slots=True)
class QueryCall:
    """Captured coarse ANN query arguments."""

    collection_name: str
    query: list[float]
    using: str
    limit: int
    with_payload: bool
    with_vectors: list[str]


class SearchOnlyQdrantClient:
    """Read-only fake that deliberately exposes no create/delete/upsert operations."""

    def __init__(
        self,
        *,
        exists: bool = True,
        metadata: Metadata | None = None,
        points: list[FakeScoredPoint] | None = None,
    ) -> None:
        """Configure collection existence, metadata, and ANN response points."""
        self.exists = exists
        self.info = SearchCollectionInfo(
            config=SearchCollectionConfig(
                params=SearchCollectionParams(
                    vectors={
                        "visual": models.VectorParams(
                            size=768,
                            distance=models.Distance.COSINE,
                        )
                    }
                ),
                metadata=_metadata() if metadata is None else metadata,
            )
        )
        self.points = [] if points is None else points
        self.query_calls: list[QueryCall] = []
        self.closed = False

    async def collection_exists(self, collection_name: str) -> bool:
        """Return whether the configured read-only collection exists."""
        return self.exists

    async def get_collection(self, collection_name: str) -> SearchCollectionInfo:
        """Return the configured collection metadata without mutation."""
        return self.info

    async def query_points(
        self,
        collection_name: str,
        *,
        query: list[float],
        using: str,
        limit: int,
        with_payload: bool,
        with_vectors: list[str],
    ) -> FakeQueryResponse:
        """Capture one coarse ANN query and return configured points."""
        self.query_calls.append(
            QueryCall(
                collection_name=collection_name,
                query=query,
                using=using,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
        )
        return FakeQueryResponse(points=self.points)

    async def close(self) -> None:
        """Record client closure."""
        self.closed = True


def _client(
    *,
    exists: bool = True,
    metadata: Metadata | None = None,
    points: list[FakeScoredPoint] | None = None,
) -> SearchOnlyQdrantClient:
    """Build one mutation-free Qdrant fake for semantic-read contracts."""
    return SearchOnlyQdrantClient(exists=exists, metadata=metadata, points=points)


@pytest.mark.asyncio
async def test_open_search_validates_existing_collection_without_writes() -> None:
    """Read-only open returns both persisted UUIDs and never needs a mutation method."""
    client = _client()
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]

    session = await index.open_search(IDENTITY)

    assert session.dataset_instance_id == DATASET_ID
    assert session.index_instance_id == INDEX_INSTANCE_ID
    assert client.query_calls == []


@pytest.mark.asyncio
async def test_open_search_missing_collection_tells_user_to_index_first() -> None:
    """Search must not create missing storage and instead direct the caller to indexing."""
    client = _client(exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(VisualIndexConfigurationError, match="index.*first"):
        await index.open_search(IDENTITY)


@pytest.mark.asyncio
async def test_open_search_rejects_invalid_dataset_uuid_in_collection_metadata() -> None:
    """Stored dataset ownership remains mandatory even though search receives no dataset path."""
    metadata = _metadata()
    metadata["dataset_instance_id"] = "not-a-uuid"
    client = _client(metadata=metadata)
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(IncompatibleVisualIndexError, match="dataset_instance_id"):
        await index.open_search(IDENTITY)


@pytest.mark.asyncio
async def test_search_visuals_queries_named_vector_and_maps_winston_visual() -> None:
    """Coarse ANN retrieval asks only for the named visual vector plus payload."""
    stored_vector = [0.0] * 768
    stored_vector[17] = 1.0
    point = FakeScoredPoint(
        score=0.731,
        payload=_payload(),
        vector={"visual": stored_vector},
    )
    client = _client(points=[point])
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]
    await index.open_search(IDENTITY)
    query_vector = np.zeros(768, dtype=np.float32)
    query_vector[17] = 1.0

    matches = await index.search_visuals(query_vector, limit=23)

    assert len(client.query_calls) == 1
    call = client.query_calls[0]
    assert call.collection_name == "winston_visual"
    assert call.query == query_vector.tolist()
    assert call.using == "visual"
    assert call.limit == 23
    assert call.with_payload is True
    assert call.with_vectors == ["visual"]

    assert len(matches) == 1
    match = matches[0]
    assert match.score == pytest.approx(0.731)
    assert match.visual.asset_id == ASSET_ID
    assert match.visual.source_path == "cameras/brussels/cam01.mkv"
    assert match.visual.media_type is MediaType.VIDEO
    assert match.visual.sample_kind is SampleKind.KEYFRAME
    assert match.visual.timestamp_seconds == pytest.approx(12.5)
    assert match.visual.region_kind is RegionKind.TILE
    assert match.visual.region.x == 100
    assert match.visual.vector.dtype == np.float32
    assert match.visual.vector[17] == pytest.approx(1.0)
    assert match.visual.embedding_identity == IDENTITY


@pytest.mark.asyncio
async def test_search_visuals_requires_successful_read_open() -> None:
    """ANN retrieval cannot bypass collection compatibility validation."""
    client = _client()
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(VisualIndexConfigurationError, match="open_search"):
        await index.search_visuals(np.ones(768, dtype=np.float32), limit=5)

    assert client.query_calls == []


@pytest.mark.asyncio
async def test_search_visuals_rejects_payload_embedding_identity_mismatch() -> None:
    """Per-point payload identity is validated even after collection metadata passed compatibility."""
    point = FakeScoredPoint(
        score=0.5,
        payload=_payload(model_id="other-model"),
        vector={"visual": [1.0] * 768},
    )
    client = _client(points=[point])
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]
    await index.open_search(IDENTITY)

    with pytest.raises(IncompatibleVisualIndexError, match="model_id"):
        await index.search_visuals(np.ones(768, dtype=np.float32), limit=5)


@pytest.mark.asyncio
async def test_search_visuals_rejects_missing_named_stored_vector() -> None:
    """A scored point without Winston's named vector is incompatible with local refinement."""
    point = FakeScoredPoint(score=0.5, payload=_payload(), vector={"other": [1.0] * 768})
    client = _client(points=[point])
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]
    await index.open_search(IDENTITY)

    with pytest.raises(IncompatibleVisualIndexError, match="visual"):
        await index.search_visuals(np.ones(768, dtype=np.float32), limit=5)


@pytest.mark.asyncio
async def test_search_visuals_rejects_invalid_query_vector_before_qdrant() -> None:
    """Query dimension/dtype errors must fail locally before an ANN request is issued."""
    client = _client()
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]
    await index.open_search(IDENTITY)

    with pytest.raises(VisualIndexConfigurationError, match="float32"):
        await index.search_visuals(np.ones(768, dtype=np.float64), limit=5)  # type: ignore[arg-type]
    with pytest.raises(VisualIndexConfigurationError, match="dimension"):
        await index.search_visuals(np.ones(767, dtype=np.float32), limit=5)
    with pytest.raises(VisualIndexConfigurationError, match="positive"):
        await index.search_visuals(np.ones(768, dtype=np.float32), limit=0)

    assert client.query_calls == []
