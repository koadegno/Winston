from dataclasses import dataclass

import pytest
from qdrant_client import models

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import IncompatibleVisualIndexError, VisualIndexError
from winston.index.qdrant import QdrantVisualIndex

IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=768,
    preprocessing_version=1,
)


def expected_metadata(identity: EmbeddingIdentity = IDENTITY) -> dict[str, object]:
    """Return the exact Winston collection metadata expected by Phase 1D."""
    return {
        "winston_schema_version": 1,
        "model_id": identity.model_id,
        "dimension": identity.dimension,
        "preprocessing_version": identity.preprocessing_version,
        "vector_name": "visual",
        "distance": "cosine",
    }


@dataclass(slots=True)
class FakeCollectionParams:
    """Minimal collection parameter shape consumed by QdrantVisualIndex."""

    vectors: dict[str, models.VectorParams] | None


@dataclass(slots=True)
class FakeCollectionConfig:
    """Minimal collection config shape consumed by QdrantVisualIndex."""

    params: FakeCollectionParams
    metadata: dict[str, object] | None


@dataclass(slots=True)
class FakeCollectionInfo:
    """Minimal collection info shape consumed by QdrantVisualIndex."""

    config: FakeCollectionConfig


@dataclass(frozen=True, slots=True)
class CreateCall:
    """Captured collection creation arguments."""

    collection_name: str
    vectors_config: dict[str, models.VectorParams]
    metadata: dict[str, object]


class FakeQdrantClient:
    """Non-destructive async fake implementing only the operations Winston may use."""

    def __init__(
        self,
        *,
        exists: bool,
        info: FakeCollectionInfo | None = None,
        failure: Exception | None = None,
    ) -> None:
        """Configure existence, collection info, or a provider failure."""
        self.exists = exists
        self.info = info
        self.failure = failure
        self.create_calls: list[CreateCall] = []
        self.closed = False

    async def collection_exists(self, collection_name: str) -> bool:
        """Return the configured collection existence or raise the configured failure."""
        if self.failure is not None:
            raise self.failure
        return self.exists

    async def create_collection(
        self,
        collection_name: str,
        *,
        vectors_config: dict[str, models.VectorParams],
        metadata: dict[str, object],
    ) -> bool:
        """Capture non-destructive collection creation arguments."""
        self.create_calls.append(
            CreateCall(
                collection_name=collection_name,
                vectors_config=vectors_config,
                metadata=metadata,
            )
        )
        self.exists = True
        return True

    async def get_collection(self, collection_name: str) -> FakeCollectionInfo:
        """Return configured collection information."""
        if self.failure is not None:
            raise self.failure
        if self.info is None:
            raise RuntimeError(f"no fake info configured for {collection_name}")
        return self.info

    async def close(self) -> None:
        """Record client closure."""
        self.closed = True


def make_info(
    *,
    size: int = 768,
    distance: models.Distance = models.Distance.COSINE,
    vector_name: str = "visual",
    metadata: dict[str, object] | None = None,
) -> FakeCollectionInfo:
    """Build compatible-looking collection info with selected incompatibilities."""
    return FakeCollectionInfo(
        config=FakeCollectionConfig(
            params=FakeCollectionParams(
                vectors={
                    vector_name: models.VectorParams(size=size, distance=distance),
                }
            ),
            metadata=expected_metadata() if metadata is None else metadata,
        )
    )


@pytest.mark.asyncio
async def test_ensure_compatible_creates_missing_collection() -> None:
    """A missing collection must be created with the exact named-vector and Winston metadata contract."""
    client = FakeQdrantClient(exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    await index.ensure_compatible(IDENTITY)

    assert len(client.create_calls) == 1
    call = client.create_calls[0]
    assert call.collection_name == "winston_visual"
    assert set(call.vectors_config) == {"visual"}
    vector = call.vectors_config["visual"]
    assert vector.size == 768
    assert vector.distance is models.Distance.COSINE
    assert call.metadata == expected_metadata()


@pytest.mark.asyncio
async def test_ensure_compatible_accepts_matching_existing_collection() -> None:
    """A fully matching owned collection must be reused without mutation."""
    client = FakeQdrantClient(exists=True, info=make_info())
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    await index.ensure_compatible(IDENTITY)

    assert client.create_calls == []


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_collection_without_winston_metadata() -> None:
    """An unowned collection must never be silently adopted."""
    info = make_info()
    info.config.metadata = None
    client = FakeQdrantClient(exists=True, info=info)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="metadata"):
        await index.ensure_compatible(IDENTITY)

    assert client.create_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("winston_schema_version", 2),
        ("model_id", "other-model"),
        ("dimension", 512),
        ("preprocessing_version", 2),
        ("vector_name", "other-vector"),
        ("distance", "dot"),
    ],
)
async def test_ensure_compatible_rejects_incompatible_winston_metadata(
    field: str,
    value: object,
) -> None:
    """Every required Winston compatibility field must match exactly."""
    metadata = expected_metadata()
    metadata[field] = value
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match=field):
        await index.ensure_compatible(IDENTITY)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_missing_required_metadata_field() -> None:
    """Incomplete Winston metadata must be treated as incompatible rather than inferred."""
    metadata = expected_metadata()
    del metadata["model_id"]
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="model_id"):
        await index.ensure_compatible(IDENTITY)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_missing_named_vector() -> None:
    """The configured named vector must exist in the actual Qdrant vector config."""
    client = FakeQdrantClient(exists=True, info=make_info(vector_name="other"))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="visual"):
        await index.ensure_compatible(IDENTITY)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_actual_vector_dimension_mismatch() -> None:
    """Actual Qdrant vector size wins over metadata claims of compatibility."""
    client = FakeQdrantClient(exists=True, info=make_info(size=512))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="size=768.*size=512"):
        await index.ensure_compatible(IDENTITY)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_actual_distance_mismatch() -> None:
    """Actual Qdrant distance must remain Cosine even when metadata claims cosine."""
    client = FakeQdrantClient(exists=True, info=make_info(distance=models.Distance.DOT))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="cosine.*dot"):
        await index.ensure_compatible(IDENTITY)


@pytest.mark.asyncio
async def test_ensure_compatible_wraps_qdrant_failures_with_original_cause() -> None:
    """Provider/network failures must cross the Winston boundary without losing their cause."""
    failure = RuntimeError("network down")
    client = FakeQdrantClient(exists=False, failure=failure)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(VisualIndexError, match="winston_visual") as caught:
        await index.ensure_compatible(IDENTITY)

    assert caught.value.__cause__ is failure


@pytest.mark.asyncio
async def test_close_releases_qdrant_client() -> None:
    """The reusable client must be closed when the index session ends."""
    client = FakeQdrantClient(exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    await index.close()

    assert client.closed
