from dataclasses import dataclass
from uuid import UUID

import pytest
from qdrant_client import models

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import (
    IncompatibleVisualIndexError,
    VisualIndexConfigurationError,
    VisualIndexError,
)
from winston.index.qdrant import QdrantVisualIndex

IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=768,
    preprocessing_version=1,
)
DATASET_ID = "8f7ad0c0-7ab7-4ec0-9025-22f40dd70c4a"
INDEX_INSTANCE_ID = "0f27f24e-7436-4c46-9260-4a37f419b3a6"


def expected_metadata(
    identity: EmbeddingIdentity = IDENTITY,
    *,
    dataset_instance_id: str = DATASET_ID,
    index_instance_id: str = INDEX_INSTANCE_ID,
) -> dict[str, object]:
    """Return the exact Phase 1E Winston collection metadata contract."""
    return {
        "winston_schema_version": 2,
        "dataset_instance_id": dataset_instance_id,
        "index_instance_id": index_instance_id,
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


@dataclass(frozen=True, slots=True)
class DeleteCall:
    """Captured stale-revision deletion arguments."""

    collection_name: str
    points_selector: models.FilterSelector
    wait: bool


class FakeQdrantClient:
    """Async fake implementing the Qdrant operations Winston may use."""

    def __init__(
        self,
        *,
        exists: bool,
        info: FakeCollectionInfo | None = None,
        failure: Exception | None = None,
        delete_failure: Exception | None = None,
    ) -> None:
        """Configure existence, collection info, provider failure, or delete failure."""
        self.exists = exists
        self.info = info
        self.failure = failure
        self.delete_failure = delete_failure
        self.create_calls: list[CreateCall] = []
        self.delete_calls: list[DeleteCall] = []
        self.closed = False

    async def collection_exists(self, collection_name: str) -> bool:
        """Return configured collection existence or raise the configured provider failure."""
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

    async def delete(
        self,
        collection_name: str,
        *,
        points_selector: models.FilterSelector,
        wait: bool,
    ) -> object:
        """Capture a deletion request or raise the configured failure."""
        if self.delete_failure is not None:
            raise self.delete_failure
        self.delete_calls.append(
            DeleteCall(
                collection_name=collection_name,
                points_selector=points_selector,
                wait=wait,
            )
        )
        return object()

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
async def test_ensure_compatible_creates_missing_collection_with_instance_id() -> None:
    """A missing collection must persist dataset ownership and one generated instance UUID."""
    client = FakeQdrantClient(exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    session = await index.ensure_compatible(IDENTITY, DATASET_ID)

    assert len(client.create_calls) == 1
    call = client.create_calls[0]
    assert call.collection_name == "winston_visual"
    assert set(call.vectors_config) == {"visual"}
    vector = call.vectors_config["visual"]
    assert vector.size == 768
    assert vector.distance is models.Distance.COSINE
    assert call.metadata["winston_schema_version"] == 2
    assert call.metadata["dataset_instance_id"] == DATASET_ID
    assert call.metadata["index_instance_id"] == session.index_instance_id
    assert str(UUID(session.index_instance_id)) == session.index_instance_id
    assert call.metadata["model_id"] == IDENTITY.model_id
    assert call.metadata["dimension"] == IDENTITY.dimension
    assert call.metadata["preprocessing_version"] == IDENTITY.preprocessing_version
    assert call.metadata["vector_name"] == "visual"
    assert call.metadata["distance"] == "cosine"


@pytest.mark.asyncio
async def test_ensure_compatible_returns_existing_collection_instance_id() -> None:
    """A fully matching collection must return its persisted instance identity without mutation."""
    client = FakeQdrantClient(exists=True, info=make_info())
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    session = await index.ensure_compatible(IDENTITY, DATASET_ID)

    assert session.index_instance_id == INDEX_INSTANCE_ID
    assert client.create_calls == []


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_collection_without_winston_metadata() -> None:
    """An unowned collection must never be silently adopted."""
    info = make_info()
    info.config.metadata = None
    client = FakeQdrantClient(exists=True, info=info)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="metadata"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)

    assert client.create_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("winston_schema_version", 1),
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
    """Every compatibility field other than the collection instance must match exactly."""
    metadata = expected_metadata()
    metadata[field] = value
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match=field):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_collection_owned_by_another_dataset() -> None:
    """A collection may never be shared by two independent dataset identities."""
    metadata = expected_metadata(
        dataset_instance_id="6fd3c642-165f-4743-a871-a16df5df7cc7"
    )
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="dataset_instance_id"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_schema_one_with_reindex_guidance() -> None:
    """A Phase 1D collection must be rejected rather than silently upgraded in place."""
    metadata = expected_metadata()
    metadata["winston_schema_version"] = 1
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="reindex"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_missing_index_instance_id() -> None:
    """Restart state is unsafe when the collection instance identity is absent."""
    metadata = expected_metadata()
    del metadata["index_instance_id"]
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="index_instance_id"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_invalid_index_instance_id() -> None:
    """The collection instance identity must be a canonicalizable UUID."""
    metadata = expected_metadata(index_instance_id="not-a-uuid")
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="index_instance_id"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_invalid_requested_dataset_id() -> None:
    """Caller-supplied dataset ownership must be validated before touching Qdrant."""
    client = FakeQdrantClient(exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(VisualIndexConfigurationError, match="dataset_instance_id"):
        await index.ensure_compatible(IDENTITY, "not-a-uuid")

    assert client.create_calls == []


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_missing_required_metadata_field() -> None:
    """Incomplete Winston metadata must be treated as incompatible rather than inferred."""
    metadata = expected_metadata()
    del metadata["model_id"]
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="model_id"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_missing_named_vector() -> None:
    """The configured named vector must exist in the actual Qdrant vector config."""
    client = FakeQdrantClient(exists=True, info=make_info(vector_name="other"))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="visual"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_actual_vector_dimension_mismatch() -> None:
    """Actual Qdrant vector size wins over metadata claims of compatibility."""
    client = FakeQdrantClient(exists=True, info=make_info(size=512))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="size=768.*size=512"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_actual_distance_mismatch() -> None:
    """Actual Qdrant distance must remain Cosine even when metadata claims cosine."""
    client = FakeQdrantClient(exists=True, info=make_info(distance=models.Distance.DOT))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="cosine.*dot"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_wraps_qdrant_failures_with_original_cause() -> None:
    """Provider/network failures must cross the Winston boundary without losing their cause."""
    failure = RuntimeError("network down")
    client = FakeQdrantClient(exists=False, failure=failure)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(VisualIndexError, match="winston_visual") as caught:
        await index.ensure_compatible(IDENTITY, DATASET_ID)

    assert caught.value.__cause__ is failure


@pytest.mark.asyncio
async def test_delete_old_revisions_requires_compatibility_establishment() -> None:
    """Revision cleanup must not run before collection ownership is validated."""
    client = FakeQdrantClient(exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(VisualIndexConfigurationError, match="ensure_compatible"):
        await index.delete_old_revisions(
            source_path="cameras/cam01.mkv",
            current_asset_id="a" * 64,
        )

    assert client.delete_calls == []


@pytest.mark.asyncio
async def test_delete_old_revisions_filters_same_path_and_excludes_current_asset() -> None:
    """Cleanup must remove only older revisions at the same normalized source path."""
    client = FakeQdrantClient(exists=True, info=make_info())
    index = QdrantVisualIndex(QdrantSettings(), client=client)
    await index.ensure_compatible(IDENTITY, DATASET_ID)

    await index.delete_old_revisions(
        source_path="cameras/cam01.mkv",
        current_asset_id="a" * 64,
    )

    assert len(client.delete_calls) == 1
    call = client.delete_calls[0]
    assert call.collection_name == "winston_visual"
    assert call.wait is True
    assert call.points_selector.filter.must == [
        models.FieldCondition(
            key="source_path",
            match=models.MatchValue(value="cameras/cam01.mkv"),
        )
    ]
    assert call.points_selector.filter.must_not == [
        models.FieldCondition(
            key="asset_id",
            match=models.MatchValue(value="a" * 64),
        )
    ]


@pytest.mark.asyncio
async def test_delete_old_revisions_wraps_provider_failure_with_original_cause() -> None:
    """Delete failures must be actionable while retaining the Qdrant cause."""
    failure = RuntimeError("delete failed")
    client = FakeQdrantClient(
        exists=True,
        info=make_info(),
        delete_failure=failure,
    )
    index = QdrantVisualIndex(QdrantSettings(), client=client)
    await index.ensure_compatible(IDENTITY, DATASET_ID)

    with pytest.raises(VisualIndexError, match="delete old visual revisions") as caught:
        await index.delete_old_revisions(
            source_path="cameras/cam01.mkv",
            current_asset_id="a" * 64,
        )

    assert caught.value.__cause__ is failure


@pytest.mark.asyncio
async def test_close_releases_qdrant_client() -> None:
    """The reusable client must be closed when the index session ends."""
    client = FakeQdrantClient(exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    await index.close()

    assert client.closed
