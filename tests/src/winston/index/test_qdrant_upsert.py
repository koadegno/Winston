from dataclasses import dataclass

import numpy as np
import pytest
from qdrant_client import models

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.identity import visual_point_id
from winston.index.models import (
    IndexedVisual,
    RegionGeometry,
    SampleKind,
    VisualIndexConfigurationError,
    VisualIndexError,
)
from winston.index.qdrant import QdrantVisualIndex
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind

IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)


@dataclass(frozen=True, slots=True)
class UpsertCall:
    """Captured Qdrant upsert arguments."""

    collection_name: str
    points: list[models.PointStruct]
    wait: bool


class UpsertFakeClient:
    """Async fake covering collection creation and point upsert without destructive methods."""

    def __init__(self, *, upsert_failure: Exception | None = None) -> None:
        """Create a missing-collection fake with optional point-write failure."""
        self.created = False
        self.upsert_failure = upsert_failure
        self.upsert_calls: list[UpsertCall] = []
        self.closed = False

    async def collection_exists(self, collection_name: str) -> bool:
        """Return whether the fake collection has been created."""
        return self.created

    async def create_collection(
        self,
        collection_name: str,
        *,
        vectors_config: dict[str, models.VectorParams],
        metadata: dict[str, object],
    ) -> bool:
        """Mark the collection as created."""
        self.created = True
        return True

    async def get_collection(self, collection_name: str) -> object:
        """Fail if a test unexpectedly tries to validate pre-existing collection state."""
        raise AssertionError(f"unexpected get_collection({collection_name})")

    async def upsert(
        self,
        collection_name: str,
        *,
        points: list[models.PointStruct],
        wait: bool,
    ) -> object:
        """Capture one write or raise the configured provider failure."""
        if self.upsert_failure is not None:
            raise self.upsert_failure
        self.upsert_calls.append(
            UpsertCall(
                collection_name=collection_name,
                points=points,
                wait=wait,
            )
        )
        return object()

    async def close(self) -> None:
        """Record client closure."""
        self.closed = True


def make_video_visual(
    *,
    timestamp_seconds: float = 123.456,
    embedding_identity: EmbeddingIdentity = IDENTITY,
) -> IndexedVisual:
    """Build one valid keyframe tile for Qdrant mapping tests."""
    return IndexedVisual(
        asset_id="c" * 64,
        source_path="cameras/brussels/cam01.mkv",
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=timestamp_seconds,
        region_kind=RegionKind.TILE,
        region=RegionGeometry(x=1008, y=567, width=1344, height=756, scale=0.5),
        vector=np.arange(embedding_identity.dimension, dtype=np.float32),
        embedding_identity=embedding_identity,
    )


def make_image_visual() -> IndexedVisual:
    """Build one valid full-frame photo for payload tests."""
    return IndexedVisual(
        asset_id="d" * 64,
        source_path="photos/front-door.jpg",
        media_type=MediaType.IMAGE,
        sample_kind=SampleKind.IMAGE,
        timestamp_seconds=None,
        region_kind=RegionKind.FULL,
        region=RegionGeometry(x=0, y=0, width=1920, height=1080, scale=1.0),
        vector=np.ones(768, dtype=np.float32),
        embedding_identity=IDENTITY,
    )


@pytest.mark.asyncio
async def test_upsert_requires_compatibility_establishment() -> None:
    """No point may be written before ensure_compatible establishes the collection identity."""
    client = UpsertFakeClient()
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(VisualIndexConfigurationError, match="ensure_compatible"):
        await index.upsert([make_video_visual()])

    assert client.upsert_calls == []


@pytest.mark.asyncio
async def test_upsert_rejects_mixed_identity_before_first_network_write() -> None:
    """A mixed embedding identity batch must fail atomically before any Qdrant upsert."""
    client = UpsertFakeClient()
    index = QdrantVisualIndex(QdrantSettings(), client=client)
    await index.ensure_compatible(IDENTITY)
    incompatible = EmbeddingIdentity("other-model", 768, 1)

    with pytest.raises(VisualIndexConfigurationError, match="embedding identity"):
        await index.upsert(
            [
                make_video_visual(timestamp_seconds=1.0),
                make_video_visual(timestamp_seconds=2.0, embedding_identity=incompatible),
            ]
        )

    assert client.upsert_calls == []


@pytest.mark.asyncio
async def test_upsert_empty_batch_performs_no_write_after_compatibility() -> None:
    """An empty batch is a no-op only after the index session is established."""
    client = UpsertFakeClient()
    index = QdrantVisualIndex(QdrantSettings(), client=client)
    await index.ensure_compatible(IDENTITY)

    await index.upsert([])

    assert client.upsert_calls == []


@pytest.mark.asyncio
async def test_upsert_maps_video_tile_to_named_vector_and_complete_payload() -> None:
    """A keyframe tile must preserve exact temporal/spatial provenance without raw media."""
    client = UpsertFakeClient()
    index = QdrantVisualIndex(QdrantSettings(), client=client)
    await index.ensure_compatible(IDENTITY)
    visual = make_video_visual()

    await index.upsert([visual])

    assert len(client.upsert_calls) == 1
    call = client.upsert_calls[0]
    assert call.collection_name == "winston_visual"
    assert call.wait is True
    assert len(call.points) == 1
    point = call.points[0]
    assert point.id == str(visual_point_id(visual))
    assert isinstance(point.vector, dict)
    assert set(point.vector) == {"visual"}
    assert point.vector["visual"] == visual.vector.tolist()
    assert point.payload == {
        "asset_id": "c" * 64,
        "source_path": "cameras/brussels/cam01.mkv",
        "media_type": "video",
        "sample_kind": "keyframe",
        "timestamp_seconds": 123.456,
        "timestamp_us": 123_456_000,
        "region_kind": "tile",
        "region": {
            "x": 1008,
            "y": 567,
            "width": 1344,
            "height": 756,
            "scale": 0.5,
        },
        "model_id": "jinaai/jina-clip-v1",
        "dimension": 768,
        "preprocessing_version": 1,
    }
    assert "rgb24" not in point.payload


@pytest.mark.asyncio
async def test_upsert_maps_photo_timestamp_fields_to_null() -> None:
    """A photo payload must explicitly carry null temporal fields and full-frame geometry."""
    client = UpsertFakeClient()
    index = QdrantVisualIndex(QdrantSettings(), client=client)
    await index.ensure_compatible(IDENTITY)
    visual = make_image_visual()

    await index.upsert([visual])

    payload = client.upsert_calls[0].points[0].payload
    assert payload is not None
    assert payload["media_type"] == "image"
    assert payload["sample_kind"] == "image"
    assert payload["timestamp_seconds"] is None
    assert payload["timestamp_us"] is None
    assert payload["region_kind"] == "full"
    assert payload["region"] == {
        "x": 0,
        "y": 0,
        "width": 1920,
        "height": 1080,
        "scale": 1.0,
    }


@pytest.mark.asyncio
async def test_upsert_splits_writes_into_bounded_sequential_batches() -> None:
    """The default write path must never materialize more than 256 Qdrant points at once."""
    client = UpsertFakeClient()
    index = QdrantVisualIndex(QdrantSettings(upsert_batch_size=256), client=client)
    await index.ensure_compatible(IDENTITY)
    visuals = [make_video_visual(timestamp_seconds=float(index)) for index in range(600)]

    await index.upsert(visuals)

    assert [len(call.points) for call in client.upsert_calls] == [256, 256, 88]
    assert all(call.wait is True for call in client.upsert_calls)


@pytest.mark.asyncio
async def test_upsert_uses_configured_collection_vector_and_batch_size() -> None:
    """Qdrant names and batching must come entirely from typed configuration."""
    client = UpsertFakeClient()
    settings = QdrantSettings(
        collection="custom_visual",
        vector_name="image",
        upsert_batch_size=2,
    )
    index = QdrantVisualIndex(settings, client=client)
    await index.ensure_compatible(IDENTITY)

    await index.upsert(
        [make_video_visual(timestamp_seconds=float(index)) for index in range(3)]
    )

    assert [len(call.points) for call in client.upsert_calls] == [2, 1]
    assert all(call.collection_name == "custom_visual" for call in client.upsert_calls)
    assert all(set(call.points[0].vector) == {"image"} for call in client.upsert_calls)


@pytest.mark.asyncio
async def test_upsert_wraps_provider_failure_with_original_cause() -> None:
    """Write failures must become Winston index errors while retaining the Qdrant cause."""
    failure = RuntimeError("write failed")
    client = UpsertFakeClient(upsert_failure=failure)
    index = QdrantVisualIndex(QdrantSettings(), client=client)
    await index.ensure_compatible(IDENTITY)

    with pytest.raises(VisualIndexError, match="upsert") as caught:
        await index.upsert([make_video_visual()])

    assert caught.value.__cause__ is failure
