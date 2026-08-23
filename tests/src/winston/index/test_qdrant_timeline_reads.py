from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import JsonValue
from qdrant_client import models

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import VisualIndexConfigurationError
from winston.index.qdrant import QdrantVisualIndex

IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=768,
    preprocessing_version=1,
)
DATASET_ID = "52cc120e-91c1-4114-92f1-007afb735f97"
INDEX_INSTANCE_ID = "8b97abb3-35a4-4724-86af-f5dadc88c5d9"
ASSET_ID = "b" * 64


type MetadataValue = str | int
type Metadata = dict[str, MetadataValue]
type DenseVectorOutput = list[float] | dict[str, list[float]] | None
type ScrollOffset = int | str | None


def _metadata() -> Metadata:
    """Return compatible schema-v2 collection metadata for timeline reads."""
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


def _payload(timestamp_seconds: float) -> dict[str, JsonValue]:
    """Build one keyframe payload at the requested video timestamp."""
    return {
        "asset_id": ASSET_ID,
        "source_path": "cameras/luxembourg/cam02.mkv",
        "media_type": "video",
        "sample_kind": "keyframe",
        "timestamp_seconds": timestamp_seconds,
        "timestamp_us": int(timestamp_seconds * 1_000_000),
        "region_kind": "full",
        "region": {"x": 0, "y": 0, "width": 1280, "height": 720, "scale": 1.0},
        "model_id": IDENTITY.model_id,
        "dimension": IDENTITY.dimension,
        "preprocessing_version": IDENTITY.preprocessing_version,
    }


def _vector(position: int) -> dict[str, list[float]]:
    """Build one deterministic dense named vector for a fake stored keyframe."""
    values = [0.0] * 768
    values[position] = 1.0
    return {"visual": values}


@dataclass(slots=True)
class TimelineCollectionParams:
    """Minimal collection vector configuration consumed by open_search()."""

    vectors: dict[str, models.VectorParams]


@dataclass(slots=True)
class TimelineCollectionConfig:
    """Minimal collection config consumed by read-only compatibility checks."""

    params: TimelineCollectionParams
    metadata: Metadata


@dataclass(slots=True)
class TimelineCollectionInfo:
    """Minimal collection info consumed by the Qdrant adapter."""

    config: TimelineCollectionConfig


@dataclass(frozen=True, slots=True)
class FakeRecord:
    """One stored Qdrant record returned by a filtered scroll page."""

    payload: dict[str, JsonValue] | None
    vector: DenseVectorOutput


@dataclass(frozen=True, slots=True)
class ScrollPage:
    """One bounded fake scroll response page and its continuation offset."""

    records: list[FakeRecord]
    next_offset: ScrollOffset


@dataclass(frozen=True, slots=True)
class ScrollCall:
    """Captured Qdrant scroll arguments used to prove bounded filtered retrieval."""

    collection_name: str
    scroll_filter: models.Filter
    limit: int
    offset: ScrollOffset
    with_payload: bool
    with_vectors: list[str]


class TimelineQdrantClient:
    """Read-only fake serving deterministic filtered scroll pages."""

    def __init__(self, pages: list[ScrollPage]) -> None:
        """Configure the exact pages that successive scroll calls should return."""
        self.info = TimelineCollectionInfo(
            config=TimelineCollectionConfig(
                params=TimelineCollectionParams(
                    vectors={
                        "visual": models.VectorParams(
                            size=768,
                            distance=models.Distance.COSINE,
                        )
                    }
                ),
                metadata=_metadata(),
            )
        )
        self.pages = pages
        self.scroll_calls: list[ScrollCall] = []

    async def collection_exists(self, collection_name: str) -> bool:
        """Report the pre-existing collection required by read-only search."""
        return True

    async def get_collection(self, collection_name: str) -> TimelineCollectionInfo:
        """Return compatible collection metadata."""
        return self.info

    async def scroll(
        self,
        collection_name: str,
        *,
        scroll_filter: models.Filter,
        limit: int,
        offset: ScrollOffset,
        with_payload: bool,
        with_vectors: list[str],
    ) -> tuple[list[FakeRecord], ScrollOffset]:
        """Capture one page request and return the corresponding configured page."""
        self.scroll_calls.append(
            ScrollCall(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
        )
        page = self.pages[len(self.scroll_calls) - 1]
        return page.records, page.next_offset

    async def close(self) -> None:
        """Satisfy the reusable client lifecycle surface."""


@pytest.mark.asyncio
async def test_iter_visuals_scrolls_complete_window_page_by_page() -> None:
    """Timeline reads request only one asset/window and follow continuation offsets sequentially."""
    records = [
        FakeRecord(payload=_payload(8.0), vector=_vector(8)),
        FakeRecord(payload=_payload(12.0), vector=_vector(12)),
        FakeRecord(payload=_payload(16.0), vector=_vector(16)),
    ]
    client = TimelineQdrantClient(
        pages=[
            ScrollPage(records=records[:2], next_offset="page-2"),
            ScrollPage(records=records[2:], next_offset=None),
        ]
    )
    index = QdrantVisualIndex(
        QdrantSettings(),
        timeline_page_size=2,
        client=client,  # type: ignore[arg-type]
    )
    await index.open_search(IDENTITY)

    visuals = [
        visual
        async for visual in index.iter_visuals(
            asset_id=ASSET_ID,
            start_timestamp_us=5_000_000,
            end_timestamp_us=20_000_000,
        )
    ]

    assert [visual.timestamp_seconds for visual in visuals] == [8.0, 12.0, 16.0]
    assert all(visual.vector.dtype == np.float32 for visual in visuals)
    assert len(client.scroll_calls) == 2
    first, second = client.scroll_calls
    assert first.collection_name == "winston_visual"
    assert first.limit == 2
    assert first.offset is None
    assert first.with_payload is True
    assert first.with_vectors == ["visual"]
    assert second.offset == "page-2"

    must = first.scroll_filter.must
    assert must is not None
    assert must == [
        models.FieldCondition(
            key="asset_id",
            match=models.MatchValue(value=ASSET_ID),
        ),
        models.FieldCondition(
            key="media_type",
            match=models.MatchValue(value="video"),
        ),
        models.FieldCondition(
            key="timestamp_us",
            range=models.Range(gte=5_000_000, lte=20_000_000),
        ),
    ]


@pytest.mark.asyncio
async def test_iter_visuals_requires_read_open_before_scroll() -> None:
    """Filtered timeline reads cannot bypass collection compatibility validation."""
    client = TimelineQdrantClient(pages=[])
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(VisualIndexConfigurationError, match="open_search"):
        _ = [
            visual
            async for visual in index.iter_visuals(
                asset_id=ASSET_ID,
                start_timestamp_us=0,
                end_timestamp_us=1_000_000,
            )
        ]

    assert client.scroll_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("asset_id", "start_timestamp_us", "end_timestamp_us", "message"),
    [
        ("not-a-sha", 0, 1, "asset_id"),
        (ASSET_ID, -1, 1, "non-negative"),
        (ASSET_ID, 2, 1, "start.*end"),
    ],
)
async def test_iter_visuals_rejects_invalid_window_before_scroll(
    asset_id: str,
    start_timestamp_us: int,
    end_timestamp_us: int,
    message: str,
) -> None:
    """Malformed timeline bounds fail locally before any provider request is made."""
    client = TimelineQdrantClient(pages=[])
    index = QdrantVisualIndex(QdrantSettings(), client=client)  # type: ignore[arg-type]
    await index.open_search(IDENTITY)

    with pytest.raises(VisualIndexConfigurationError, match=message):
        _ = [
            visual
            async for visual in index.iter_visuals(
                asset_id=asset_id,
                start_timestamp_us=start_timestamp_us,
                end_timestamp_us=end_timestamp_us,
            )
        ]

    assert client.scroll_calls == []
