import gc
import weakref
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from winston.config import Settings
from winston.embeddings.base import RGBImage
from winston.embeddings.models import EmbeddingBatch, EmbeddingIdentity
from winston.index.models import (
    IndexedVisual,
    RegionGeometry,
    SampleKind,
    ScoredVisual,
    VisualSearchSession,
    VisualVector,
)
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind
from winston.search.pipeline import SearchPipeline

IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=3,
    preprocessing_version=1,
)
QUERY_VECTOR = np.array([1.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class WindowCall:
    asset_id: str
    start_timestamp_us: int
    end_timestamp_us: int


class FakeEmbedder:
    @property
    def identity(self) -> EmbeddingIdentity:
        return IDENTITY

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        raise AssertionError("semantic search must not embed images")

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        assert len(texts) == 1
        return EmbeddingBatch(vectors=np.asarray([QUERY_VECTOR], dtype=np.float32))

    async def close(self) -> None:
        return None


class PagedFakeVisualIndex:
    """Mimic Qdrant top-K semantics instead of returning every configured hit."""

    def __init__(
        self,
        *,
        coarse: Sequence[ScoredVisual],
        timeline: dict[WindowCall, tuple[IndexedVisual, ...]],
    ) -> None:
        self.coarse = tuple(coarse)
        self.timeline = timeline
        self.search_limits: list[int] = []

    async def open_search(self, identity: EmbeddingIdentity) -> VisualSearchSession:
        assert identity == IDENTITY
        return VisualSearchSession(
            dataset_instance_id="52cc120e-91c1-4114-92f1-007afb735f97",
            index_instance_id="8b97abb3-35a4-4724-86af-f5dadc88c5d9",
        )

    async def search_visuals(
        self,
        query_vector: VisualVector,
        *,
        limit: int,
    ) -> Sequence[ScoredVisual]:
        assert np.array_equal(query_vector, QUERY_VECTOR)
        self.search_limits.append(limit)
        return self.coarse[:limit]

    async def iter_visuals(
        self,
        *,
        asset_id: str,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> AsyncIterator[IndexedVisual]:
        call = WindowCall(asset_id, start_timestamp_us, end_timestamp_us)
        for visual in self.timeline.get(call, ()):
            yield visual


class MemoryProbeVisualIndex(PagedFakeVisualIndex):
    """Detect whether a losing region vector survives beyond the next timestamp."""

    def __init__(self, *, coarse: Sequence[ScoredVisual]) -> None:
        super().__init__(coarse=coarse, timeline={})
        self.losing_region_was_retained = False

    async def iter_visuals(
        self,
        *,
        asset_id: str,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> AsyncIterator[IndexedVisual]:
        previous_loser_vector: weakref.ReferenceType[np.ndarray] | None = None
        for timestamp_seconds in (6.0, 8.0, 10.0, 12.0, 14.0):
            winner = _video_visual(
                "a",
                "videos/event.mkv",
                timestamp_seconds,
                (1.0, 0.0, 0.0),
                region_kind=RegionKind.FULL,
            )
            yield winner

            if previous_loser_vector is not None:
                gc.collect()
                if previous_loser_vector() is not None:
                    self.losing_region_was_retained = True

            loser = _video_visual(
                "a",
                "videos/event.mkv",
                timestamp_seconds,
                (0.0, 1.0, 0.0),
                region_kind=RegionKind.TILE,
                tile_index=1,
            )
            yield loser
            previous_loser_vector = weakref.ref(loser.vector)
            del loser


def _settings(
    *,
    candidate_limit: int,
    candidate_max_limit: int | None = None,
    context_seconds: float = 5.0,
) -> Settings:
    search_config: dict[str, int | float] = {
        "result_limit": 10,
        "candidate_limit": candidate_limit,
        "temporal_context_seconds": context_seconds,
        "moving_average_frames": 3,
        "timeline_page_size": 2,
    }
    if candidate_max_limit is not None:
        search_config["candidate_max_limit"] = candidate_max_limit
    return Settings(search=search_config)


def _geometry(region_kind: RegionKind, tile_index: int = 0) -> RegionGeometry:
    if region_kind is RegionKind.FULL:
        return RegionGeometry(x=0, y=0, width=1920, height=1080, scale=1.0)
    return RegionGeometry(
        x=tile_index * 100,
        y=tile_index * 50,
        width=960,
        height=540,
        scale=0.5,
    )


def _video_visual(
    asset_character: str,
    source_path: str,
    timestamp_seconds: float,
    vector: tuple[float, float, float],
    *,
    region_kind: RegionKind = RegionKind.FULL,
    tile_index: int = 0,
) -> IndexedVisual:
    return IndexedVisual(
        asset_id=asset_character * 64,
        source_path=source_path,
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=timestamp_seconds,
        region_kind=region_kind,
        region=_geometry(region_kind, tile_index),
        vector=np.asarray(vector, dtype=np.float32),
        embedding_identity=IDENTITY,
    )


def _video_seed(
    asset_character: str,
    source_path: str,
    timestamp_seconds: float,
    score: float,
    *,
    region_kind: RegionKind = RegionKind.FULL,
    tile_index: int = 0,
) -> ScoredVisual:
    return ScoredVisual(
        visual=_video_visual(
            asset_character,
            source_path,
            timestamp_seconds,
            (1.0, 0.0, 0.0),
            region_kind=region_kind,
            tile_index=tile_index,
        ),
        score=score,
    )


@pytest.mark.asyncio
async def test_pipeline_overfetches_until_distinct_video_neighborhoods_can_fill_limit() -> None:
    """Duplicate raw ANN hits from one event must not hide a second user-facing passage."""
    coarse = (
        _video_seed("a", "videos/event.mkv", 10.0, 0.95),
        _video_seed(
            "a",
            "videos/event.mkv",
            10.0,
            0.94,
            region_kind=RegionKind.TILE,
            tile_index=1,
        ),
        _video_seed("a", "videos/event.mkv", 11.0, 0.93),
        _video_seed("a", "videos/event.mkv", 100.0, 0.80),
    )
    first_window = WindowCall("a" * 64, 5_000_000, 16_000_000)
    second_window = WindowCall("a" * 64, 95_000_000, 105_000_000)
    index = PagedFakeVisualIndex(
        coarse=coarse,
        timeline={
            first_window: (
                _video_visual("a", "videos/event.mkv", 10.0, (1.0, 0.0, 0.0)),
            ),
            second_window: (
                _video_visual("a", "videos/event.mkv", 100.0, (0.8, 0.6, 0.0)),
            ),
        },
    )
    pipeline = SearchPipeline(
        settings=_settings(candidate_limit=3, candidate_max_limit=6),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    results = await pipeline.run("person", limit=2)

    assert index.search_limits == [3, 6]
    assert [result.representative_timestamp_seconds for result in results] == [10.0, 100.0]


@pytest.mark.asyncio
async def test_pipeline_stops_candidate_overfetch_at_configured_hard_limit() -> None:
    """Diversity recovery must stay bounded even when top ANN points all collapse together."""
    coarse = tuple(
        _video_seed("a", "videos/event.mkv", 10.0 + offset, 0.95 - offset / 100.0)
        for offset in range(10)
    )
    index = PagedFakeVisualIndex(coarse=coarse, timeline={})
    pipeline = SearchPipeline(
        settings=_settings(candidate_limit=2, candidate_max_limit=4, context_seconds=30.0),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    assert await pipeline.run("person", limit=2) == ()
    assert index.search_limits == [2, 4]


@pytest.mark.asyncio
async def test_pipeline_never_exceeds_candidate_hard_limit_for_large_result_request() -> None:
    """A large --limit value must not override the configured raw ANN safety bound."""
    coarse = tuple(
        _video_seed("a", "videos/event.mkv", 10.0 + offset, 0.95 - offset / 100.0)
        for offset in range(10)
    )
    index = PagedFakeVisualIndex(coarse=coarse, timeline={})
    pipeline = SearchPipeline(
        settings=_settings(candidate_limit=2, candidate_max_limit=4, context_seconds=30.0),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    assert await pipeline.run("person", limit=10) == ()
    assert index.search_limits == [4]


@pytest.mark.asyncio
async def test_pipeline_discards_non_winning_region_vectors_while_streaming_timeline() -> None:
    """Local refinement must retain at most the strongest scored region for each timestamp."""
    coarse = (_video_seed("a", "videos/event.mkv", 10.0, 0.9),)
    index = MemoryProbeVisualIndex(coarse=coarse)
    pipeline = SearchPipeline(
        settings=_settings(candidate_limit=1),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    results = await pipeline.run("person", limit=1)

    assert len(results) == 1
    assert not index.losing_region_was_retained
