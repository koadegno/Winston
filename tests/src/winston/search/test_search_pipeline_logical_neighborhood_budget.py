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
ASSET_A = "a" * 64
ASSET_B = "b" * 64


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
        assert texts == ["person"]
        return EmbeddingBatch(vectors=np.asarray([QUERY_VECTOR], dtype=np.float32))

    async def close(self) -> None:
        return None


class BudgetProbeVisualIndex:
    """Expose top-K paging and record the complete local refinement budget."""

    def __init__(self, coarse: Sequence[ScoredVisual]) -> None:
        self.coarse = tuple(coarse)
        self.search_limits: list[int] = []
        self.window_calls: list[WindowCall] = []

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
        self.window_calls.append(WindowCall(asset_id, start_timestamp_us, end_timestamp_us))
        if asset_id == ASSET_A and start_timestamp_us <= 80_000_000 <= end_timestamp_us:
            yield _video_visual("a", "videos/a.mkv", 80.0)
        if asset_id == ASSET_B and start_timestamp_us <= 500_000_000 <= end_timestamp_us:
            yield _video_visual("b", "videos/b.mkv", 500.0)


def _video_visual(
    asset_character: str,
    source_path: str,
    timestamp_seconds: float,
) -> IndexedVisual:
    return IndexedVisual(
        asset_id=asset_character * 64,
        source_path=source_path,
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=timestamp_seconds,
        region_kind=RegionKind.FULL,
        region=RegionGeometry(x=0, y=0, width=1920, height=1080, scale=1.0),
        vector=QUERY_VECTOR.copy(),
        embedding_identity=IDENTITY,
    )


def _video_seed(
    asset_character: str,
    source_path: str,
    timestamp_seconds: float,
    score: float,
) -> ScoredVisual:
    return ScoredVisual(
        visual=_video_visual(asset_character, source_path, timestamp_seconds),
        score=score,
    )


def _settings() -> Settings:
    return Settings(
        search={
            "result_limit": 10,
            "candidate_limit": 8,
            "candidate_max_limit": 16,
            "timeline_page_size": 2,
            "temporal_context_seconds": 15.0,
            "temporal_max_window_seconds": 60.0,
            "moving_average_frames": 3,
        }
    )


@pytest.mark.asyncio
async def test_long_neighborhood_does_not_stop_overfetch_before_second_video() -> None:
    """Technical runtime capping must not inflate semantic diversity or total local scan work."""
    first_video = tuple(
        _video_seed(
            "a",
            "videos/a.mkv",
            float(timestamp),
            0.99 if timestamp == 80 else 0.90 - timestamp / 10_000.0,
        )
        for timestamp in range(30, 101, 10)
    )
    coarse = first_video + (
        _video_seed("b", "videos/b.mkv", 500.0, 0.70),
    )
    index = BudgetProbeVisualIndex(coarse)
    pipeline = SearchPipeline(
        settings=_settings(),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    results = await pipeline.run("person", limit=2)

    assert index.search_limits == [8, 16]
    assert [result.source_path for result in results] == ["videos/a.mkv", "videos/b.mkv"]
    assert [result.representative_timestamp_seconds for result in results] == [80.0, 500.0]

    first_video_calls = [call for call in index.window_calls if call.asset_id == ASSET_A]
    assert len(first_video_calls) == 1
    assert (
        first_video_calls[0].end_timestamp_us - first_video_calls[0].start_timestamp_us
        <= 60_000_000
    )
