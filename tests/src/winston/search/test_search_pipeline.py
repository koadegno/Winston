import asyncio
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
    VisualIndexSession,
    VisualSearchSession,
    VisualVector,
)
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind
from winston.search import SearchRunError
from winston.search import pipeline as pipeline_module
from winston.search.pipeline import SearchPipeline, run_search

IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=3,
    preprocessing_version=1,
)
QUERY_VECTOR = np.array([1.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class WindowCall:
    """One exact bounded timeline request observed by the fake visual index."""

    asset_id: str
    start_timestamp_us: int
    end_timestamp_us: int


class FakeEmbedder:
    """Precise multimodal embedder fake that records text calls and lifecycle state."""

    def __init__(
        self,
        *,
        matrix: np.ndarray | None = None,
        close_error: Exception | None = None,
    ) -> None:
        """Configure deterministic text output and an optional cleanup failure."""
        self._identity = IDENTITY
        self.matrix = (
            np.asarray([QUERY_VECTOR], dtype=np.float32)
            if matrix is None
            else matrix
        )
        self.close_error = close_error
        self.text_calls: list[tuple[str, ...]] = []
        self.close_calls = 0

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the embedding identity used by all fake vectors."""
        return self._identity

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        """Reject image embedding because semantic search must embed text only."""
        raise AssertionError("semantic search must not call embed_images")

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Record the complete text batch and return the configured matrix."""
        self.text_calls.append(tuple(texts))
        return EmbeddingBatch(vectors=self.matrix)

    async def close(self) -> None:
        """Record closure and optionally raise the configured cleanup failure."""
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeVisualIndex:
    """Storage-neutral fake covering the complete VisualIndex protocol used by search."""

    def __init__(
        self,
        *,
        coarse: Sequence[ScoredVisual] = (),
        timeline: dict[WindowCall, tuple[IndexedVisual, ...]] | None = None,
        search_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        """Configure coarse hits, exact window contents, and optional failures."""
        self.coarse = tuple(coarse)
        self.timeline = {} if timeline is None else timeline
        self.search_error = search_error
        self.close_error = close_error
        self.open_identities: list[EmbeddingIdentity] = []
        self.search_limits: list[int] = []
        self.search_vectors: list[VisualVector] = []
        self.window_calls: list[WindowCall] = []
        self.active_iterators = 0
        self.max_active_iterators = 0
        self.close_calls = 0

    async def ensure_compatible(
        self,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> VisualIndexSession:
        """Reject write-mode opening because search must stay read-only."""
        raise AssertionError("semantic search must not call ensure_compatible")

    async def open_search(self, identity: EmbeddingIdentity) -> VisualSearchSession:
        """Record the identity used to validate the existing collection."""
        self.open_identities.append(identity)
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
        """Record one coarse ANN call and return configured candidates."""
        self.search_limits.append(limit)
        self.search_vectors.append(query_vector.copy())
        if self.search_error is not None:
            raise self.search_error
        return self.coarse

    async def iter_visuals(
        self,
        *,
        asset_id: str,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> AsyncIterator[IndexedVisual]:
        """Yield one configured window while detecting accidental concurrent refinement."""
        call = WindowCall(asset_id, start_timestamp_us, end_timestamp_us)
        self.window_calls.append(call)
        self.active_iterators += 1
        self.max_active_iterators = max(
            self.max_active_iterators,
            self.active_iterators,
        )
        try:
            for visual in self.timeline.get(call, ()):
                await asyncio.sleep(0)
                yield visual
        finally:
            self.active_iterators -= 1

    async def delete_old_revisions(
        self,
        *,
        source_path: str,
        current_asset_id: str,
    ) -> None:
        """Reject write operations during semantic search."""
        raise AssertionError("semantic search must not delete revisions")

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        """Reject write operations during semantic search."""
        raise AssertionError("semantic search must not upsert")

    async def close(self) -> None:
        """Record closure and optionally raise the configured cleanup failure."""
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _geometry(region_kind: RegionKind, tile_index: int = 0) -> RegionGeometry:
    """Build valid full-frame or deterministic tile geometry for result mapping tests."""
    if region_kind is RegionKind.FULL:
        return RegionGeometry(x=0, y=0, width=1920, height=1080, scale=1.0)
    return RegionGeometry(
        x=tile_index * 100,
        y=tile_index * 50,
        width=960,
        height=540,
        scale=0.5,
    )


def _image_match(
    asset_character: str,
    source_path: str,
    score: float,
    *,
    region_kind: RegionKind = RegionKind.FULL,
    tile_index: int = 0,
) -> ScoredVisual:
    """Build one strict coarse still-image match."""
    visual = IndexedVisual(
        asset_id=asset_character * 64,
        source_path=source_path,
        media_type=MediaType.IMAGE,
        sample_kind=SampleKind.IMAGE,
        timestamp_seconds=None,
        region_kind=region_kind,
        region=_geometry(region_kind, tile_index),
        vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        embedding_identity=IDENTITY,
    )
    return ScoredVisual(visual=visual, score=score)


def _video_visual(
    asset_character: str,
    source_path: str,
    timestamp_seconds: float,
    vector: tuple[float, float, float],
    *,
    region_kind: RegionKind = RegionKind.FULL,
    tile_index: int = 0,
) -> IndexedVisual:
    """Build one strict stored video visual used during exact local reranking."""
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
    """Build one coarse video hit whose ANN score is used only to seed a local window."""
    visual = _video_visual(
        asset_character,
        source_path,
        timestamp_seconds,
        (1.0, 0.0, 0.0),
        region_kind=region_kind,
        tile_index=tile_index,
    )
    return ScoredVisual(visual=visual, score=score)


def _settings(
    *,
    candidate_limit: int = 5,
    context_seconds: float = 5.0,
    moving_average_frames: int = 3,
) -> Settings:
    """Build focused search settings without changing unrelated application defaults."""
    return Settings(
        search={
            "result_limit": 10,
            "candidate_limit": candidate_limit,
            "temporal_context_seconds": context_seconds,
            "moving_average_frames": moving_average_frames,
            "timeline_page_size": 2,
        }
    )


@pytest.mark.asyncio
async def test_pipeline_embeds_query_once_deduplicates_photos_and_refines_video_window() -> None:
    """One query drives one ANN discovery, photo dedupe, then exact bounded video refinement."""
    coarse = [
        _image_match("c", "photos/door.jpg", 0.72),
        _image_match(
            "c",
            "photos/door.jpg",
            0.75,
            region_kind=RegionKind.TILE,
            tile_index=1,
        ),
        _video_seed("d", "videos/crossing.mkv", 10.0, 0.4),
        _video_seed(
            "d",
            "videos/crossing.mkv",
            10.0,
            0.6,
            region_kind=RegionKind.TILE,
            tile_index=2,
        ),
        _video_seed("d", "videos/crossing.mkv", 14.0, 0.5),
    ]
    window = WindowCall("d" * 64, 5_000_000, 19_000_000)
    timeline = {
        window: (
            _video_visual("d", "videos/crossing.mkv", 8.0, (0.6, 0.8, 0.0)),
            _video_visual("d", "videos/crossing.mkv", 12.0, (0.6, 0.8, 0.0)),
            _video_visual(
                "d",
                "videos/crossing.mkv",
                12.0,
                (0.8, 0.6, 0.0),
                region_kind=RegionKind.TILE,
                tile_index=3,
            ),
        )
    }
    embedder = FakeEmbedder()
    index = FakeVisualIndex(coarse=coarse, timeline=timeline)
    pipeline = SearchPipeline(
        settings=_settings(),
        embedder=embedder,
        visual_index=index,
    )

    results = await pipeline.run("woman with stroller", limit=2)

    assert embedder.text_calls == [("woman with stroller",)]
    assert index.open_identities == [IDENTITY]
    assert index.search_limits == [5]
    assert len(index.search_vectors) == 1
    assert np.array_equal(index.search_vectors[0], QUERY_VECTOR)
    assert index.window_calls == [window]
    assert index.max_active_iterators == 1

    assert len(results) == 2
    video, image = results
    assert video.source_path == "videos/crossing.mkv"
    assert video.media_type is MediaType.VIDEO
    assert video.start_timestamp_seconds == pytest.approx(12.0)
    assert video.end_timestamp_seconds == pytest.approx(12.0)
    assert video.representative_timestamp_seconds == pytest.approx(12.0)
    assert video.region_kind is RegionKind.TILE
    assert video.region.x == 300
    assert video.raw_score == pytest.approx(0.8)

    assert image.source_path == "photos/door.jpg"
    assert image.media_type is MediaType.IMAGE
    assert image.start_timestamp_seconds is None
    assert image.region_kind is RegionKind.TILE
    assert image.raw_score == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_pipeline_uses_requested_limit_when_it_exceeds_candidate_limit() -> None:
    """Coarse discovery always uses max(candidate_limit, requested result limit)."""
    embedder = FakeEmbedder()
    index = FakeVisualIndex()
    pipeline = SearchPipeline(
        settings=_settings(candidate_limit=3),
        embedder=embedder,
        visual_index=index,
    )

    assert await pipeline.run("red car", limit=7) == ()

    assert index.search_limits == [7]


@pytest.mark.asyncio
async def test_pipeline_applies_result_limit_after_photo_grouping() -> None:
    """Duplicate high-scoring regions cannot consume result slots before asset grouping."""
    index = FakeVisualIndex(
        coarse=(
            _image_match("a", "photos/a.jpg", 0.95),
            _image_match(
                "a",
                "photos/a.jpg",
                0.94,
                region_kind=RegionKind.TILE,
                tile_index=1,
            ),
            _image_match("b", "photos/b.jpg", 0.70),
        )
    )
    pipeline = SearchPipeline(
        settings=_settings(candidate_limit=10),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    results = await pipeline.run("person", limit=2)

    assert [result.source_path for result in results] == ["photos/a.jpg", "photos/b.jpg"]


@pytest.mark.asyncio
async def test_pipeline_refines_disjoint_video_windows_sequentially_and_sorts_time_ties() -> None:
    """Separate passages of one video are consumed sequentially and equal scores sort by earlier time."""
    coarse = (
        _video_seed("e", "videos/road.mkv", 10.0, 0.7),
        _video_seed("e", "videos/road.mkv", 100.0, 0.7),
    )
    early = WindowCall("e" * 64, 5_000_000, 15_000_000)
    late = WindowCall("e" * 64, 95_000_000, 105_000_000)
    timeline = {
        early: (_video_visual("e", "videos/road.mkv", 10.0, (0.8, 0.6, 0.0)),),
        late: (_video_visual("e", "videos/road.mkv", 100.0, (0.8, 0.6, 0.0)),),
    }
    index = FakeVisualIndex(coarse=coarse, timeline=timeline)
    pipeline = SearchPipeline(
        settings=_settings(),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    results = await pipeline.run("blue vehicle", limit=10)

    assert index.window_calls == [early, late]
    assert index.max_active_iterators == 1
    assert [result.start_timestamp_seconds for result in results] == [10.0, 100.0]
    assert [result.raw_score for result in results] == pytest.approx([0.8, 0.8])


@pytest.mark.asyncio
async def test_pipeline_sorts_equal_scores_by_source_path() -> None:
    """Global result ties are deterministic: raw score desc, then source path asc."""
    index = FakeVisualIndex(
        coarse=(
            _image_match("b", "photos/z.jpg", 0.5),
            _image_match("a", "photos/a.jpg", 0.5),
        )
    )
    pipeline = SearchPipeline(
        settings=_settings(),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    results = await pipeline.run("door", limit=10)

    assert [result.source_path for result in results] == ["photos/a.jpg", "photos/z.jpg"]


@pytest.mark.asyncio
async def test_pipeline_returns_empty_tuple_without_timeline_reads() -> None:
    """A valid query with no ANN candidates is a successful empty search."""
    index = FakeVisualIndex()
    pipeline = SearchPipeline(
        settings=_settings(),
        embedder=FakeEmbedder(),
        visual_index=index,
    )

    assert await pipeline.run("nothing here", limit=10) == ()
    assert index.window_calls == []


@pytest.mark.asyncio
async def test_pipeline_rejects_invalid_query_and_limit() -> None:
    """Direct pipeline use rejects blank queries and non-positive limits before embedding."""
    embedder = FakeEmbedder()
    index = FakeVisualIndex()
    pipeline = SearchPipeline(
        settings=_settings(),
        embedder=embedder,
        visual_index=index,
    )

    with pytest.raises(SearchRunError, match="query"):
        await pipeline.run("   ", limit=1)
    with pytest.raises(SearchRunError, match="limit"):
        await pipeline.run("person", limit=0)

    assert embedder.text_calls == []
    assert index.open_identities == []


@pytest.mark.asyncio
async def test_run_search_builds_one_stack_and_closes_both_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One command constructs one embedder/index pair and closes each exactly once on success."""
    embedder = FakeEmbedder()
    index = FakeVisualIndex()
    embedder_factory_calls: list[Settings] = []
    index_factory_calls: list[tuple[str, int]] = []

    def create_fake_embedder(settings: Settings) -> FakeEmbedder:
        """Record one embedder construction."""
        embedder_factory_calls.append(settings)
        return embedder

    def create_fake_index(
        qdrant_settings: pipeline_module.QdrantSettings,
        *,
        timeline_page_size: int,
    ) -> FakeVisualIndex:
        """Record one visual-index construction including the bounded timeline page size."""
        index_factory_calls.append((qdrant_settings.collection, timeline_page_size))
        return index

    monkeypatch.setattr(pipeline_module, "create_embedder", create_fake_embedder)
    monkeypatch.setattr(pipeline_module, "QdrantVisualIndex", create_fake_index)
    settings = _settings()

    assert await run_search("cat", limit=2, settings=settings) == ()

    assert embedder_factory_calls == [settings]
    assert index_factory_calls == [("winston_visual", 2)]
    assert embedder.close_calls == 1
    assert index.close_calls == 1


@pytest.mark.asyncio
async def test_run_search_rejects_blank_query_before_expensive_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only user input fails before either provider-backed dependency is constructed."""
    create_calls: list[str] = []

    def forbidden_embedder(settings: Settings) -> FakeEmbedder:
        """Fail the test if blank-query validation occurs too late."""
        create_calls.append("embedder")
        return FakeEmbedder()

    monkeypatch.setattr(pipeline_module, "create_embedder", forbidden_embedder)

    with pytest.raises(SearchRunError, match="query"):
        await run_search("   ", limit=2, settings=_settings())

    assert create_calls == []


@pytest.mark.asyncio
async def test_run_search_preserves_primary_failure_and_attaches_cleanup_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider failure remains primary while both cleanup failures are retained as exception notes."""
    primary = RuntimeError("coarse retrieval exploded")
    embedder = FakeEmbedder(close_error=RuntimeError("embedder cleanup exploded"))
    index = FakeVisualIndex(
        search_error=primary,
        close_error=RuntimeError("index cleanup exploded"),
    )

    monkeypatch.setattr(pipeline_module, "create_embedder", lambda settings: embedder)
    monkeypatch.setattr(
        pipeline_module,
        "QdrantVisualIndex",
        lambda qdrant_settings, *, timeline_page_size: index,
    )

    with pytest.raises(RuntimeError, match="coarse retrieval exploded") as caught:
        await run_search("cat", limit=2, settings=_settings())

    assert caught.value is primary
    assert index.close_calls == 1
    assert embedder.close_calls == 1
    assert getattr(caught.value, "__notes__", ()) == [
        "visual index close failed: index cleanup exploded",
        "embedder close failed: embedder cleanup exploded",
    ]
