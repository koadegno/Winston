from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from winston.config import Settings
from winston.embeddings.base import RGBImage
from winston.embeddings.models import EmbeddingBatch, EmbeddingIdentity
from winston.index.identity import visual_point_id
from winston.index.models import IndexedVisual, VisualIndexError, VisualIndexSession
from winston.indexing.manifest import IndexManifest, ManifestError
from winston.indexing.models import IndexingRunError, PipelineStage
from winston.indexing.pipeline import IndexingPipeline, run_indexing
from winston.ingest.identity import identify_asset
from winston.ingest.models import ImageMetadata, MediaFile, MediaType, VideoMetadata
from winston.ingest.probe import MediaProbeError
from winston.sampling.keyframes import FrameSamplingError
from winston.sampling.models import SampledFrame, SampledImage

IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)
INDEX_INSTANCE_ID = "ddbbd067-036e-4f45-a36c-69f702377c97"


class FakeEmbedder:
    """Deterministic embedder that records orchestration batch sizes and lifecycle."""

    def __init__(self, *, close_failure: Exception | None = None) -> None:
        """Configure optional cleanup failure for lifecycle tests."""
        self.image_batch_sizes: list[int] = []
        self.closed = False
        self.close_failure = close_failure

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the fixed Phase 1E embedding identity."""
        return IDENTITY

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        """Return one ordered zero vector per input image."""
        self.image_batch_sizes.append(len(images))
        return EmbeddingBatch(
            vectors=np.zeros((len(images), IDENTITY.dimension), dtype=np.float32)
        )

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Reject text embedding because it is outside the visual indexing pipeline."""
        raise AssertionError("text embedding is not used by Phase 1E indexing")

    async def close(self) -> None:
        """Record cleanup and optionally raise the configured cleanup failure."""
        self.closed = True
        if self.close_failure is not None:
            raise self.close_failure


class FakeVisualIndex:
    """Storage fake that records compatibility, delete, upsert, and cleanup ordering."""

    def __init__(
        self,
        *,
        delete_failure_for: str | None = None,
        upsert_failures_remaining: int = 0,
        close_failure: Exception | None = None,
    ) -> None:
        """Configure selected provider failures while retaining every attempted write."""
        self.delete_failure_for = delete_failure_for
        self.upsert_failures_remaining = upsert_failures_remaining
        self.close_failure = close_failure
        self.events: list[str] = []
        self.upserted_batches: list[tuple[IndexedVisual, ...]] = []
        self.closed = False

    async def ensure_compatible(
        self,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> VisualIndexSession:
        """Accept one dataset-owned index session and expose a stable collection instance."""
        self.events.append("ensure")
        assert identity == IDENTITY
        assert dataset_instance_id
        return VisualIndexSession(index_instance_id=INDEX_INSTANCE_ID)

    async def delete_old_revisions(
        self,
        *,
        source_path: str,
        current_asset_id: str,
    ) -> None:
        """Record delete-before-index ordering and optionally fail one source path."""
        self.events.append(f"delete:{source_path}:{current_asset_id}")
        if source_path == self.delete_failure_for:
            raise VisualIndexError(f"delete failed for {source_path}")

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        """Capture every attempted deterministic write before any configured failure."""
        batch = tuple(visuals)
        self.upserted_batches.append(batch)
        source_path = batch[0].source_path if batch else "empty"
        self.events.append(f"upsert:{source_path}")
        if self.upsert_failures_remaining > 0:
            self.upsert_failures_remaining -= 1
            raise VisualIndexError("injected upsert failure")

    async def close(self) -> None:
        """Record cleanup and optionally raise the configured cleanup failure."""
        self.closed = True
        if self.close_failure is not None:
            raise self.close_failure


class FakeFrameSampler:
    """Streaming video sampler with deterministic per-path frames and optional second-frame failure."""

    def __init__(
        self,
        frames: dict[str, tuple[SampledFrame, ...]],
        *,
        fail_on_second_for: str | None = None,
    ) -> None:
        """Configure streamed frames keyed by basename and one optional sampling failure."""
        self.frames = frames
        self.fail_on_second_for = fail_on_second_for
        self.events: list[str] = []

    def sample(self, video: VideoMetadata) -> AsyncIterator[SampledFrame]:
        """Return an async iterator that exposes frame-by-frame consumption events."""
        return self._iterate(video)

    async def _iterate(self, video: VideoMetadata) -> AsyncIterator[SampledFrame]:
        """Yield configured frames without materializing any additional video state."""
        name = video.path.name
        self.events.append(f"start:{name}")
        try:
            for position, frame in enumerate(self.frames.get(name, ()), start=1):
                if position == 2 and name == self.fail_on_second_for:
                    raise FrameSamplingError(f"second frame failed for {name}")
                yield frame
        finally:
            self.events.append(f"end:{name}")


def _write_media(root: Path, name: str) -> Path:
    """Create one small stable file so real scan and asset identity code can run."""
    path = root / name
    path.write_bytes(f"fixture:{name}".encode("utf-8"))
    return path


def _install_image_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    width: int = 4,
    height: int = 4,
    probe_failure_for: str | None = None,
) -> None:
    """Replace external image probe/decode work while retaining real orchestration."""
    def fake_probe(media: MediaFile) -> ImageMetadata:
        """Return deterministic image metadata or the configured path-aware failure."""
        if media.path.name == probe_failure_for:
            raise MediaProbeError(f"probe failed for {media.path}")
        return ImageMetadata(path=media.path, width=width, height=height)

    def fake_load(metadata: ImageMetadata) -> SampledImage:
        """Return an exact RGB24 still sample matching fake probe metadata."""
        return SampledImage(
            source_path=metadata.path,
            width=metadata.width,
            height=metadata.height,
            rgb24=bytes(metadata.width * metadata.height * 3),
        )

    monkeypatch.setattr("winston.indexing.pipeline.probe_media", fake_probe)
    monkeypatch.setattr("winston.indexing.pipeline.load_image", fake_load)


def _pipeline(
    root: Path,
    *,
    embedder: FakeEmbedder | None = None,
    visual_index: FakeVisualIndex | None = None,
    frame_sampler: FakeFrameSampler | None = None,
) -> tuple[IndexingPipeline, FakeEmbedder, FakeVisualIndex]:
    """Build a pipeline around deterministic fakes while keeping real manifest/scan/identity code."""
    selected_embedder = embedder or FakeEmbedder()
    selected_index = visual_index or FakeVisualIndex()
    selected_sampler = frame_sampler or FakeFrameSampler({})
    return (
        IndexingPipeline(
            root=root,
            settings=Settings(),
            embedder=selected_embedder,
            visual_index=selected_index,
            frame_sampler=selected_sampler,
        ),
        selected_embedder,
        selected_index,
    )


@pytest.mark.asyncio
async def test_photo_reaches_regions_embeddings_and_qdrant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discovered photo must reach the shared visual pipeline and complete only after writes."""
    _write_media(tmp_path, "photo.jpg")
    _install_image_stubs(monkeypatch)
    pipeline, embedder, index = _pipeline(tmp_path)

    result = await pipeline.run()

    assert (result.indexed, result.skipped, result.failed) == (1, 0, 0)
    visuals = tuple(visual for batch in index.upserted_batches for visual in batch)
    assert visuals
    assert all(visual.media_type is MediaType.IMAGE for visual in visuals)
    assert all(visual.sample_kind.value == "image" for visual in visuals)
    assert all(visual.timestamp_seconds is None for visual in visuals)
    assert embedder.image_batch_sizes == [5]
    assert index.events[0] == "ensure"
    assert index.events[1].startswith("delete:photo.jpg:")
    asset = identify_asset(index_root=tmp_path, source_path=tmp_path / "photo.jpg")
    assert IndexManifest(tmp_path).completed_asset_ids(INDEX_INSTANCE_ID) == {asset.asset_id}


@pytest.mark.asyncio
async def test_video_streams_keyframes_in_source_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video keyframes must be consumed and persisted in sampler order without video-wide buffering."""
    path = _write_media(tmp_path, "camera.mkv")
    frames = (
        SampledFrame(path, 1.25, 4, 4, bytes(4 * 4 * 3)),
        SampledFrame(path, 3.5, 4, 4, bytes(4 * 4 * 3)),
    )
    sampler = FakeFrameSampler({"camera.mkv": frames})

    def fake_probe(media: MediaFile) -> VideoMetadata:
        """Return deterministic video metadata for the streamed fixture."""
        return VideoMetadata(media.path, 4, 4, "h264", 10.0, 25.0)

    monkeypatch.setattr("winston.indexing.pipeline.probe_media", fake_probe)
    pipeline, _, index = _pipeline(tmp_path, frame_sampler=sampler)

    result = await pipeline.run()

    assert result.failed == 0
    timestamps = [
        visual.timestamp_seconds
        for batch in index.upserted_batches
        for visual in batch
        if visual.region_kind.value == "full"
    ]
    assert timestamps == [1.25, 3.5]
    assert sampler.events == ["start:camera.mkv", "end:camera.mkv"]


@pytest.mark.asyncio
async def test_visual_batch_size_never_exceeds_nine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Region generation must stream into batches bounded by visual_batch_size=9."""
    _write_media(tmp_path, "large.jpg")
    _install_image_stubs(monkeypatch, width=100, height=100)
    pipeline, embedder, _ = _pipeline(tmp_path)

    result = await pipeline.run()

    assert result.failed == 0
    assert embedder.image_batch_sizes == [9, 1]
    assert max(embedder.image_batch_sizes) <= 9


@pytest.mark.asyncio
async def test_completed_asset_skips_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable completion must bypass delete, probe, decode, embedding, and upsert."""
    path = _write_media(tmp_path, "photo.jpg")
    asset = identify_asset(index_root=tmp_path, source_path=path)
    IndexManifest(tmp_path).mark_completed(
        index_instance_id=INDEX_INSTANCE_ID,
        asset_id=asset.asset_id,
        source_path=asset.source_path,
    )

    def forbidden_probe(media: MediaFile) -> ImageMetadata:
        """Fail loudly if a completed asset reaches the expensive probe stage."""
        raise AssertionError(f"probe must not run for {media.path}")

    monkeypatch.setattr("winston.indexing.pipeline.probe_media", forbidden_probe)
    pipeline, embedder, index = _pipeline(tmp_path)

    result = await pipeline.run()

    assert (result.indexed, result.skipped, result.failed) == (0, 1, 0)
    assert index.events == ["ensure"]
    assert embedder.image_batch_sizes == []


@pytest.mark.asyncio
async def test_interrupted_asset_is_retried_with_same_deterministic_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An asset without durable completion must retry and rewrite identical point identities."""
    _write_media(tmp_path, "photo.jpg")
    _install_image_stubs(monkeypatch)
    index = FakeVisualIndex(upsert_failures_remaining=1)
    first_pipeline, _, _ = _pipeline(tmp_path, visual_index=index)

    first = await first_pipeline.run()
    first_point_ids = [
        str(visual_point_id(visual))
        for batch in index.upserted_batches
        for visual in batch
    ]
    second_pipeline, _, _ = _pipeline(tmp_path, visual_index=index)
    second = await second_pipeline.run()
    second_point_ids = [
        str(visual_point_id(visual))
        for batch in index.upserted_batches[1:]
        for visual in batch
    ]

    assert (first.indexed, first.failed) == (0, 1)
    assert (second.indexed, second.failed) == (1, 0)
    assert first_point_ids == second_point_ids


@pytest.mark.asyncio
async def test_changed_revision_deletes_before_first_upsert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-completed revision is cleaned by source path before new vectors are written."""
    _write_media(tmp_path, "photo.jpg")
    _install_image_stubs(monkeypatch)
    pipeline, _, index = _pipeline(tmp_path)

    await pipeline.run()

    delete_position = next(i for i, event in enumerate(index.events) if event.startswith("delete:"))
    upsert_position = next(i for i, event in enumerate(index.events) if event.startswith("upsert:"))
    assert delete_position < upsert_position


@pytest.mark.asyncio
async def test_delete_failure_prevents_new_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale-revision delete failure must fail that asset before probe or any upsert."""
    _write_media(tmp_path, "photo.jpg")
    _install_image_stubs(monkeypatch)
    index = FakeVisualIndex(delete_failure_for="photo.jpg")
    pipeline, embedder, _ = _pipeline(tmp_path, visual_index=index)

    result = await pipeline.run()

    assert result.indexed == 0
    assert result.failed == 1
    assert result.failures[0].stage is PipelineStage.DELETE
    assert index.upserted_batches == []
    assert embedder.image_batch_sizes == []


@pytest.mark.asyncio
async def test_one_asset_failure_does_not_stop_next_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt media file must not prevent later discovered assets from being attempted."""
    _write_media(tmp_path, "a-broken.jpg")
    _write_media(tmp_path, "b-good.jpg")
    _install_image_stubs(monkeypatch, probe_failure_for="a-broken.jpg")
    pipeline, _, index = _pipeline(tmp_path)

    result = await pipeline.run()

    assert (result.indexed, result.skipped, result.failed) == (1, 0, 1)
    assert result.failures[0].source_path == "a-broken.jpg"
    assert result.failures[0].stage is PipelineStage.PROBE
    assert any(event.startswith("upsert:b-good.jpg") for event in index.events)


@pytest.mark.asyncio
async def test_manifest_append_failure_is_asset_failure_and_next_run_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-upsert journal failure must leave the asset retryable with deterministic writes."""
    _write_media(tmp_path, "photo.jpg")
    _install_image_stubs(monkeypatch)
    original_mark_completed = IndexManifest.mark_completed

    def failing_mark_completed(
        self: IndexManifest,
        *,
        index_instance_id: str,
        asset_id: str,
        source_path: str,
    ) -> None:
        """Inject a durable-state failure after Qdrant writes have succeeded."""
        raise ManifestError(f"manifest failed for {source_path}")

    monkeypatch.setattr(IndexManifest, "mark_completed", failing_mark_completed)
    index = FakeVisualIndex()
    first_pipeline, _, _ = _pipeline(tmp_path, visual_index=index)
    first = await first_pipeline.run()
    monkeypatch.setattr(IndexManifest, "mark_completed", original_mark_completed)
    second_pipeline, _, _ = _pipeline(tmp_path, visual_index=index)
    second = await second_pipeline.run()

    assert first.failed == 1
    assert first.failures[0].stage is PipelineStage.MANIFEST
    assert second.indexed == 1
    assert len(index.upserted_batches) == 2


@pytest.mark.asyncio
async def test_assets_are_processed_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second video must not start sampling before the first video's iterator is exhausted."""
    first_path = _write_media(tmp_path, "a.mkv")
    second_path = _write_media(tmp_path, "b.mkv")
    sampler = FakeFrameSampler(
        {
            "a.mkv": (SampledFrame(first_path, 1.0, 4, 4, bytes(48)),),
            "b.mkv": (SampledFrame(second_path, 1.0, 4, 4, bytes(48)),),
        }
    )

    def fake_probe(media: MediaFile) -> VideoMetadata:
        """Return matching metadata for both sequential video fixtures."""
        return VideoMetadata(media.path, 4, 4, "h264", 5.0, 25.0)

    monkeypatch.setattr("winston.indexing.pipeline.probe_media", fake_probe)
    pipeline, _, _ = _pipeline(tmp_path, frame_sampler=sampler)

    result = await pipeline.run()

    assert result.failed == 0
    assert sampler.events == ["start:a.mkv", "end:a.mkv", "start:b.mkv", "end:b.mkv"]


@pytest.mark.asyncio
async def test_sampling_failure_on_second_frame_reports_sampling_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetching each next frame must reset the failure stage to sampling after prior upserts."""
    path = _write_media(tmp_path, "camera.mkv")
    sampler = FakeFrameSampler(
        {
            "camera.mkv": (
                SampledFrame(path, 1.0, 4, 4, bytes(48)),
                SampledFrame(path, 2.0, 4, 4, bytes(48)),
            )
        },
        fail_on_second_for="camera.mkv",
    )

    def fake_probe(media: MediaFile) -> VideoMetadata:
        """Return matching video metadata for the failure-stage test."""
        return VideoMetadata(media.path, 4, 4, "h264", 5.0, 25.0)

    monkeypatch.setattr("winston.indexing.pipeline.probe_media", fake_probe)
    pipeline, _, _ = _pipeline(tmp_path, frame_sampler=sampler)

    result = await pipeline.run()

    assert result.failed == 1
    assert result.failures[0].stage is PipelineStage.SAMPLING


@pytest.mark.asyncio
async def test_run_indexing_closes_both_dependencies_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reusable embedding and Qdrant resources must close after an empty successful run."""
    embedder = FakeEmbedder()
    index = FakeVisualIndex()
    monkeypatch.setattr("winston.indexing.pipeline.create_embedder", lambda settings: embedder)
    monkeypatch.setattr("winston.indexing.pipeline.QdrantVisualIndex", lambda settings: index)

    result = await run_indexing(tmp_path, Settings())

    assert result.failed == 0
    assert embedder.closed
    assert index.closed


@pytest.mark.asyncio
async def test_run_indexing_closes_both_dependencies_after_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fatal pipeline initialization/execution failure must still release both dependencies."""
    embedder = FakeEmbedder()
    index = FakeVisualIndex()
    monkeypatch.setattr("winston.indexing.pipeline.create_embedder", lambda settings: embedder)
    monkeypatch.setattr("winston.indexing.pipeline.QdrantVisualIndex", lambda settings: index)

    async def failing_run(self: IndexingPipeline) -> object:
        """Inject one fatal error outside asset-level isolation."""
        raise ManifestError("fatal manifest corruption")

    monkeypatch.setattr(IndexingPipeline, "run", failing_run)

    with pytest.raises(ManifestError, match="fatal manifest corruption"):
        await run_indexing(tmp_path, Settings())

    assert embedder.closed
    assert index.closed


@pytest.mark.asyncio
async def test_run_indexing_cleanup_failure_becomes_run_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup-only failure must produce a non-successful run-level outcome."""
    embedder = FakeEmbedder()
    index = FakeVisualIndex(close_failure=VisualIndexError("qdrant close failed"))
    monkeypatch.setattr("winston.indexing.pipeline.create_embedder", lambda settings: embedder)
    monkeypatch.setattr("winston.indexing.pipeline.QdrantVisualIndex", lambda settings: index)

    with pytest.raises(IndexingRunError, match="qdrant close failed"):
        await run_indexing(tmp_path, Settings())

    assert embedder.closed
    assert index.closed


@pytest.mark.asyncio
async def test_run_indexing_cleanup_does_not_mask_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A primary fatal error remains raised while cleanup diagnostics are attached as notes."""
    embedder = FakeEmbedder(close_failure=RuntimeError("embedder close failed"))
    index = FakeVisualIndex(close_failure=VisualIndexError("index close failed"))
    monkeypatch.setattr("winston.indexing.pipeline.create_embedder", lambda settings: embedder)
    monkeypatch.setattr("winston.indexing.pipeline.QdrantVisualIndex", lambda settings: index)

    primary = ManifestError("primary failure")

    async def failing_run(self: IndexingPipeline) -> object:
        """Raise the exact primary exception so identity can be asserted after cleanup."""
        raise primary

    monkeypatch.setattr(IndexingPipeline, "run", failing_run)

    with pytest.raises(ManifestError) as caught:
        await run_indexing(tmp_path, Settings())

    assert caught.value is primary
    notes = getattr(caught.value, "__notes__", [])
    assert any("index close failed" in note for note in notes)
    assert any("embedder close failed" in note for note in notes)
