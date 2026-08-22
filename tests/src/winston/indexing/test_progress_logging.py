"""Progress logging contract for the restartable indexing CLI."""

from collections.abc import AsyncIterator, Sequence
import logging
from pathlib import Path

import numpy as np
import pytest

from winston.cli import index_command
from winston.config import Settings
from winston.embeddings.base import RGBImage
from winston.embeddings.models import EmbeddingBatch, EmbeddingIdentity
from winston.index.models import IndexedVisual, VisualIndexSession
from winston.indexing.manifest import IndexManifest
from winston.indexing.models import IndexRunResult
from winston.indexing.pipeline import IndexingPipeline
from winston.ingest.identity import identify_asset
from winston.ingest.models import ImageMetadata, MediaFile, VideoMetadata
from winston.sampling.models import SampledFrame, SampledImage

IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)
INDEX_INSTANCE_ID = "3e755fec-acde-4b6c-a63e-1f7fc85a691c"


class ProgressEmbedder:
    """Minimal deterministic embedder used to exercise progress messages."""

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the fixed Jina CLIP identity used by indexing tests."""
        return IDENTITY

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        """Return one deterministic vector per input visual."""
        return EmbeddingBatch(
            vectors=np.zeros((len(images), IDENTITY.dimension), dtype=np.float32)
        )

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Reject text embedding because visual indexing never uses it."""
        raise AssertionError("text embeddings are not used by indexing progress tests")

    async def close(self) -> None:
        """Close the fake embedder without side effects."""


class ProgressVisualIndex:
    """Minimal visual index that accepts every operation for progress tests."""

    async def ensure_compatible(
        self,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> VisualIndexSession:
        """Return one stable collection instance after validating the expected identity."""
        assert identity == IDENTITY
        assert dataset_instance_id
        return VisualIndexSession(index_instance_id=INDEX_INSTANCE_ID)

    async def delete_old_revisions(
        self,
        *,
        source_path: str,
        current_asset_id: str,
    ) -> None:
        """Accept stale-revision cleanup without external storage."""
        assert source_path
        assert current_asset_id

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        """Accept one non-empty deterministic visual batch."""
        assert visuals

    async def close(self) -> None:
        """Close the fake index without side effects."""


class ProgressFrameSampler:
    """Stream a fixed frame sequence for one video."""

    def __init__(self, frames: tuple[SampledFrame, ...]) -> None:
        """Store the ordered frame sequence that should be yielded."""
        self._frames = frames

    def sample(self, video: VideoMetadata) -> AsyncIterator[SampledFrame]:
        """Return an async iterator over the configured frames."""
        return self._iterate(video)

    async def _iterate(self, video: VideoMetadata) -> AsyncIterator[SampledFrame]:
        """Yield frames in source order without buffering additional data."""
        assert video.path
        for frame in self._frames:
            yield frame


def _write_media(root: Path, name: str) -> Path:
    """Create one stable media placeholder for real scan and asset identity code."""
    path = root / name
    path.write_bytes(f"fixture:{name}".encode("utf-8"))
    return path


def _pipeline(
    root: Path,
    *,
    frame_sampler: ProgressFrameSampler | None = None,
) -> IndexingPipeline:
    """Build the real pipeline around minimal deterministic provider fakes."""
    return IndexingPipeline(
        root=root,
        settings=Settings(),
        embedder=ProgressEmbedder(),
        visual_index=ProgressVisualIndex(),
        frame_sampler=frame_sampler or ProgressFrameSampler(()),
    )


def _progress_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return user-facing Winston progress records in emission order."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "winston.progress"
    ]


@pytest.mark.asyncio
async def test_photo_logs_asset_and_batch_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A photo run reports discovery, decode, embedding, persistence, and completion."""
    path = _write_media(tmp_path, "photo.jpg")

    def fake_probe(media: MediaFile) -> ImageMetadata:
        """Return deterministic image geometry for the placeholder."""
        return ImageMetadata(path=media.path, width=4, height=4)

    def fake_load(metadata: ImageMetadata) -> SampledImage:
        """Return one RGB24 photo sample matching fake probe geometry."""
        return SampledImage(metadata.path, 4, 4, bytes(4 * 4 * 3))

    monkeypatch.setattr("winston.indexing.pipeline.probe_media", fake_probe)
    monkeypatch.setattr("winston.indexing.pipeline.load_image", fake_load)
    caplog.set_level(logging.INFO, logger="winston.progress")

    result = await _pipeline(tmp_path).run()

    assert result.failed == 0
    assert path.exists()
    messages = _progress_messages(caplog)
    assert "Found 1 media files" in messages
    assert "[1/1] IMAGE photo.jpg" in messages
    assert "      Removing stale revisions..." in messages
    assert "      Probing media..." in messages
    assert "      Decoding image..." in messages
    assert "        Embedding 5 regions..." in messages
    assert "        Upserting 5 vectors..." in messages
    assert "      Completed: 5 vectors" in messages


@pytest.mark.asyncio
async def test_video_logs_each_keyframe_with_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A video reports every streamed keyframe so long-running inference remains observable."""
    path = _write_media(tmp_path, "camera.mkv")
    frames = (
        SampledFrame(path, 1.25, 4, 4, bytes(4 * 4 * 3)),
        SampledFrame(path, 3.5, 4, 4, bytes(4 * 4 * 3)),
    )

    def fake_probe(media: MediaFile) -> VideoMetadata:
        """Return deterministic video metadata for the placeholder."""
        return VideoMetadata(media.path, 4, 4, "h264", 10.0, 25.0)

    monkeypatch.setattr("winston.indexing.pipeline.probe_media", fake_probe)
    caplog.set_level(logging.INFO, logger="winston.progress")

    result = await _pipeline(
        tmp_path,
        frame_sampler=ProgressFrameSampler(frames),
    ).run()

    assert result.failed == 0
    messages = _progress_messages(caplog)
    assert "[1/1] VIDEO camera.mkv" in messages
    assert "      Sampling keyframes..." in messages
    assert "      Keyframe 1 @ 00:00:01.250" in messages
    assert "      Keyframe 2 @ 00:00:03.500" in messages
    assert "      Completed: 10 vectors" in messages


@pytest.mark.asyncio
async def test_completed_asset_logs_skip(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A completed asset is visibly reported as skipped instead of appearing to do nothing."""
    path = _write_media(tmp_path, "photo.jpg")
    asset = identify_asset(index_root=tmp_path, source_path=path)
    IndexManifest(tmp_path).mark_completed(
        index_instance_id=INDEX_INSTANCE_ID,
        asset_id=asset.asset_id,
        source_path=asset.source_path,
    )
    caplog.set_level(logging.INFO, logger="winston.progress")

    result = await _pipeline(tmp_path).run()

    assert (result.indexed, result.skipped, result.failed) == (0, 1, 0)
    assert "[1/1] SKIP photo.jpg - already indexed" in _progress_messages(caplog)


def test_index_command_enables_progress_logging_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The index CLI surfaces Winston INFO progress without enabling third-party INFO noise."""
    async def fake_run_indexing(root: Path, settings: Settings) -> IndexRunResult:
        """Emit the same named progress logger used by the real pipeline."""
        assert root == tmp_path.resolve()
        logging.getLogger("winston.progress").info("progress message")
        return IndexRunResult(indexed=1, skipped=0, failures=())

    monkeypatch.setattr("winston.cli.run_indexing", fake_run_indexing)

    exit_code = index_command(tmp_path, Settings())

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Winston index:" in captured.err
    assert "progress message" in captured.err
    assert "Indexed: 1" in captured.out
