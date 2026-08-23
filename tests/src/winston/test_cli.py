from pathlib import Path

import pytest

from winston.cli import main
from winston.config import Settings
from winston.indexing.models import AssetFailure, IndexingRunError, IndexRunResult, PipelineStage
from winston.ingest.models import ImageMetadata, MediaFile, MediaType, VideoMetadata


def test_scan_command_prints_video_and_image_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The existing scan command must retain its metadata output while index is added."""
    video = MediaFile(tmp_path / "camera.mkv", MediaType.VIDEO)
    image = MediaFile(tmp_path / "photo.jpg", MediaType.IMAGE)

    monkeypatch.setattr("winston.cli.scan_media", lambda _: [video, image])

    def fake_probe(media: MediaFile) -> VideoMetadata | ImageMetadata:
        """Return deterministic metadata for the existing scan output contract."""
        if media.media_type is MediaType.VIDEO:
            return VideoMetadata(media.path, 1920, 1080, "h264", 3723.5, 25.0)
        return ImageMetadata(media.path, 2048, 1536)

    monkeypatch.setattr("winston.cli.probe_media", fake_probe)

    assert main(["scan", str(tmp_path)]) == 0
    assert capsys.readouterr().out == (
        "VIDEO  camera.mkv\n"
        "       1920x1080\n"
        "       h264\n"
        "       25 fps\n"
        "       01:02:03.500\n"
        "IMAGE  photo.jpg\n"
        "       2048x1536\n"
    )


def test_index_command_passes_resolved_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The high-level index command must resolve the user path before orchestration."""
    seen: list[tuple[Path, Settings]] = []

    async def fake_run_indexing(root: Path, settings: Settings) -> IndexRunResult:
        """Capture the exact root/settings snapshot passed to asynchronous orchestration."""
        seen.append((root, settings))
        return IndexRunResult(indexed=1, skipped=0, failures=())

    monkeypatch.setattr("winston.cli.run_indexing", fake_run_indexing)

    assert main(["index", str(tmp_path)]) == 0
    assert len(seen) == 1
    assert seen[0][0] == tmp_path.resolve()


def test_index_command_uses_configured_data_dir_when_path_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The index path default and execution must use the same Settings snapshot."""
    configured = Settings(data_dir=tmp_path)
    seen: list[tuple[Path, Settings]] = []
    monkeypatch.setattr("winston.cli.get_config", lambda: configured)

    async def fake_run_indexing(root: Path, settings: Settings) -> IndexRunResult:
        """Capture the default path and prove the same settings object reaches execution."""
        seen.append((root, settings))
        return IndexRunResult(indexed=0, skipped=2, failures=())

    monkeypatch.setattr("winston.cli.run_indexing", fake_run_indexing)

    assert main(["index"]) == 0
    assert seen == [(tmp_path.resolve(), configured)]


def test_index_command_all_skipped_run_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A restart that finds only completed assets is a successful no-op."""
    async def fake_run_indexing(root: Path, settings: Settings) -> IndexRunResult:
        """Return one successful all-skipped run."""
        return IndexRunResult(indexed=0, skipped=4, failures=())

    monkeypatch.setattr("winston.cli.run_indexing", fake_run_indexing)

    assert main(["index", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "Indexed: 0\nSkipped: 4\nFailed: 0\n"


def test_index_command_prints_asset_failure_summary_and_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Asset failures must retain relative path, stage, and actionable message in CLI output."""
    failure = AssetFailure(
        source_path="cameras/cam12.mkv",
        stage=PipelineStage.SAMPLING,
        message="ffmpeg failed to decode keyframes",
        cause=RuntimeError("decoder error"),
    )

    async def fake_run_indexing(root: Path, settings: Settings) -> IndexRunResult:
        """Return a mixed run containing one isolated asset failure."""
        return IndexRunResult(indexed=3, skipped=2, failures=(failure,))

    monkeypatch.setattr("winston.cli.run_indexing", fake_run_indexing)

    assert main(["index", str(tmp_path)]) == 1
    assert capsys.readouterr().out == (
        "Indexed: 3\n"
        "Skipped: 2\n"
        "Failed: 1\n"
        "\n"
        "FAILED cameras/cam12.mkv [sampling]\n"
        "  ffmpeg failed to decode keyframes\n"
    )


def test_index_command_maps_fatal_run_error_to_stderr_and_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fatal initialization/cleanup errors must produce progress context and a concise failure."""
    async def fake_run_indexing(root: Path, settings: Settings) -> IndexRunResult:
        """Inject one fatal run-level failure."""
        raise IndexingRunError("manifest corrupt")

    monkeypatch.setattr("winston.cli.run_indexing", fake_run_indexing)

    assert main(["index", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"Winston index: {tmp_path.resolve()}\n"
        "Indexing failed: manifest corrupt\n"
    )
