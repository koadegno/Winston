from pathlib import Path

import pytest

from winston.cli import main
from winston.config import Settings
from winston.index.models import RegionGeometry
from winston.indexing.models import AssetFailure, IndexingRunError, IndexRunResult, PipelineStage
from winston.ingest.models import ImageMetadata, MediaFile, MediaType, VideoMetadata
from winston.sampling.regions import RegionKind
from winston.search.models import SearchResult


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


def _image_search_result() -> SearchResult:
    """Build one still-image semantic result for CLI formatting tests."""
    return SearchResult(
        source_path="photos/door.jpg",
        media_type=MediaType.IMAGE,
        start_timestamp_seconds=None,
        end_timestamp_seconds=None,
        representative_timestamp_seconds=None,
        region_kind=RegionKind.TILE,
        region=RegionGeometry(x=100, y=50, width=960, height=540, scale=0.5),
        raw_score=0.75,
    )


def _video_search_result() -> SearchResult:
    """Build one temporal semantic result for CLI formatting tests."""
    return SearchResult(
        source_path="videos/crossing.mkv",
        media_type=MediaType.VIDEO,
        start_timestamp_seconds=12.0,
        end_timestamp_seconds=16.0,
        representative_timestamp_seconds=14.0,
        region_kind=RegionKind.TILE,
        region=RegionGeometry(x=200, y=100, width=960, height=540, scale=0.5),
        raw_score=0.8123456,
    )


def test_search_command_uses_configured_default_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting --limit uses the same nested Settings snapshot passed to search orchestration."""
    configured = Settings(search={"result_limit": 4})
    seen: list[tuple[str, int, Settings]] = []
    monkeypatch.setattr("winston.cli.get_config", lambda: configured)

    async def fake_run_search(
        query: str,
        *,
        limit: int,
        settings: Settings,
    ) -> tuple[SearchResult, ...]:
        """Capture the query, configured default limit, and exact settings object."""
        seen.append((query, limit, settings))
        return ()

    monkeypatch.setattr("winston.cli.run_search", fake_run_search)

    assert main(["search", "woman with a pink stroller"]) == 0
    assert seen == [("woman with a pink stroller", 4, configured)]


def test_search_command_accepts_explicit_positive_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive --limit overrides the configured result count before orchestration."""
    seen: list[int] = []

    async def fake_run_search(
        query: str,
        *,
        limit: int,
        settings: Settings,
    ) -> tuple[SearchResult, ...]:
        """Capture the explicit result limit."""
        seen.append(limit)
        return ()

    monkeypatch.setattr("winston.cli.run_search", fake_run_search)

    assert main(["search", "red car", "--limit", "7"]) == 0
    assert seen == [7]


def test_search_parser_rejects_non_positive_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """argparse rejects a zero result limit before command dispatch."""
    with pytest.raises(SystemExit) as caught:
        main(["search", "red car", "--limit", "0"])

    assert caught.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_search_command_rejects_blank_query_before_async_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Whitespace-only input returns exit 1 without constructing the asynchronous search stack."""
    calls: list[str] = []

    async def forbidden_run_search(
        query: str,
        *,
        limit: int,
        settings: Settings,
    ) -> tuple[SearchResult, ...]:
        """Record an invalid late dispatch if CLI query validation regresses."""
        calls.append(query)
        return ()

    monkeypatch.setattr("winston.cli.run_search", forbidden_run_search)

    assert main(["search", "   "]) == 1
    captured = capsys.readouterr()
    assert calls == []
    assert captured.out == ""
    assert captured.err == "Search failed: search query must not be blank\n"


def test_search_command_valid_empty_result_exits_zero_without_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No semantic matches is a successful empty search rather than an error."""
    async def fake_run_search(
        query: str,
        *,
        limit: int,
        settings: Settings,
    ) -> tuple[SearchResult, ...]:
        """Return a valid empty result set."""
        return ()

    monkeypatch.setattr("winston.cli.run_search", fake_run_search)

    assert main(["search", "empty street"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_search_command_formats_image_and_video_results_without_fake_confidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI exposes provenance, temporal passage, geometry, and raw cosine only."""
    async def fake_run_search(
        query: str,
        *,
        limit: int,
        settings: Settings,
    ) -> tuple[SearchResult, ...]:
        """Return one video and one image in already-ranked order."""
        return (_video_search_result(), _image_search_result())

    monkeypatch.setattr("winston.cli.run_search", fake_run_search)

    assert main(["search", "person crossing road"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "1. videos/crossing.mkv\n"
        "   passage: 00:00:12.000 -> 00:00:16.000\n"
        "   representative: 00:00:14.000\n"
        "   region: tile x=200 y=100 width=960 height=540 scale=0.5\n"
        "   raw score: 0.812346\n"
        "\n"
        "2. photos/door.jpg\n"
        "   region: tile x=100 y=50 width=960 height=540 scale=0.5\n"
        "   raw score: 0.750000\n"
    )
    lowered = captured.out.lower()
    assert "%" not in captured.out
    assert "confidence" not in lowered


def test_search_command_maps_failure_to_stderr_and_exit_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search initialization/provider failures produce one concise stderr diagnostic."""
    async def fake_run_search(
        query: str,
        *,
        limit: int,
        settings: Settings,
    ) -> tuple[SearchResult, ...]:
        """Inject one fatal semantic-search failure."""
        raise RuntimeError("Qdrant unavailable")

    monkeypatch.setattr("winston.cli.run_search", fake_run_search)

    assert main(["search", "red car"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Search failed: Qdrant unavailable\n"
