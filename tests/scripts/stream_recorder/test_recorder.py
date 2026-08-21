from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.stream_recorder import recorder
from scripts.stream_recorder.recorder import build_ffmpeg_command, output_path_for_hour


def test_output_path_is_organized_by_source_camera_and_utc_date(tmp_path: Path):
    """Recording paths encode source, camera, and UTC date hierarchy."""
    when = datetime(2026, 8, 22, 10, 15, tzinfo=timezone.utc)
    path = output_path_for_hour(tmp_path, "belgium/brussels/grand-place", "camera-001", when)
    assert path == tmp_path / "belgium/brussels/grand-place/cameras/camera-001/2026/08/22/2026-08-22T10-00-00Z.mkv"


def test_ffmpeg_command_stream_copies_and_stops_at_boundary(tmp_path: Path):
    """FFmpeg copies media without re-encoding and receives an explicit duration."""
    output = tmp_path / "out.mkv"
    command = build_ffmpeg_command("https://example.test/live.m3u8", output, duration_seconds=123)
    assert command[:2] == ["ffmpeg", "-hide_banner"]
    assert ["-c", "copy"] == command[command.index("-c"):command.index("-c") + 2]
    assert command[command.index("-t") + 1] == "123"
    assert command[-1] == str(output)


@pytest.mark.asyncio
async def test_record_one_hour_slice_awaits_async_ffmpeg(monkeypatch, tmp_path: Path):
    """One slice awaits the asynchronously spawned FFmpeg process."""
    waited = False

    class Process:
        """Minimal async subprocess test double."""

        async def wait(self) -> int:
            """Record that the process wait was awaited."""
            nonlocal waited
            waited = True
            return 7

    async def fake_create_subprocess_exec(*_args):
        """Return the async subprocess test double."""
        return Process()

    monkeypatch.setattr(recorder.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    result = await recorder.record_one_hour_slice(
        "https://example.test/live.m3u8",
        tmp_path,
        "country/city/place",
        "camera-001",
    )
    assert result == 7
    assert waited
