import asyncio
from datetime import datetime, timezone
from pathlib import Path
import signal

import pytest

from scripts.stream_recorder import recorder
from scripts.stream_recorder.recorder import build_ffmpeg_command, output_path_for_hour


def test_output_path_is_organized_by_source_camera_and_utc_date(tmp_path: Path):
    """Recording paths encode source, camera, UTC hour, and slice start time."""
    when = datetime(2026, 8, 22, 10, 15, 0, 123456, tzinfo=timezone.utc)
    path = output_path_for_hour(
        tmp_path,
        "belgium/brussels/grand-place",
        "camera-001",
        when,
    )
    assert path == (
        tmp_path
        / "belgium/brussels/grand-place/cameras/camera-001/2026/08/22/"
        "2026-08-22T10-00-00Z__start-10-15-00-123456Z.mkv"
    )


def test_output_paths_do_not_collide_within_the_same_hour(tmp_path: Path):
    """Retries and restarts in one UTC hour must never reuse the same output path."""
    first = output_path_for_hour(
        tmp_path,
        "belgium/brussels/grand-place",
        "camera-001",
        datetime(2026, 8, 22, 10, 15, 0, 1, tzinfo=timezone.utc),
    )
    second = output_path_for_hour(
        tmp_path,
        "belgium/brussels/grand-place",
        "camera-001",
        datetime(2026, 8, 22, 10, 45, 0, 2, tzinfo=timezone.utc),
    )

    assert first != second
    assert first.parent == second.parent


def test_ffmpeg_command_stream_copies_and_stops_at_boundary(tmp_path: Path):
    """FFmpeg copies media without re-encoding and receives an explicit duration."""
    output = tmp_path / "out.mkv"
    command = build_ffmpeg_command(
        "https://example.test/live.m3u8",
        output,
        duration_seconds=123,
    )
    assert command[:2] == ["ffmpeg", "-hide_banner"]
    assert ["-c", "copy"] == command[command.index("-c"):command.index("-c") + 2]
    assert command[command.index("-t") + 1] == "123"
    assert command[-1] == str(output)


def test_ffmpeg_command_refuses_to_overwrite_existing_output(tmp_path: Path):
    """An unexpected filename collision must fail instead of deleting recorded video."""
    command = build_ffmpeg_command(
        "https://example.test/live.m3u8",
        tmp_path / "existing.mkv",
        duration_seconds=123,
    )

    assert "-n" in command
    assert "-y" not in command


@pytest.mark.asyncio
async def test_record_one_hour_slice_awaits_async_ffmpeg(monkeypatch, tmp_path: Path):
    """One slice awaits the asynchronously spawned FFmpeg process."""
    waited = False

    class Process:
        """Minimal async subprocess test double."""

        pid = 1234

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


@pytest.mark.asyncio
async def test_record_one_hour_slice_gracefully_stops_ffmpeg_when_cancelled(
    monkeypatch,
    tmp_path: Path,
):
    """Cancelling a recorder asks FFmpeg to finalize its MKV before task cancellation."""
    started = asyncio.Event()
    stopped = asyncio.Event()
    received_signals: list[int] = []

    class Process:
        """Async subprocess double that runs until it receives a stop signal."""

        pid = 5678

        async def wait(self) -> int:
            """Wait until the fake FFmpeg process has been asked to stop."""
            started.set()
            await stopped.wait()
            return 0

        def send_signal(self, value: int) -> None:
            """Record the graceful signal and let the process finish."""
            received_signals.append(value)
            stopped.set()

        def terminate(self) -> None:
            """Fail the test if graceful shutdown unexpectedly escalates."""
            raise AssertionError("terminate() should not be needed")

        def kill(self) -> None:
            """Fail the test if graceful shutdown unexpectedly escalates."""
            raise AssertionError("kill() should not be needed")

    process = Process()

    async def fake_create_subprocess_exec(*_args):
        """Return the long-running async subprocess test double."""
        return process

    monkeypatch.setattr(recorder.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    task = asyncio.create_task(
        recorder.record_one_hour_slice(
            "https://example.test/live.m3u8",
            tmp_path,
            "country/city/place",
            "camera-001",
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert received_signals == [signal.SIGINT]
