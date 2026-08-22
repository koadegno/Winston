"""Create hour-aligned, restart-safe MKV slices from live HLS streams."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import signal

from .log import log


FFMPEG_GRACEFUL_SHUTDOWN_SECONDS = 5.0
FFMPEG_KILL_SHUTDOWN_SECONDS = 2.0


def hour_start(moment: datetime) -> datetime:
    """Return the UTC clock-hour boundary containing ``moment``."""
    moment = moment.astimezone(timezone.utc)
    return moment.replace(minute=0, second=0, microsecond=0)


def seconds_until_next_hour(moment: datetime) -> int:
    """Return whole seconds remaining until the next UTC clock hour."""
    current = moment.astimezone(timezone.utc)
    next_hour = hour_start(current) + timedelta(hours=1)
    return max(1, int((next_hour - current).total_seconds()))


def output_path_for_hour(
    root: Path,
    source_slug: str,
    camera_id: str,
    moment: datetime,
) -> Path:
    """Build a unique slice path inside the UTC hour containing ``moment``."""
    current = moment.astimezone(timezone.utc)
    start = hour_start(current)
    filename = (
        f"{start:%Y-%m-%dT%H-00-00Z}"
        f"__start-{current:%H-%M-%S-%fZ}.mkv"
    )
    return (
        root
        / Path(source_slug)
        / "cameras"
        / camera_id
        / f"{start:%Y}"
        / f"{start:%m}"
        / f"{start:%d}"
        / filename
    )


def build_ffmpeg_command(
    stream_url: str,
    output: Path,
    duration_seconds: int,
) -> list[str]:
    """Build a non-overwriting FFmpeg stream-copy command for one live slice."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-n",
        "-i",
        stream_url,
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-t",
        str(duration_seconds),
        str(output),
    ]


async def _wait_for_ffmpeg_exit(
    process: asyncio.subprocess.Process,
    timeout_seconds: float,
) -> bool:
    """Wait up to ``timeout_seconds`` for FFmpeg and report whether it exited."""
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        return False
    return True


async def _stop_ffmpeg_gracefully(process: asyncio.subprocess.Process) -> None:
    """Ask FFmpeg to finalize its output, escalating to kill only if it hangs."""
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return

    if await _wait_for_ffmpeg_exit(process, FFMPEG_GRACEFUL_SHUTDOWN_SECONDS):
        return

    log(f"[ffmpeg] pid={process.pid} did not stop after SIGINT; killing it")
    try:
        process.kill()
    except ProcessLookupError:
        return
    await _wait_for_ffmpeg_exit(process, FFMPEG_KILL_SHUTDOWN_SECONDS)


async def record_one_hour_slice(
    stream_url: str,
    output_root: Path,
    source_slug: str,
    camera_id: str,
) -> int:
    """Record one immutable slice ending on the next UTC hour boundary."""
    now = datetime.now(timezone.utc)
    output = output_path_for_hour(output_root, source_slug, camera_id, now)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = seconds_until_next_hour(now)

    log(
        f"[ffmpeg] start {source_slug}/{camera_id}: duration={duration}s, "
        f"output={output}, stream={stream_url}"
    )
    # FFmpeg runs as an independent OS process; awaiting it keeps other cameras active.
    process = await asyncio.create_subprocess_exec(
        *build_ffmpeg_command(stream_url, output, duration)
    )
    pid = getattr(process, "pid", "unknown")
    log(f"[ffmpeg] pid={pid} recording {source_slug}/{camera_id}")
    try:
        returncode = await process.wait()
    except asyncio.CancelledError:
        log(f"[ffmpeg] stopping {source_slug}/{camera_id}: pid={pid}")
        await _stop_ffmpeg_gracefully(process)
        raise

    log(
        f"[ffmpeg] exit {source_slug}/{camera_id}: "
        f"pid={pid}, returncode={returncode}, output={output}"
    )
    return returncode
