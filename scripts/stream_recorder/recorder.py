"""Create hour-aligned MKV slices from live HLS streams with FFmpeg."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .log import log


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
    """Build the deterministic recording path for one camera and UTC hour."""
    start = hour_start(moment)
    return (
        root
        / Path(source_slug)
        / "cameras"
        / camera_id
        / f"{start:%Y}"
        / f"{start:%m}"
        / f"{start:%d}"
        / f"{start:%Y-%m-%dT%H-00-00Z}.mkv"
    )


def build_ffmpeg_command(
    stream_url: str,
    output: Path,
    duration_seconds: int,
) -> list[str]:
    """Build the FFmpeg command that copies a live stream without re-encoding."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
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


async def record_one_hour_slice(
    stream_url: str,
    output_root: Path,
    source_slug: str,
    camera_id: str,
) -> int:
    """Record one slice ending on the next UTC hour boundary asynchronously."""
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
    log(f"[ffmpeg] pid={process.pid} recording {source_slug}/{camera_id}")
    returncode = await process.wait()
    log(
        f"[ffmpeg] exit {source_slug}/{camera_id}: "
        f"pid={process.pid}, returncode={returncode}, output={output}"
    )
    return returncode
