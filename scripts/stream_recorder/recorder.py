from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess


def hour_start(moment: datetime) -> datetime:
    moment = moment.astimezone(timezone.utc)
    return moment.replace(minute=0, second=0, microsecond=0)


def seconds_until_next_hour(moment: datetime) -> int:
    current = moment.astimezone(timezone.utc)
    next_hour = hour_start(current) + timedelta(hours=1)
    return max(1, int((next_hour - current).total_seconds()))


def output_path_for_hour(root: Path, source_slug: str, camera_id: str, moment: datetime) -> Path:
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


def build_ffmpeg_command(stream_url: str, output: Path, duration_seconds: int) -> list[str]:
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


def record_one_hour_slice(stream_url: str, output_root: Path, source_slug: str, camera_id: str) -> int:
    now = datetime.now(timezone.utc)
    output = output_path_for_hour(output_root, source_slug, camera_id, now)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = seconds_until_next_hour(now)
    completed = subprocess.run(build_ffmpeg_command(stream_url, output, duration), check=False)
    return completed.returncode
