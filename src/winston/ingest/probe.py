import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from winston.config import settings
from winston.ingest.models import ImageMetadata, MediaFile, MediaMetadata, MediaType, VideoMetadata


class MediaProbeError(RuntimeError):
    pass


def run_ffprobe(path: Path) -> dict[str, Any]:
    command = [
        settings.ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise MediaProbeError(f"ffprobe executable not found: {settings.ffprobe_binary}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or "ffprobe returned a non-zero exit status"
        raise MediaProbeError(message) from exc

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise MediaProbeError("ffprobe returned an unexpected JSON document")
    return payload


def _video_stream(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise MediaProbeError(f"No video stream found in {path}")
    return streams[0]


def _integer_field(path: Path, stream: dict[str, Any], name: str) -> int:
    try:
        return int(stream[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaProbeError(f"Invalid {name} in ffprobe metadata for {path}") from exc


def _frame_rate(path: Path, stream: dict[str, Any]) -> float:
    for field in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(field)
        if not value:
            continue
        try:
            rate = Fraction(str(value))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return float(rate)
    raise MediaProbeError(f"Invalid frame rate in ffprobe metadata for {path}")


def parse_probe(path: Path, media_type: MediaType, payload: dict[str, Any]) -> MediaMetadata:
    stream = _video_stream(path, payload)
    width = _integer_field(path, stream, "width")
    height = _integer_field(path, stream, "height")

    if media_type is MediaType.IMAGE:
        return ImageMetadata(path=path, width=width, height=height)

    codec = stream.get("codec_name")
    if not isinstance(codec, str) or not codec:
        raise MediaProbeError(f"Invalid codec in ffprobe metadata for {path}")

    format_metadata = payload.get("format")
    if not isinstance(format_metadata, dict):
        raise MediaProbeError(f"Invalid format metadata for {path}")
    try:
        duration = float(format_metadata["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaProbeError(f"Invalid duration in ffprobe metadata for {path}") from exc

    return VideoMetadata(
        path=path,
        width=width,
        height=height,
        codec=codec,
        duration_seconds=duration,
        fps=_frame_rate(path, stream),
    )


def probe_media(media: MediaFile) -> MediaMetadata:
    try:
        payload = run_ffprobe(media.path)
        return parse_probe(media.path, media.media_type, payload)
    except MediaProbeError as exc:
        raise MediaProbeError(f"Failed to probe {media.path}: {exc}") from exc
