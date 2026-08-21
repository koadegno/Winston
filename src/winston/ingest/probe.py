"""Read image and video metadata through ffprobe."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from winston.config import get_config
from winston.ingest.models import ImageMetadata, MediaFile, MediaMetadata, MediaType, VideoMetadata


type JsonObject = dict[str, JsonValue]


class MediaProbeError(RuntimeError):
    """Raised when ffprobe cannot provide valid metadata for a media file."""


def run_ffprobe(path: Path) -> JsonObject:
    """Run ffprobe for the first visual stream and return its JSON metadata."""
    config = get_config()

    # Images are exposed by ffprobe as a video stream too, so the same command
    # can inspect both supported media types.
    command = [
        config.ffprobe_binary,
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
        raise MediaProbeError(f"ffprobe executable not found: {config.ffprobe_binary}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or "ffprobe returned a non-zero exit status"
        raise MediaProbeError(message) from exc

    try:
        payload = cast(JsonValue, json.loads(result.stdout))
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise MediaProbeError("ffprobe returned an unexpected JSON document")
    return cast(JsonObject, payload)


def _video_stream(path: Path, payload: JsonObject) -> JsonObject:
    """Return the first visual stream from an ffprobe payload."""
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise MediaProbeError(f"No video stream found in {path}")
    return streams[0]


def _integer_field(path: Path, stream: JsonObject, name: str) -> int:
    """Read an integer-valued field from an ffprobe stream."""
    value = stream.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise MediaProbeError(f"Invalid {name} in ffprobe metadata for {path}")
    try:
        return int(value)
    except ValueError as exc:
        raise MediaProbeError(f"Invalid {name} in ffprobe metadata for {path}") from exc


def _frame_rate(path: Path, stream: JsonObject) -> float:
    """Return a positive frame rate, preferring average rate over raw rate."""
    # avg_frame_rate is the best representation for playback; some files expose
    # only r_frame_rate, so retain that as a fallback.
    for field in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        try:
            rate = Fraction(str(value))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return float(rate)
    raise MediaProbeError(f"Invalid frame rate in ffprobe metadata for {path}")


def parse_probe(path: Path, media_type: MediaType, payload: JsonObject) -> MediaMetadata:
    """Convert validated ffprobe JSON into Winston media metadata."""
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

    duration_value = format_metadata.get("duration")
    if isinstance(duration_value, bool) or not isinstance(duration_value, (int, float, str)):
        raise MediaProbeError(f"Invalid duration in ffprobe metadata for {path}")
    try:
        duration = float(duration_value)
    except ValueError as exc:
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
    """Probe one media file and attach its path to any resulting error."""
    try:
        payload = run_ffprobe(media.path)
        return parse_probe(media.path, media.media_type, payload)
    except MediaProbeError as exc:
        raise MediaProbeError(f"Failed to probe {media.path}: {exc}") from exc
