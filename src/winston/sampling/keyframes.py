"""Asynchronous decoder-keyframe sampling backed by FFmpeg/ffprobe."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from winston.config import get_config
from winston.ingest.models import VideoMetadata
from winston.sampling.base import FrameSampler
from winston.sampling.models import SampledFrame


type JsonObject = dict[str, JsonValue]


class FrameSamplingError(RuntimeError):
    """Raised when keyframe discovery or decoding fails."""


async def _run_ffprobe(path: Path) -> JsonObject:
    """Run ffprobe asynchronously and return its validated JSON object."""
    # Keep probing off the event loop so multiple videos can be processed concurrently.
    config = get_config()
    command = [
        config.ffprobe_binary,
        "-v",
        "error",
        "-skip_frame",
        "nokey",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=key_frame,best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise FrameSamplingError(f"ffprobe executable not found: {config.ffprobe_binary}") from exc

    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip() or "ffprobe returned a non-zero exit status"
        raise FrameSamplingError(f"Failed to inspect keyframes for {path}: {message}")

    try:
        payload = cast(JsonValue, json.loads(stdout))
    except json.JSONDecodeError as exc:
        raise FrameSamplingError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(payload, dict):
        raise FrameSamplingError(f"ffprobe returned an unexpected JSON document for {path}")
    return cast(JsonObject, payload)


def _parse_keyframe_timestamps(path: Path, payload: JsonObject) -> tuple[float, ...]:
    """Extract unique decoder-keyframe timestamps while preserving source order."""
    # Validate ffprobe output before associating timestamps with decoded frame bytes.
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise FrameSamplingError(f"ffprobe returned no frame list for {path}")

    timestamps: list[float] = []
    seen: set[float] = set()
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("key_frame") != 1:
            continue
        raw_timestamp = frame.get("best_effort_timestamp_time")
        try:
            timestamp = float(raw_timestamp)
        except (TypeError, ValueError) as exc:
            raise FrameSamplingError(f"Invalid keyframe timestamp for {path}: {raw_timestamp!r}") from exc
        if timestamp in seen:
            continue
        seen.add(timestamp)
        timestamps.append(timestamp)

    if not timestamps:
        raise FrameSamplingError(f"No decoder keyframes found in {path}")
    return tuple(timestamps)


async def probe_keyframe_timestamps(path: Path) -> tuple[float, ...]:
    """Return unique decoder-keyframe timestamps in source order."""
    # Keep subprocess execution and JSON interpretation behind one public helper.
    return _parse_keyframe_timestamps(path, await _run_ffprobe(path))


async def _read_stderr(stream: asyncio.StreamReader | None) -> bytes:
    """Drain a subprocess stderr stream without blocking frame decoding."""
    # Reading stderr concurrently prevents FFmpeg from blocking on a full pipe.
    if stream is None:
        return b""
    return await stream.read()


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Terminate a running subprocess and wait until it has exited."""
    # Always reap the child process so cancelled or partially consumed sampling leaves no FFmpeg behind.
    if process.returncode is None:
        process.terminate()
    await process.wait()


class KeyframeSampler(FrameSampler):
    """Yield decoder keyframes as RGB24 frames without container-specific logic."""

    async def sample(self, video: VideoMetadata) -> AsyncIterator[SampledFrame]:
        """Stream decoded keyframes paired with their exact source timestamps."""
        # Probe timestamps first so each streamed RGB frame can be paired deterministically.
        timestamps = await probe_keyframe_timestamps(video.path)
        config = get_config()
        command = [
            config.ffmpeg_binary,
            "-v",
            "error",
            "-skip_frame",
            "nokey",
            "-noautorotate",
            "-i",
            str(video.path),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise FrameSamplingError(f"ffmpeg executable not found: {config.ffmpeg_binary}") from exc

        if process.stdout is None:
            await _stop_process(process)
            raise FrameSamplingError(f"ffmpeg stdout pipe unavailable for {video.path}")

        stderr_task = asyncio.create_task(_read_stderr(process.stderr))
        frame_size = video.width * video.height * 3
        completed = False
        try:
            for timestamp in timestamps:
                try:
                    rgb24 = await process.stdout.readexactly(frame_size)
                except asyncio.IncompleteReadError as exc:
                    await process.wait()
                    stderr = (await stderr_task).decode(errors="replace").strip()
                    detail = stderr or f"decoded only {len(exc.partial)} bytes of a {frame_size}-byte frame"
                    raise FrameSamplingError(f"Failed to decode keyframes for {video.path}: {detail}") from exc

                yield SampledFrame(
                    source_path=video.path,
                    timestamp_seconds=timestamp,
                    width=video.width,
                    height=video.height,
                    rgb24=rgb24,
                )

            extra = await process.stdout.read(1)
            if extra:
                await _stop_process(process)
                await stderr_task
                raise FrameSamplingError(
                    f"ffmpeg decoded more keyframes than ffprobe reported for {video.path}"
                )

            returncode = await process.wait()
            stderr = (await stderr_task).decode(errors="replace").strip()
            if returncode != 0:
                raise FrameSamplingError(
                    f"Failed to decode keyframes for {video.path}: "
                    f"{stderr or 'ffmpeg returned a non-zero exit status'}"
                )
            completed = True
        finally:
            if not completed:
                await _stop_process(process)
                if not stderr_task.done():
                    await stderr_task
