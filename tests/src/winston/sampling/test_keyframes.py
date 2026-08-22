import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from winston.ingest.models import VideoMetadata
from winston.sampling.keyframes import FrameSamplingError, KeyframeSampler
from winston.sampling.models import SampledFrame


async def collect_frames(sampler: KeyframeSampler, metadata: VideoMetadata) -> list[SampledFrame]:
    return [frame async for frame in sampler.sample(metadata)]


def make_h264_mkv(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-g",
            "5",
            "-keyint_min",
            "5",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg required")
def test_keyframe_sampler_extracts_only_decoder_keyframes(tmp_path: Path) -> None:
    video = tmp_path / "fixture.mkv"
    make_h264_mkv(video)
    metadata = VideoMetadata(
        path=video,
        width=64,
        height=48,
        codec="h264",
        duration_seconds=3.0,
        fps=5.0,
    )

    frames = asyncio.run(collect_frames(KeyframeSampler(), metadata))

    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0.0, 1.0, 2.0])
    assert len({frame.timestamp_seconds for frame in frames}) == len(frames)
    assert all(frame.source_path == video for frame in frames)
    assert all(frame.width == 64 for frame in frames)
    assert all(frame.height == 48 for frame in frames)
    assert all(len(frame.rgb24) == 64 * 48 * 3 for frame in frames)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg required")
def test_keyframe_sampler_does_not_gate_on_file_suffix(tmp_path: Path) -> None:
    video = tmp_path / "fixture.custom"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=32x24:rate=1",
            "-t",
            "1",
            "-c:v",
            "ffv1",
            "-f",
            "matroska",
            str(video),
        ],
        check=True,
    )
    metadata = VideoMetadata(
        path=video,
        width=32,
        height=24,
        codec="ffv1",
        duration_seconds=1.0,
        fps=1.0,
    )

    frames = asyncio.run(collect_frames(KeyframeSampler(), metadata))

    assert len(frames) == 1
    assert frames[0].timestamp_seconds == pytest.approx(0.0)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required")
def test_keyframe_sampler_rejects_corrupt_video(tmp_path: Path) -> None:
    video = tmp_path / "broken.mkv"
    video.write_bytes(b"not a video")
    metadata = VideoMetadata(
        path=video,
        width=64,
        height=48,
        codec="h264",
        duration_seconds=1.0,
        fps=5.0,
    )

    with pytest.raises(FrameSamplingError, match="broken.mkv"):
        asyncio.run(collect_frames(KeyframeSampler(), metadata))
