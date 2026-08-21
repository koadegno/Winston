from pathlib import Path

import pytest

from winston.ingest.models import ImageMetadata, MediaFile, MediaType, VideoMetadata
from winston.ingest.probe import MediaProbeError, parse_probe, probe_media


def test_parse_video_probe_extracts_metadata() -> None:
    payload = {
        "streams": [{"codec_name": "h264", "width": 1920, "height": 1080, "avg_frame_rate": "25/1"}],
        "format": {"duration": "3723.5"},
    }

    metadata = parse_probe(Path("camera.mkv"), MediaType.VIDEO, payload)

    assert metadata == VideoMetadata(
        path=Path("camera.mkv"),
        width=1920,
        height=1080,
        codec="h264",
        duration_seconds=3723.5,
        fps=25.0,
    )


def test_parse_video_probe_falls_back_to_r_frame_rate() -> None:
    payload = {
        "streams": [
            {
                "codec_name": "hevc",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "0/0",
                "r_frame_rate": "30000/1001",
            }
        ],
        "format": {"duration": "10.0"},
    }

    metadata = parse_probe(Path("camera.mkv"), MediaType.VIDEO, payload)

    assert isinstance(metadata, VideoMetadata)
    assert metadata.fps == pytest.approx(29.97002997)


def test_parse_image_probe_extracts_dimensions() -> None:
    payload = {"streams": [{"codec_name": "mjpeg", "width": 2048, "height": 1536}], "format": {}}

    metadata = parse_probe(Path("photo.jpg"), MediaType.IMAGE, payload)

    assert metadata == ImageMetadata(path=Path("photo.jpg"), width=2048, height=1536)


def test_parse_probe_rejects_missing_video_stream() -> None:
    with pytest.raises(MediaProbeError, match="camera.mkv"):
        parse_probe(Path("camera.mkv"), MediaType.VIDEO, {"streams": [], "format": {}})


def test_probe_media_wraps_ffprobe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    media = MediaFile(path=Path("broken.mkv"), media_type=MediaType.VIDEO)

    def fail(_: Path) -> dict[str, object]:
        raise MediaProbeError("ffprobe failed")

    monkeypatch.setattr("winston.ingest.probe.run_ffprobe", fail)

    with pytest.raises(MediaProbeError, match="broken.mkv"):
        probe_media(media)
