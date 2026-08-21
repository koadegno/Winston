from pathlib import Path

import pytest

from winston.cli import main
from winston.ingest.models import ImageMetadata, MediaFile, MediaType, VideoMetadata


def test_scan_command_prints_video_and_image_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    video = MediaFile(tmp_path / "camera.mkv", MediaType.VIDEO)
    image = MediaFile(tmp_path / "photo.jpg", MediaType.IMAGE)

    monkeypatch.setattr("winston.cli.scan_media", lambda _: [video, image])

    def fake_probe(media: MediaFile) -> VideoMetadata | ImageMetadata:
        if media.media_type is MediaType.VIDEO:
            return VideoMetadata(media.path, 1920, 1080, "h264", 3723.5, 25.0)
        return ImageMetadata(media.path, 2048, 1536)

    monkeypatch.setattr("winston.cli.probe_media", fake_probe)

    assert main(["scan", str(tmp_path)]) == 0
    assert capsys.readouterr().out == (
        "VIDEO  camera.mkv\n"
        "       1920x1080\n"
        "       h264\n"
        "       25 fps\n"
        "       01:02:03.500\n"
        "IMAGE  photo.jpg\n"
        "       2048x1536\n"
    )
