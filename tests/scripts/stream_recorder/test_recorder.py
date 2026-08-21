from datetime import datetime, timezone
from pathlib import Path

from scripts.stream_recorder.recorder import build_ffmpeg_command, output_path_for_hour


def test_output_path_is_organized_by_source_camera_and_utc_date(tmp_path: Path):
    when = datetime(2026, 8, 22, 10, 15, tzinfo=timezone.utc)
    path = output_path_for_hour(tmp_path, "belgium/brussels/grand-place", "camera-001", when)
    assert path == tmp_path / "belgium/brussels/grand-place/cameras/camera-001/2026/08/22/2026-08-22T10-00-00Z.mkv"


def test_ffmpeg_command_stream_copies_and_stops_at_boundary(tmp_path: Path):
    output = tmp_path / "out.mkv"
    cmd = build_ffmpeg_command("https://example.test/live.m3u8", output, duration_seconds=123)
    assert cmd[:2] == ["ffmpeg", "-hide_banner"]
    assert ["-c", "copy"] == cmd[cmd.index("-c"):cmd.index("-c") + 2]
    assert cmd[cmd.index("-t") + 1] == "123"
    assert cmd[-1] == str(output)
