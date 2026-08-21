import json
from pathlib import Path

from scripts.stream_recorder.metadata import write_json_atomic


def test_write_json_atomic_writes_valid_json(tmp_path: Path):
    path = tmp_path / "nested" / "camera.json"
    write_json_atomic(path, {"camera_id": "camera-001", "url": "https://example.test/live.m3u8"})
    assert json.loads(path.read_text()) == {"camera_id": "camera-001", "url": "https://example.test/live.m3u8"}
