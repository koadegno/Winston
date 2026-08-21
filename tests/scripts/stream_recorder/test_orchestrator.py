from pathlib import Path

from scripts.stream_recorder.hls import HlsVariant
from scripts.stream_recorder.orchestrator import assign_camera_ids, record_camera_loop
from scripts.stream_recorder.sources import Source


def test_assign_camera_ids_is_deterministic():
    streams = [
        HlsVariant("https://example.test/z.m3u8", resolution=(1920, 1080)),
        HlsVariant("https://example.test/a.m3u8", resolution=(1280, 720)),
    ]
    cameras = assign_camera_ids(streams)
    assert [(c.id, c.stream.url) for c in cameras] == [
        ("camera-001", "https://example.test/a.m3u8"),
        ("camera-002", "https://example.test/z.m3u8"),
    ]


def test_record_camera_loop_rediscovers_after_ffmpeg_failure(monkeypatch, tmp_path: Path):
    source = Source("1", "Square", "City", "Country", "https://example.test/page")
    initial = HlsVariant("https://example.test/old.m3u8")
    replacement = HlsVariant("https://example.test/new.m3u8")
    calls: list[str] = []

    def fake_record(stream_url, *_args, **_kwargs):
        calls.append(stream_url)
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr("scripts.stream_recorder.orchestrator.record_one_hour_slice", fake_record)
    monkeypatch.setattr(
        "scripts.stream_recorder.orchestrator.discover_source_cameras",
        lambda *_args, **_kwargs: assign_camera_ids([replacement]),
    )
    monkeypatch.setattr("scripts.stream_recorder.orchestrator.time.sleep", lambda *_args: None)

    record_camera_loop(source, "camera-001", initial, tmp_path, max_cycles=2)
    assert calls == [initial.url, replacement.url]
