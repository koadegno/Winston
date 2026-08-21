import asyncio
from pathlib import Path

import pytest

from scripts.stream_recorder import orchestrator
from scripts.stream_recorder.hls import HlsVariant
from scripts.stream_recorder.orchestrator import Camera, assign_camera_ids, record_camera_loop
from scripts.stream_recorder.sources import Source


def test_assign_camera_ids_is_deterministic():
    """Camera IDs are assigned from deterministic stream URL ordering."""
    streams = [
        HlsVariant("https://example.test/z.m3u8", resolution=(1920, 1080)),
        HlsVariant("https://example.test/a.m3u8", resolution=(1280, 720)),
    ]
    cameras = assign_camera_ids(streams)
    assert [(camera.id, camera.stream.url) for camera in cameras] == [
        ("camera-001", "https://example.test/a.m3u8"),
        ("camera-002", "https://example.test/z.m3u8"),
    ]


@pytest.mark.asyncio
async def test_record_camera_loop_rediscovers_after_ffmpeg_failure(monkeypatch, tmp_path: Path):
    """A failed FFmpeg run rediscoveries the camera before the next cycle."""
    source = Source("1", "Square", "City", "Country", "https://example.test/page")
    initial = HlsVariant("https://example.test/old.m3u8")
    replacement = HlsVariant("https://example.test/new.m3u8")
    calls: list[str] = []

    async def fake_record(stream_url, *_args, **_kwargs):
        """Fail the first recording cycle and succeed the second."""
        calls.append(stream_url)
        return 1 if len(calls) == 1 else 0

    async def fake_discover(*_args, **_kwargs):
        """Return the replacement stream during rediscovery."""
        return assign_camera_ids([replacement])

    monkeypatch.setattr(orchestrator, "record_one_hour_slice", fake_record)
    monkeypatch.setattr(orchestrator, "discover_source_cameras", fake_discover)
    await record_camera_loop(
        source,
        "camera-001",
        initial,
        tmp_path,
        retry_seconds=0,
        max_cycles=2,
    )
    assert calls == [initial.url, replacement.url]


@pytest.mark.asyncio
async def test_record_source_starts_all_camera_workers_concurrently(monkeypatch, tmp_path: Path):
    """All camera loops for one source start concurrently."""
    source = Source("1", "Square", "City", "Country", "https://example.test/page")
    cameras = [
        Camera("camera-001", HlsVariant("https://example.test/a.m3u8")),
        Camera("camera-002", HlsVariant("https://example.test/b.m3u8")),
    ]
    both_started = asyncio.Event()
    started: set[str] = set()

    async def fake_discover(*_args, **_kwargs):
        """Return two cameras without performing real network discovery."""
        return cameras

    async def fake_record_loop(_source, camera_id, *_args, **_kwargs):
        """Block each camera until both camera loops have started."""
        started.add(camera_id)
        if len(started) == len(cameras):
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)

    monkeypatch.setattr(orchestrator, "discover_source_cameras", fake_discover)
    monkeypatch.setattr(orchestrator, "record_camera_loop", fake_record_loop)
    monkeypatch.setattr(orchestrator, "write_source_metadata", lambda *_args, **_kwargs: None)
    await orchestrator.record_source_forever(source, tmp_path)
    assert started == {"camera-001", "camera-002"}
