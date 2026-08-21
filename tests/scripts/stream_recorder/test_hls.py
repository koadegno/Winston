import asyncio

import pytest

from scripts.stream_recorder import hls
from scripts.stream_recorder.hls import HlsVariant, choose_best_stream, parse_master_playlist

MASTER = '''#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360
low/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2400000,RESOLUTION=1280x720
https://cdn.example/high/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
full/index.m3u8
'''


def test_parse_master_playlist_resolves_variants():
    """Master parsing resolves relative URLs and preserves resolutions."""
    variants = parse_master_playlist(MASTER, "https://origin.example/live/master.m3u8")
    assert [variant.resolution for variant in variants] == [(640, 360), (1280, 720), (1920, 1080)]
    assert variants[-1].url == "https://origin.example/live/full/index.m3u8"


def test_choose_best_stream_prefers_highest_resolution_then_bandwidth():
    """Variant selection prioritizes pixels before bandwidth."""
    variants = parse_master_playlist(MASTER, "https://origin.example/live/master.m3u8")
    assert choose_best_stream(variants).resolution == (1920, 1080)


@pytest.mark.asyncio
async def test_hls_candidates_resolve_concurrently(monkeypatch):
    """Independent HLS candidates resolve concurrently."""
    candidates = {"https://example.test/a.m3u8", "https://example.test/b.m3u8"}
    both_started = asyncio.Event()
    started: set[str] = set()

    async def fake_resolve_candidate(url: str) -> tuple[HlsVariant, set[str]]:
        """Block each resolver until all candidate resolvers have started."""
        started.add(url)
        if len(started) == len(candidates):
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)
        return HlsVariant(url=url), set()

    monkeypatch.setattr(hls, "resolve_candidate", fake_resolve_candidate)
    cameras = await hls.resolve_cameras(candidates)
    assert {camera.url for camera in cameras} == candidates
