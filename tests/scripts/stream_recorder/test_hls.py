import asyncio
from types import SimpleNamespace

import httpx
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
async def test_resolve_candidate_retries_only_certificate_failures_without_tls_verification(monkeypatch):
    """A discovered HLS URL with a broken certificate gets one narrowly scoped insecure retry."""
    url = "https://online2.kamery24.org/cam/debica.m3u8"
    secure_client = object()
    insecure_client = object()
    fetch_clients: list[object] = []
    async_client_kwargs: list[dict[str, object]] = []

    async def fake_fetch_text(_url: str, *, client=None, **_kwargs):
        """Fail certificate verification once, then return a media playlist via the insecure client."""
        fetch_clients.append(client)
        if client is secure_client:
            raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        assert client is insecure_client
        return "#EXTM3U\n#EXTINF:4,\nsegment.ts\n", url

    class FakeAsyncClient:
        """Model the one temporary client created with certificate verification disabled."""

        def __init__(self, **kwargs) -> None:
            """Capture construction options so the retry stays narrowly scoped."""
            async_client_kwargs.append(kwargs)

        async def __aenter__(self):
            """Return the synthetic insecure HTTP client."""
            return insecure_client

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            """Close the synthetic client."""
            return None

    monkeypatch.setattr(hls, "fetch_text", fake_fetch_text)
    monkeypatch.setattr(
        hls,
        "httpx",
        SimpleNamespace(AsyncClient=FakeAsyncClient, ConnectError=httpx.ConnectError),
        raising=False,
    )

    stream, variants = await hls.resolve_candidate(url, client=secure_client)

    assert stream.url == url
    assert variants == set()
    assert fetch_clients == [secure_client, insecure_client]
    assert len(async_client_kwargs) == 1
    assert async_client_kwargs[0]["verify"] is False


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


@pytest.mark.asyncio
async def test_resolve_cameras_groups_master_and_dynamic_chunklist(monkeypatch):
    """A master playlist and its changing chunklist must represent one physical camera."""
    master = (
        "https://5b10f5ee6a4c9.streamlock.net:2937/vlmcam01/"
        "vlmcam01.stream/playlists.m3u8"
    )
    chunklist = (
        "https://5b10f5ee6a4c9.streamlock.net:2937/vlmcam01/"
        "vlmcam01.stream/chunklist_w1569344971.m3u8"
    )
    resolved_urls: list[str] = []

    async def fake_resolve_candidate(url: str) -> tuple[HlsVariant, set[str]]:
        """Record which representative candidate is actually resolved."""
        resolved_urls.append(url)
        return HlsVariant(url=url), set()

    monkeypatch.setattr(hls, "resolve_candidate", fake_resolve_candidate)
    cameras = await hls.resolve_cameras({master, chunklist})

    assert len(cameras) == 1
    assert resolved_urls == [master]


@pytest.mark.asyncio
async def test_resolve_cameras_groups_master_and_session_output(monkeypatch):
    """A stable UUID master and its session-scoped output playlist must be one camera."""
    master = "https://kamery.netsystem.net.pl/memfs/886f3c4d-8533-49cd-8679-a5cbc1824418.m3u8"
    output = (
        "https://kamery.netsystem.net.pl/memfs/"
        "886f3c4d-8533-49cd-8679-a5cbc1824418_output_0.m3u8?session=temporary"
    )
    resolved_urls: list[str] = []

    async def fake_resolve_candidate(url: str) -> tuple[HlsVariant, set[str]]:
        """Record which representative candidate is actually resolved."""
        resolved_urls.append(url)
        return HlsVariant(url=url), set()

    monkeypatch.setattr(hls, "resolve_candidate", fake_resolve_candidate)
    cameras = await hls.resolve_cameras({master, output})

    assert len(cameras) == 1
    assert resolved_urls == [master]


@pytest.mark.asyncio
async def test_resolve_cameras_groups_default_https_port_duplicates(monkeypatch):
    """The same HLS camera observed with implicit and explicit HTTPS ports is one camera."""
    implicit = "https://6034e09794f07.streamlock.net/mzdim/smil:rynek1.smil/playlist.m3u8"
    explicit = "https://6034e09794f07.streamlock.net:443/mzdim/smil:rynek1.smil/playlist.m3u8"
    resolved_urls: list[str] = []

    async def fake_resolve_candidate(url: str) -> tuple[HlsVariant, set[str]]:
        """Record the single representative candidate chosen for the camera."""
        resolved_urls.append(url)
        return HlsVariant(url=url), set()

    monkeypatch.setattr(hls, "resolve_candidate", fake_resolve_candidate)
    cameras = await hls.resolve_cameras({implicit, explicit})

    assert len(cameras) == 1
    assert len(resolved_urls) == 1


@pytest.mark.asyncio
async def test_resolve_cameras_groups_cdn_mirrors_with_same_stream_identity(monkeypatch):
    """Equivalent CDN hosts serving the same stream UUID must not start duplicate recordings."""
    cdn01 = (
        "https://cdn01.aztv.pl/live_lubliniec/52c84a50-1209-11ec-92cf-8f52e18fd989/"
        "playlist.m3u8?scendtime=1&schash=a"
    )
    cdn02 = (
        "https://cdn02.aztv.pl/live_lubliniec/52c84a50-1209-11ec-92cf-8f52e18fd989/"
        "playlist.m3u8?scendtime=2&schash=b"
    )
    resolved_urls: list[str] = []

    async def fake_resolve_candidate(url: str) -> tuple[HlsVariant, set[str]]:
        """Record the single CDN representative chosen for this camera."""
        resolved_urls.append(url)
        return HlsVariant(url=url), set()

    monkeypatch.setattr(hls, "resolve_candidate", fake_resolve_candidate)
    cameras = await hls.resolve_cameras({cdn01, cdn02})

    assert len(cameras) == 1
    assert len(resolved_urls) == 1
