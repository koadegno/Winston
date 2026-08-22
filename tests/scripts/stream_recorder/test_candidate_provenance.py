import pytest

import scripts.stream_recorder.discovery as discovery
import scripts.stream_recorder.hls as hls
from scripts.stream_recorder.hls import HlsVariant


@pytest.mark.asyncio
async def test_discover_page_reports_browser_observed_candidates(monkeypatch):
    """Discovery exposes which HLS URLs were actually requested by the rendered player."""
    static = "https://static.example/stale.m3u8"
    dynamic = "https://cdn.example/live.m3u8"
    observed: set[str] = set()
    sentinel = object()

    async def fake_http(_url: str, **_kwargs) -> set[str]:
        """Return one URL found only in static page text."""
        return {static}

    async def fake_browser(_url: str, *, browser, **_kwargs) -> set[str]:
        """Return one URL emitted by browser network traffic."""
        assert browser is sentinel
        return {dynamic}

    monkeypatch.setattr(discovery, "discover_http", fake_http)
    monkeypatch.setattr(discovery, "discover_browser", fake_browser)

    candidates = await discovery.discover_page(
        "https://example.test/page",
        browser=sentinel,
        client=object(),
        browser_observed_urls=observed,
    )

    assert candidates == {static, dynamic}
    assert observed == {dynamic}


@pytest.mark.asyncio
async def test_resolve_cameras_preserves_failed_browser_observed_candidate(monkeypatch):
    """A network-observed camera survives a transient probing failure with unknown metadata."""
    debica = "https://online2.kamery24.org/cam/debica.m3u8"

    async def fail_resolve(_url: str) -> tuple[HlsVariant, set[str]]:
        """Model an upstream HLS endpoint that is temporarily returning 404."""
        raise RuntimeError("404 Not Found")

    monkeypatch.setattr(hls, "resolve_candidate", fail_resolve)

    cameras = await hls.resolve_cameras(
        {debica},
        browser_observed_urls={debica},
    )

    assert cameras == [HlsVariant(url=debica)]


@pytest.mark.asyncio
async def test_resolve_cameras_drops_failed_static_only_candidate(monkeypatch):
    """A stale HTML-only HLS string must not become a camera when its probe fails."""
    stale = "https://www.skylinewebcams.com/player/livee.m3u8?a=stale"
    live = "https://hd-auth.skylinewebcams.com/live.m3u8?a=current"

    async def fake_resolve(url: str) -> tuple[HlsVariant, set[str]]:
        """Resolve the browser stream and reject the stale HTML-only URL."""
        if url == stale:
            raise RuntimeError("404 Not Found")
        return HlsVariant(url=url), set()

    monkeypatch.setattr(hls, "resolve_candidate", fake_resolve)

    cameras = await hls.resolve_cameras(
        {stale, live},
        browser_observed_urls={live},
    )

    assert cameras == [HlsVariant(url=live)]


@pytest.mark.asyncio
async def test_resolve_cameras_retries_observed_variant_in_same_family(monkeypatch):
    """A failed preferred master falls back to an observed sibling before becoming unresolved."""
    master = "https://camera.example/live/playlist.m3u8"
    chunklist = "https://camera.example/live/chunklist_w123.m3u8"
    calls: list[str] = []

    async def fake_resolve(url: str) -> tuple[HlsVariant, set[str]]:
        """Reject the preferred master but resolve the browser-observed media playlist."""
        calls.append(url)
        if url == master:
            raise RuntimeError("404 Not Found")
        return HlsVariant(url=url), set()

    monkeypatch.setattr(hls, "resolve_candidate", fake_resolve)

    cameras = await hls.resolve_cameras(
        {master, chunklist},
        browser_observed_urls={chunklist},
    )

    assert calls == [master, chunklist]
    assert cameras == [HlsVariant(url=chunklist)]


@pytest.mark.asyncio
async def test_resolve_cameras_keeps_failed_observed_family_beside_working_family(monkeypatch):
    """One temporarily down browser-observed camera is not lost because another camera works."""
    working = "https://camera.example/front.m3u8"
    down = "https://camera.example/back.m3u8"

    async def fake_resolve(url: str) -> tuple[HlsVariant, set[str]]:
        """Resolve the front camera while the back camera remains temporarily unavailable."""
        if url == down:
            raise RuntimeError("404 Not Found")
        return HlsVariant(url=url), set()

    monkeypatch.setattr(hls, "resolve_candidate", fake_resolve)

    cameras = await hls.resolve_cameras(
        {working, down},
        browser_observed_urls={working, down},
    )

    assert {camera.url for camera in cameras} == {working, down}
