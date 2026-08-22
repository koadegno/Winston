import asyncio

import pytest

import scripts.stream_recorder.discovery as discovery
from scripts.stream_recorder.discovery import discover_http, extract_m3u8_urls


def test_extracts_absolute_relative_and_escaped_m3u8_urls():
    """Static extraction resolves absolute, relative, and escaped HLS URLs."""
    html = r'''
    <video data-src="https://cdn.example/live/master.m3u8"></video>
    <script>const x = "\/streams\/cam-2.m3u8?token=abc";</script>
    <a href="relative/cam-3.m3u8">x</a>
    '''
    urls = extract_m3u8_urls(html, "https://example.test/page/")
    assert urls == {
        "https://cdn.example/live/master.m3u8",
        "https://example.test/streams/cam-2.m3u8?token=abc",
        "https://example.test/page/relative/cam-3.m3u8",
    }


@pytest.mark.asyncio
async def test_http_discovery_leaves_iframe_loading_to_browser(monkeypatch):
    """HTTP discovery only fetches the source page and never follows third-party iframes."""
    calls: list[str] = []
    page = '<iframe src="https://third-party.example/player"></iframe>'

    async def fake_fetch(url: str, timeout: float = 20.0):
        """Return the synthetic source page and record HTTP calls."""
        calls.append(url)
        if url != "https://example.test/page":
            raise AssertionError("HTTP discovery must not fetch third-party iframes")
        return page, url

    monkeypatch.setattr("scripts.stream_recorder.discovery.fetch_text", fake_fetch)
    assert await discover_http("https://example.test/page") == set()
    assert calls == ["https://example.test/page"]


@pytest.mark.asyncio
async def test_browser_discovery_visits_iframe_urls_as_independent_targets():
    """Browser crawling follows iframe URLs as isolated targets when the parent has no stream."""
    root_url = "https://example.test/webcam"
    player_url = "https://player.example/embed/123"
    stream_url = "https://cdn.example/live/camera.m3u8"
    visited: list[str] = []

    async def visit(url: str) -> tuple[set[str], set[str]]:
        """Return a child player from the root and an HLS stream from the player."""
        visited.append(url)
        if url == root_url:
            return set(), {player_url}
        if url == player_url:
            return {stream_url}, set()
        raise AssertionError(f"unexpected target: {url}")

    assert await discovery.crawl_browser_targets(root_url, visit, max_iframe_depth=1) == {stream_url}
    assert visited == [root_url, player_url]


@pytest.mark.asyncio
async def test_browser_does_not_crawl_iframes_after_parent_finds_stream():
    """A target that already exposes HLS must not enqueue its iframe children."""
    root_url = "https://example.test/webcam"
    child_url = "https://player.example/unnecessary"
    stream_url = "https://cdn.example/live/camera.m3u8"
    visited: list[str] = []

    async def visit(url: str) -> tuple[set[str], set[str]]:
        """Expose one stream and one irrelevant child from the root page."""
        visited.append(url)
        if url == root_url:
            return {stream_url}, {child_url}
        return set(), set()

    assert await discovery.crawl_browser_targets(root_url, visit, max_iframe_depth=1) == {stream_url}
    assert visited == [root_url]


@pytest.mark.asyncio
async def test_browser_target_siblings_start_concurrently():
    """Sibling player targets at one iframe depth start concurrently."""
    root_url = "https://example.test/root"
    children = {"https://player.example/a", "https://player.example/b"}
    both_started = asyncio.Event()
    started: set[str] = set()

    async def visit(url: str) -> tuple[set[str], set[str]]:
        """Block each child until both sibling visits have started."""
        if url == root_url:
            return set(), children
        started.add(url)
        if len(started) == len(children):
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)
        return {f"{url}/stream.m3u8"}, set()

    streams = await discovery.crawl_browser_targets(root_url, visit, max_iframe_depth=1)
    assert streams == {f"{url}/stream.m3u8" for url in children}


@pytest.mark.asyncio
async def test_browser_is_skipped_when_http_already_finds_hls(monkeypatch):
    """Direct HTML HLS discovery must avoid consuming a Playwright tab."""
    calls: list[str] = []
    sentinel = object()
    stream_url = "https://example.test/direct.m3u8"

    async def fake_http(_url: str) -> set[str]:
        """Return a stream directly from the HTML source."""
        calls.append("http")
        return {stream_url}

    async def fake_browser(_url: str, *, browser) -> set[str]:
        """Record any unnecessary browser fallback invocation."""
        assert browser is sentinel
        calls.append("browser")
        return {"https://example.test/browser.m3u8"}

    monkeypatch.setattr(discovery, "discover_http", fake_http)
    monkeypatch.setattr(discovery, "discover_browser", fake_browser)

    assert await discovery.discover_page("https://example.test/page", browser=sentinel) == {stream_url}
    assert calls == ["http"]


@pytest.mark.asyncio
async def test_browser_runs_after_http_miss(monkeypatch):
    """Playwright remains available as the fallback when direct HTML has no HLS."""
    sentinel = object()
    browser_stream = "https://example.test/browser.m3u8"
    calls: list[str] = []

    async def fake_http(_url: str) -> set[str]:
        """Simulate a source page without an HLS URL in its HTML."""
        calls.append("http")
        return set()

    async def fake_browser(_url: str, *, browser) -> set[str]:
        """Return the HLS URL found from browser network traffic."""
        assert browser is sentinel
        calls.append("browser")
        return {browser_stream}

    monkeypatch.setattr(discovery, "discover_http", fake_http)
    monkeypatch.setattr(discovery, "discover_browser", fake_browser)

    assert await discovery.discover_page("https://example.test/page", browser=sentinel) == {browser_stream}
    assert calls == ["http", "browser"]


@pytest.mark.asyncio
async def test_browser_targets_share_one_context_and_respect_page_limit():
    """Browser targets reuse one context and never exceed the configured active-tab limit."""
    active_pages = 0
    max_active_pages = 0
    total_pages = 0

    class FakeLocator:
        """Return no child iframes for the synthetic browser page."""

        async def evaluate_all(self, _script: str) -> list[str]:
            """Return an empty iframe list."""
            return []

    class FakePage:
        """Model one browser tab while tracking active-tab accounting."""

        def on(self, _event: str, _callback) -> None:
            """Accept request listeners without emitting synthetic requests."""
            return None

        async def goto(self, _url: str, **_kwargs) -> None:
            """Keep the tab alive briefly so concurrent visits overlap."""
            await asyncio.sleep(0.02)

        async def wait_for_timeout(self, _milliseconds: int) -> None:
            """Yield control without adding test latency."""
            await asyncio.sleep(0)

        def locator(self, _selector: str) -> FakeLocator:
            """Return the fake iframe locator."""
            return FakeLocator()

        async def close(self) -> None:
            """Mark this synthetic tab as closed."""
            nonlocal active_pages
            active_pages -= 1

    class FakeContext:
        """Create tabs from one shared browser context and track concurrency."""

        async def new_page(self) -> FakePage:
            """Create one synthetic tab and update active-tab counters."""
            nonlocal active_pages, max_active_pages, total_pages
            active_pages += 1
            total_pages += 1
            max_active_pages = max(max_active_pages, active_pages)
            return FakePage()

    session_type = getattr(discovery, "BrowserSession", None)
    assert session_type is not None, "bounded shared BrowserSession is not implemented"
    session = session_type(context=FakeContext(), page_semaphore=asyncio.Semaphore(2))

    await asyncio.gather(
        *(
            discovery._visit_browser_target(
                session,
                f"https://example.test/{index}",
                timeout_ms=1_000,
                settle_ms=0,
            )
            for index in range(6)
        )
    )

    assert total_pages == 6
    assert max_active_pages == 2
