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
    """Browser crawling follows iframe URLs as isolated targets."""
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
async def test_http_and_browser_layers_run_concurrently(monkeypatch):
    """Direct HTTP and browser observation start together for one source page."""
    both_started = asyncio.Event()
    started: set[str] = set()
    sentinel = object()

    async def wait_for_peer(name: str, result: set[str]) -> set[str]:
        """Wait until both independent discovery layers have entered."""
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)
        return result

    async def fake_http(_url: str) -> set[str]:
        """Simulate the direct HTTP layer."""
        return await wait_for_peer("http", {"https://example.test/http.m3u8"})

    async def fake_browser(_url: str, *, browser) -> set[str]:
        """Simulate the browser network-observation layer."""
        assert browser is sentinel
        return await wait_for_peer("browser", {"https://example.test/browser.m3u8"})

    monkeypatch.setattr(discovery, "discover_http", fake_http)
    monkeypatch.setattr(discovery, "discover_browser", fake_browser)
    streams = await discovery.discover_page("https://example.test/page", browser=sentinel)
    assert streams == {
        "https://example.test/http.m3u8",
        "https://example.test/browser.m3u8",
    }


@pytest.mark.asyncio
async def test_browser_target_visits_share_context_and_limit_active_pages():
    """Browser target visits reuse one context and cap concurrent pages."""
    max_pages = 4
    state = {
        "active_pages": 0,
        "max_active_pages": 0,
        "new_context_calls": 0,
    }

    class FakeLocator:
        """Return no iframes for the synthetic browser page."""

        async def evaluate_all(self, _expression: str) -> list[str]:
            """Return an empty iframe URL list."""
            return []

    class FakePage:
        """Track synthetic page lifetime for concurrency assertions."""

        def __init__(self) -> None:
            """Initialize one synthetic page as open and inactive."""
            self._active = False
            self._closed = False

        def on(self, _event: str, _callback) -> None:
            """Accept request listeners without emitting synthetic requests."""

        async def goto(self, _url: str, **_kwargs) -> None:
            """Mark this synthetic page as actively navigating."""
            self._active = True
            state["active_pages"] += 1
            state["max_active_pages"] = max(
                state["max_active_pages"],
                state["active_pages"],
            )

        async def wait_for_timeout(self, _timeout_ms: int) -> None:
            """Keep pages overlapping long enough to measure peak concurrency."""
            await asyncio.sleep(0.02)

        def locator(self, _selector: str) -> FakeLocator:
            """Return the synthetic iframe locator."""
            return FakeLocator()

        async def close(self) -> None:
            """Close the page once and update the active-page count."""
            if self._closed:
                return
            self._closed = True
            if self._active:
                state["active_pages"] -= 1
                self._active = False

    class FakeContext:
        """Create synthetic pages and close any pages it owns."""

        def __init__(self) -> None:
            """Initialize an empty synthetic page collection."""
            self.pages: list[FakePage] = []

        async def new_page(self) -> FakePage:
            """Create and remember one synthetic page."""
            page = FakePage()
            self.pages.append(page)
            return page

        async def close(self) -> None:
            """Close every page created by this synthetic context."""
            await asyncio.gather(*(page.close() for page in self.pages))

    class FakeBrowserSession:
        """Expose both old and desired browser APIs to detect context fan-out."""

        def __init__(self) -> None:
            """Create one shared context and the desired global page semaphore."""
            self.context = FakeContext()
            self.page_semaphore = asyncio.Semaphore(max_pages)

        async def new_context(self) -> FakeContext:
            """Count legacy per-target context creation and return a fresh context."""
            state["new_context_calls"] += 1
            return FakeContext()

    browser = FakeBrowserSession()
    await asyncio.gather(
        *(
            discovery._visit_browser_target(
                browser,
                f"https://example.test/{index}",
                timeout_ms=1_000,
                settle_ms=1,
            )
            for index in range(8)
        )
    )

    assert state["new_context_calls"] == 0
    assert state["max_active_pages"] <= max_pages
    assert state["active_pages"] == 0
