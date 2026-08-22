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


def test_default_browser_deadline_leaves_time_for_late_player_activation():
    """The global page budget must outlive navigation, settle, and interactive player startup."""
    minimum_budget_ms = (
        discovery.DEFAULT_BROWSER_NAVIGATION_TIMEOUT_MS
        + discovery.DEFAULT_BROWSER_SETTLE_MS
        + discovery._MEDIA_ACTIVATION_GRACE_MS
        + 10_000
    )

    assert discovery.DEFAULT_BROWSER_TARGET_HARD_TIMEOUT_MS >= minimum_budget_ms


@pytest.mark.asyncio
async def test_http_discovery_fetches_only_source_page():
    """HTTP discovery reads one page response and never follows embedded third-party URLs."""
    calls: list[str] = []

    class Response:
        """Provide one HTML response containing an iframe and HLS URL."""

        text = (
            '<iframe src="https://third-party.example/player"></iframe>'
            '<script>const stream="/live/camera.m3u8";</script>'
        )
        url = "https://example.test/page"

        def raise_for_status(self) -> None:
            """Model a successful response."""
            return None

    class Client:
        """Record exactly which URLs HTTP discovery requests."""

        async def get(self, url: str) -> Response:
            """Return the synthetic source page."""
            calls.append(url)
            return Response()

    assert await discover_http("https://example.test/page", client=Client()) == {
        "https://example.test/live/camera.m3u8"
    }
    assert calls == ["https://example.test/page"]


@pytest.mark.asyncio
async def test_discover_page_runs_http_before_browser(monkeypatch):
    """Single-page discovery finishes HTTP before starting its one browser page."""
    order: list[str] = []
    sentinel = object()

    async def fake_http(_url: str, **_kwargs) -> set[str]:
        """Record completion of the HTTP stage."""
        order.append("http-start")
        await asyncio.sleep(0)
        order.append("http-done")
        return {"https://example.test/static.m3u8"}

    async def fake_browser(_url: str, *, browser, **_kwargs) -> set[str]:
        """Assert browser observation starts only after HTTP completed."""
        assert browser is sentinel
        order.append("browser-start")
        return {"https://example.test/dynamic.m3u8"}

    monkeypatch.setattr(discovery, "discover_http", fake_http)
    monkeypatch.setattr(discovery, "discover_browser", fake_browser)
    streams = await discovery.discover_page(
        "https://example.test/page",
        browser=sentinel,
        client=object(),
    )

    assert order == ["http-start", "http-done", "browser-start"]
    assert streams == {
        "https://example.test/static.m3u8",
        "https://example.test/dynamic.m3u8",
    }


@pytest.mark.asyncio
async def test_browser_target_logs_queue_open_result_and_close(capsys):
    """Browser visits expose enough stderr progress to diagnose queued or slow sources."""

    class FakePage:
        """Provide the Playwright methods used by one synthetic source page."""

        frames: list[object] = []

        def on(self, _event: str, _callback) -> None:
            """Accept network listeners without emitting a stream."""
            return None

        async def goto(self, _url: str, **_kwargs) -> None:
            """Simulate an immediately loaded page."""
            return None

        async def wait_for_timeout(self, _milliseconds: int) -> None:
            """Yield control once for deterministic async behavior."""
            await asyncio.sleep(0)

        async def close(self) -> None:
            """Simulate closing the browser tab."""
            return None

    class FakeContext:
        """Create one synthetic page from the shared context."""

        async def new_page(self) -> FakePage:
            """Return a new synthetic browser page."""
            return FakePage()

    session = discovery.BrowserSession(
        context=FakeContext(),
        page_semaphore=asyncio.Semaphore(1),
    )
    url = "https://example.test/player"

    await discovery._visit_browser_target(session, url, settle_ms=0)

    stderr = capsys.readouterr().err
    assert f"queued {url}" in stderr
    assert f"open {url}" in stderr
    assert "done in" in stderr
    assert f"closed {url}" in stderr


@pytest.mark.asyncio
async def test_browser_target_activates_media_when_initial_load_emits_no_hls():
    """A player that waits for Play is activated without opening another browser page."""
    hls_url = "https://7005.szczecin.pl/camss/streams/czersk.m3u8"
    activated = False

    class Request:
        """Represent the HLS request emitted after synthetic media activation."""

        url = hls_url

    class FakeVideoLocator:
        """Expose one video element whose activation starts the HLS request."""

        async def count(self) -> int:
            """Report one video element in the synthetic iframe."""
            return 1

        async def evaluate_all(self, _script: str) -> list[object]:
            """Model calling play() on all video elements."""
            nonlocal activated
            activated = True
            page.request_callback(Request())
            return []

    class FakeFrame:
        """Expose the video locator used by the media-activation fallback."""

        url = "https://stream360.pl/v/Czersk/index1.php"

        def locator(self, selector: str) -> FakeVideoLocator:
            """Return the synthetic video collection for the video selector."""
            assert selector == "video"
            return FakeVideoLocator()

    class FakePage:
        """Load without HLS until the embedded video is explicitly activated."""

        request_callback = None

        def __init__(self) -> None:
            """Attach one embedded player frame."""
            self.frames = [FakeFrame()]

        def on(self, event: str, callback) -> None:
            """Store the request listener used by discovery."""
            assert event == "request"
            self.request_callback = callback

        async def goto(self, _url: str, **_kwargs) -> None:
            """Load the page without starting media automatically."""
            return None

        async def close(self) -> None:
            """Close the synthetic page."""
            return None

    page = FakePage()

    class FakeContext:
        """Return exactly one source page from the shared context."""

        async def new_page(self) -> FakePage:
            """Return the prebuilt synthetic page."""
            return page

    session = discovery.BrowserSession(
        context=FakeContext(),
        page_semaphore=asyncio.Semaphore(1),
    )

    streams = await discovery._visit_browser_target(
        session,
        "https://czersk.pl/strona/552-kamera-line",
        settle_ms=1,
        hard_timeout_ms=1_000,
    )

    assert activated
    assert streams == {hls_url}


@pytest.mark.asyncio
async def test_browser_targets_share_one_context_and_respect_page_limit():
    """Source pages reuse one context and never exceed the configured active-tab limit."""
    active_pages = 0
    max_active_pages = 0
    total_pages = 0

    class FakePage:
        """Model one browser tab while tracking active-tab accounting."""

        frames: list[object] = []

        def on(self, _event: str, _callback) -> None:
            """Accept request listeners without emitting synthetic requests."""
            return None

        async def goto(self, _url: str, **_kwargs) -> None:
            """Keep the tab alive briefly so concurrent visits overlap."""
            await asyncio.sleep(0.02)

        async def wait_for_timeout(self, _milliseconds: int) -> None:
            """Yield control without adding test latency."""
            await asyncio.sleep(0)

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

    session = discovery.BrowserSession(
        context=FakeContext(),
        page_semaphore=asyncio.Semaphore(2),
    )

    await asyncio.gather(
        *(
            discovery._visit_browser_target(
                session,
                f"https://example.test/{index}",
                settle_ms=0,
            )
            for index in range(6)
        )
    )

    assert total_pages == 6
    assert max_active_pages == 2
