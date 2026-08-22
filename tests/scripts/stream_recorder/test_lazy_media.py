import asyncio

import pytest

import scripts.stream_recorder.discovery as discovery


@pytest.mark.asyncio
async def test_browser_target_reveals_lazy_embed_before_media_activation() -> None:
    """A lazy iframe must be materialized in the same page before media activation runs."""
    hls_url = "https://7005.szczecin.pl/camss/streams/czersk.m3u8"
    revealed = False

    class Request:
        """Represent the HLS request emitted by the lazy embedded player."""

        url = hls_url

    class EmptyLocator:
        """Represent a selector that has no matching visible element."""

        @property
        def first(self) -> "EmptyLocator":
            """Return the same empty locator for Playwright's ``first`` API."""
            return self

        async def count(self) -> int:
            """Report no matching element."""
            return 0

        async def is_visible(self) -> bool:
            """Report that an absent element is not visible."""
            return False

    class VideoLocator:
        """Represent the video element inside the materialized lazy iframe."""

        @property
        def first(self) -> "VideoLocator":
            """Return this video as the first matching play control."""
            return self

        async def count(self) -> int:
            """Report one video element."""
            return 1

        async def is_visible(self) -> bool:
            """Report that the embedded player video is visible."""
            return True

        async def click(self, **_kwargs) -> None:
            """Emit the HLS request when the user-gesture fallback clicks the video."""
            page.request_callback(Request())

        async def evaluate_all(self, _script: str) -> list[object]:
            """Model programmatic playback without changing the already emitted request."""
            return []

    class MainFrame:
        """Represent the source page before its lazy iframe is materialized."""

        url = "https://czersk.pl/strona/552-kamera-line"

        def locator(self, _selector: str) -> EmptyLocator:
            """Expose no playable media on the source frame itself."""
            return EmptyLocator()

    class PlayerFrame:
        """Represent the Stream360 iframe after lazy loading has been triggered."""

        url = "https://stream360.pl/v/Czersk/index1.php"

        def locator(self, selector: str):
            """Expose a playable video only for the video selector."""
            if selector == "video":
                return VideoLocator()
            return EmptyLocator()

    class LazyMediaLocator:
        """Materialize the iframe when discovery reveals lazy media in the viewport."""

        async def evaluate_all(self, _script: str) -> int:
            """Add the embedded player frame and report one revealed lazy element."""
            nonlocal revealed
            revealed = True
            page.frames = [MainFrame(), PlayerFrame()]
            return 1

    class FakePage:
        """Expose a page whose player frame exists only after lazy media is revealed."""

        request_callback = None

        def __init__(self) -> None:
            """Start with only the main frame, as a lazy iframe has not loaded yet."""
            self.frames = [MainFrame()]

        def on(self, event: str, callback) -> None:
            """Store the network request listener used by HLS discovery."""
            assert event == "request"
            self.request_callback = callback

        async def goto(self, _url: str, **_kwargs) -> None:
            """Load the source page without materializing the lazy player."""
            return None

        def locator(self, selector: str) -> LazyMediaLocator:
            """Expose the page-level lazy-media collection expected by discovery."""
            assert selector == "iframe, video"
            return LazyMediaLocator()

        async def close(self) -> None:
            """Close the synthetic source page."""
            return None

    page = FakePage()

    class FakeContext:
        """Return the one synthetic page used by this regression test."""

        async def new_page(self) -> FakePage:
            """Return the prebuilt lazy-media page."""
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

    assert revealed
    assert streams == {hls_url}
