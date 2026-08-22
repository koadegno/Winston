import asyncio

import pytest

import scripts.stream_recorder.discovery as discovery


@pytest.mark.asyncio
async def test_browser_clicks_play_before_programmatic_video_play():
    """A user-gesture player must be clicked before a potentially blocking video.play() fallback."""
    hls_url = "https://7005.szczecin.pl/camss/streams/czersk.m3u8"
    clicked = False
    programmatic_play_called = False

    class Request:
        """Represent the HLS request emitted by the synthetic Stream360 player."""

        url = hls_url

    class FakeLocator:
        """Model only the visible video control used by the embedded player."""

        def __init__(self, selector: str) -> None:
            """Remember which selector Playwright is probing."""
            self.selector = selector

        @property
        def first(self) -> "FakeLocator":
            """Return the only synthetic locator match."""
            return self

        async def count(self) -> int:
            """Expose one match only for the video element itself."""
            return 1 if self.selector == "video" else 0

        async def is_visible(self) -> bool:
            """Keep the synthetic video control visible."""
            return self.selector == "video"

        async def click(self, **_kwargs) -> None:
            """Emit HLS immediately after the user-gesture style click."""
            nonlocal clicked
            clicked = True
            page.request_callback(Request())

        async def evaluate_all(self, _script: str) -> list[object]:
            """Model a video.play() call that hangs longer than the source-page deadline."""
            nonlocal programmatic_play_called
            programmatic_play_called = True
            await asyncio.sleep(1.0)
            return []

    class FakeFrame:
        """Represent the Stream360 iframe already loaded inside the source page."""

        url = "https://stream360.pl/v/Czersk/index1.php"

        def locator(self, selector: str) -> FakeLocator:
            """Return a selector-aware synthetic locator."""
            return FakeLocator(selector)

    class FakePage:
        """Load successfully but emit no HLS until the embedded video is clicked."""

        request_callback = None

        def __init__(self) -> None:
            """Attach the existing Stream360 frame without opening another page."""
            self.frames = [FakeFrame()]

        def on(self, event: str, callback) -> None:
            """Store the request listener installed by discovery."""
            assert event == "request"
            self.request_callback = callback

        async def goto(self, _url: str, **_kwargs) -> None:
            """Model an immediately loaded source page."""
            return None

        async def close(self) -> None:
            """Model deterministic browser-tab cleanup."""
            return None

    page = FakePage()

    class FakeContext:
        """Return exactly one page from the shared browser context."""

        async def new_page(self) -> FakePage:
            """Return the prebuilt source page."""
            return page

    session = discovery.BrowserSession(
        context=FakeContext(),
        page_semaphore=asyncio.Semaphore(1),
    )

    streams = await discovery._visit_browser_target(
        session,
        "https://czersk.pl/strona/552-kamera-line",
        settle_ms=1,
        hard_timeout_ms=250,
    )

    assert clicked
    assert not programmatic_play_called
    assert streams == {hls_url}
