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


@pytest.mark.asyncio
async def test_browser_waits_for_delayed_hls_when_player_controls_are_not_ready():
    """A late player initialization gets a grace window even before controls/videos exist."""
    hls_url = "https://online2.kamery24.org/cam/debica.m3u8"
    delayed_task: asyncio.Task[None] | None = None

    class Request:
        """Represent the HLS request emitted after delayed third-party player initialization."""

        url = hls_url

    class EmptyLocator:
        """Represent a player whose controls and video element are not attached yet."""

        @property
        def first(self) -> "EmptyLocator":
            """Return the synthetic empty locator itself."""
            return self

        async def count(self) -> int:
            """Report that the requested control is not present yet."""
            return 0

    class DelayedFrame:
        """Represent a frame whose media controls appear later than initial DOM load."""

        url = "https://www.kamery24.org/i24/index.php/test/82-debica-rynek"

        def locator(self, _selector: str) -> EmptyLocator:
            """Return no immediately usable play control or video element."""
            return EmptyLocator()

    class DelayedPage:
        """Emit HLS shortly after the initial settle window without a clickable control."""

        request_callback = None

        def __init__(self) -> None:
            """Expose one not-yet-initialized frame."""
            self.frames = [DelayedFrame()]

        def on(self, event: str, callback) -> None:
            """Capture the network listener installed by stream discovery."""
            assert event == "request"
            self.request_callback = callback

        async def goto(self, _url: str, **_kwargs) -> None:
            """Schedule an HLS request after the short initial settle window expires."""
            nonlocal delayed_task

            async def emit_later() -> None:
                """Model a slow third-party player attaching and starting its stream."""
                # This delay is intentionally longer than the 1 ms initial settle below, so the
                # test only passes if discovery keeps the page alive for the fallback grace window.
                await asyncio.sleep(0.025)
                assert self.request_callback is not None
                self.request_callback(Request())

            delayed_task = asyncio.create_task(emit_later())

        async def close(self) -> None:
            """Model deterministic page cleanup."""
            return None

    page = DelayedPage()

    class FakeContext:
        """Provide the single delayed player page."""

        async def new_page(self) -> DelayedPage:
            """Return the delayed page."""
            return page

    session = discovery.BrowserSession(
        context=FakeContext(),
        page_semaphore=asyncio.Semaphore(1),
    )

    streams = await discovery._visit_browser_target(
        session,
        "https://www.kamery24.org/i24/index.php/test/82-debica-rynek",
        settle_ms=1,
        hard_timeout_ms=150,
    )
    # Freeze the value returned by discovery; the request callback owns the live set and may fire
    # after the function has already closed the page, which is precisely the regression under test.
    returned_streams = set(streams)

    assert delayed_task is not None
    await delayed_task
    assert returned_streams == {hls_url}
