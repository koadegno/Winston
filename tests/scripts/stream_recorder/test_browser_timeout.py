"""Regression tests for bounded Playwright target execution."""

import asyncio
import time

import pytest

from scripts.stream_recorder.discovery import BrowserSession, _visit_browser_target


@pytest.mark.asyncio
async def test_browser_target_hard_deadline_releases_slot() -> None:
    """A wedged browser navigation must stop and release its global tab slot."""
    closed = asyncio.Event()

    class Locator:
        """Return no iframes for the synthetic page."""

        async def evaluate_all(self, _script: str) -> list[str]:
            """Return an empty iframe list."""
            return []

    class Page:
        """Model a page whose navigation never completes."""

        def on(self, _event: str, _callback) -> None:
            """Accept Playwright event handlers without emitting requests."""
            return None

        async def goto(self, _url: str, **_kwargs) -> None:
            """Block forever to model a wedged Playwright navigation."""
            await asyncio.Event().wait()

        async def wait_for_timeout(self, _milliseconds: int) -> None:
            """Yield without adding artificial latency."""
            await asyncio.sleep(0)

        def locator(self, _selector: str) -> Locator:
            """Return the synthetic iframe locator."""
            return Locator()

        async def close(self) -> None:
            """Record that timed-out page cleanup ran."""
            closed.set()

    class Context:
        """Create the synthetic wedged page."""

        async def new_page(self) -> Page:
            """Return one synthetic page."""
            return Page()

    session = BrowserSession(context=Context(), page_semaphore=asyncio.Semaphore(1))
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        await _visit_browser_target(
            session,
            "https://example.test/wedged",
            timeout_ms=20_000,
            settle_ms=5_000,
            hard_timeout_ms=50,
        )

    assert time.monotonic() - started < 0.25
    assert closed.is_set()
    assert session.page_semaphore._value == 1


@pytest.mark.asyncio
async def test_browser_page_close_is_bounded_after_visit() -> None:
    """A page whose close call wedges must still release the global browser slot."""

    class Locator:
        """Return no iframes for the synthetic page."""

        async def evaluate_all(self, _script: str) -> list[str]:
            """Return an empty iframe list."""
            return []

    class Page:
        """Model a successful page whose close call never completes."""

        def on(self, _event: str, _callback) -> None:
            """Accept request listeners without emitting requests."""
            return None

        async def goto(self, _url: str, **_kwargs) -> None:
            """Complete navigation immediately."""
            return None

        async def wait_for_timeout(self, _milliseconds: int) -> None:
            """Yield without adding artificial latency."""
            await asyncio.sleep(0)

        def locator(self, _selector: str) -> Locator:
            """Return the synthetic iframe locator."""
            return Locator()

        async def close(self) -> None:
            """Block forever to model a stuck browser cleanup call."""
            await asyncio.Event().wait()

    class Context:
        """Create the synthetic page with a wedged close call."""

        async def new_page(self) -> Page:
            """Return one synthetic page."""
            return Page()

    session = BrowserSession(context=Context(), page_semaphore=asyncio.Semaphore(1))
    started = time.monotonic()

    streams, iframes = await _visit_browser_target(
        session,
        "https://example.test/close-wedged",
        timeout_ms=1_000,
        settle_ms=0,
        hard_timeout_ms=100,
        close_timeout_seconds=0.05,
    )

    assert streams == set()
    assert iframes == set()
    assert time.monotonic() - started < 0.25
    assert session.page_semaphore._value == 1
