"""Regression tests for bounded Playwright source-page execution."""

import asyncio
import time

import pytest

from scripts.stream_recorder import discovery
from scripts.stream_recorder.discovery import BrowserSession, _visit_browser_target


@pytest.mark.asyncio
async def test_browser_target_hard_deadline_releases_slot() -> None:
    """A wedged browser navigation must stop, close its page, and release its tab slot."""
    closed = asyncio.Event()

    class Page:
        """Model a page whose navigation never completes."""

        def on(self, _event: str, _callback) -> None:
            """Accept Playwright event handlers without emitting requests."""
            return None

        async def goto(self, _url: str, **_kwargs) -> None:
            """Block forever to model a wedged Playwright navigation."""
            await asyncio.Event().wait()

        async def close(self) -> None:
            """Record that timed-out page cleanup completed."""
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
async def test_hard_deadline_does_not_wait_for_cancellation_resistant_navigation() -> None:
    """A navigation that suppresses cancellation must not extend the browser target deadline."""

    class Page:
        """Model a Playwright call that ignores cancellation before eventually returning."""

        def on(self, _event: str, _callback) -> None:
            """Accept request listeners without emitting requests."""
            return None

        async def goto(self, _url: str, **_kwargs) -> None:
            """Suppress cancellation to reproduce a stuck transport operation."""
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.20)

        async def close(self) -> None:
            """Close the synthetic page immediately."""
            return None

    class Context:
        """Create the cancellation-resistant synthetic page."""

        async def new_page(self) -> Page:
            """Return one synthetic page."""
            return Page()

    session = BrowserSession(context=Context(), page_semaphore=asyncio.Semaphore(1))
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        await _visit_browser_target(
            session,
            "https://example.test/cancellation-resistant",
            timeout_ms=20_000,
            settle_ms=0,
            hard_timeout_ms=30,
        )

    # A cooperative asyncio.timeout() implementation would incorrectly take ~0.23s here.
    assert time.monotonic() - started < 0.12


@pytest.mark.asyncio
async def test_unconfirmed_page_close_poison_browser_instead_of_reusing_slot() -> None:
    """A wedged page close must make an unrecoverable synthetic session fail closed."""

    class Page:
        """Model a successful page whose close call never completes."""

        def on(self, _event: str, _callback) -> None:
            """Accept request listeners without emitting requests."""
            return None

        async def goto(self, _url: str, **_kwargs) -> None:
            """Complete navigation immediately."""
            return None

        async def close(self) -> None:
            """Block forever to model a stuck renderer cleanup call."""
            await asyncio.Event().wait()

    class Context:
        """Create the page and support bounded context cleanup."""

        async def new_page(self) -> Page:
            """Return one synthetic page."""
            return Page()

        async def close(self) -> None:
            """Close the synthetic context immediately."""
            return None

    session = BrowserSession(context=Context(), page_semaphore=asyncio.Semaphore(1))
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="browser recycle failed"):
        await _visit_browser_target(
            session,
            "https://example.test/close-wedged",
            timeout_ms=1_000,
            settle_ms=0,
            hard_timeout_ms=100,
            close_timeout_seconds=0.05,
        )

    assert time.monotonic() - started < 0.25
    assert session.failed is not None
    assert session.page_semaphore._value == 1


@pytest.mark.asyncio
async def test_browser_shutdown_is_internally_bounded(monkeypatch) -> None:
    """Shared browser shutdown must finish under its own deadline when resources wedge."""

    class StuckResource:
        """Model a Playwright resource whose close call never returns."""

        async def close(self) -> None:
            """Block forever until the production cleanup deadline cancels this call."""
            await asyncio.Event().wait()

    monkeypatch.setattr(discovery, "DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS", 0.03)
    session = BrowserSession(
        context=StuckResource(),
        browser=StuckResource(),
        page_semaphore=asyncio.Semaphore(1),
    )
    started = time.monotonic()

    await session.shutdown()

    assert time.monotonic() - started < 0.15
    assert session.failed == "browser shutdown did not complete cleanly"
