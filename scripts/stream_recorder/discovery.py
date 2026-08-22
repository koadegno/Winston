"""Discover public HLS streams from HTML pages and browser network traffic."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import time
from typing import Any, TypeVar
from urllib.parse import urljoin, urlparse

import httpx

from .log import log


DEFAULT_BROWSER_CONCURRENCY = 4
DEFAULT_BROWSER_NAVIGATION_TIMEOUT_MS = 10_000
DEFAULT_BROWSER_SETTLE_MS = 4_000
# Heavy public players can consume the navigation budget before exposing their play control.
# Keep a separate total budget so interaction and the post-click HLS grace period still run.
DEFAULT_BROWSER_TARGET_HARD_TIMEOUT_MS = 30_000
DEFAULT_BROWSER_PAGE_CLOSE_TIMEOUT_SECONDS = 2.0
DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
DEFAULT_HTTP_CONNECTIONS = 32
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_HTTP_TOTAL_TIMEOUT_SECONDS = 12.0
_BROWSER_STREAM_GRACE_MS = 750
_MEDIA_ACTIVATION_GRACE_MS = 4_000
_MAX_HLS_URL_TOKEN_LENGTH = 4096
_HLS_MARKER = ".m3u8"
_LEFT_URL_BOUNDARIES = frozenset(" \t\r\n'\"<>`()[]{},;=")
_RIGHT_URL_BOUNDARIES = frozenset(" \t\r\n'\"<>`()[]{},;")
_BLOCKED_STREAM_HOST_SUFFIXES = (
    "facebook.com",
    "fbcdn.net",
    "google.com",
    "googleapis.com",
    "google-analytics.com",
    "googleadservices.com",
    "googlesyndication.com",
    "googletagmanager.com",
    "gstatic.com",
    "doubleclick.net",
    "youtube.com",
    "youtu.be",
    "googlevideo.com",
)
_PLAY_CONTROL_SELECTORS = (
    "button.vjs-big-play-button",
    ".vjs-big-play-button",
    ".plyr__control--overlaid",
    "button[aria-label*='play' i]",
    "button[title*='play' i]",
    "video",
)

_T = TypeVar("_T")


class _BoundedOperationTimeout(TimeoutError):
    """Signal that an awaitable exceeded a wall-clock deadline without awaiting cancellation."""


@dataclass(slots=True)
class BrowserSession:
    """Share one Chromium process while bounding isolated active source sessions."""

    context: Any | None
    page_semaphore: asyncio.Semaphore
    browser: Any | None = None
    browser_type: Any | None = None
    generation: int = 0
    failed: str | None = None
    restart_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def recycle(self, *, expected_generation: int, reason: str) -> None:
        """Recycle a poisoned Chromium process before any new source session is created."""
        async with self.restart_lock:
            # Another page may already have recycled the shared browser while this task waited.
            if expected_generation != self.generation:
                return

            self.generation += 1
            log(f"[browser] RECYCLE generation={expected_generation}: {reason}")
            old_context = self.context
            old_browser = self.browser
            self.context = None
            self.browser = None

            # ``context`` is retained only as a compatibility seam for synthetic/unit callers.
            # Real browser sessions use temporary isolated contexts that are closed per source.
            context_closed = await _close_resource_bounded(
                old_context,
                "legacy browser context",
                timeout_seconds=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
            )
            browser_closed = await _close_resource_bounded(
                old_browser,
                "Chromium",
                timeout_seconds=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
            )
            if not context_closed or not browser_closed or self.browser_type is None:
                self.failed = f"browser recycle failed after: {reason}"
                log(f"[browser] FATAL {self.failed}")
                return

            try:
                new_browser = await _await_bounded(
                    self.browser_type.launch(headless=True),
                    timeout_seconds=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
                    operation="Chromium restart",
                )
            except Exception as exc:
                self.failed = f"browser restart failed: {type(exc).__name__}: {exc}"
                log(f"[browser] FATAL {self.failed}")
                return

            self.browser = new_browser
            self.context = None
            self.failed = None
            log(f"[browser] recycle complete; generation={self.generation}")

    async def shutdown(self) -> None:
        """Bound shared Chromium shutdown so command completion cannot hang forever."""
        context = self.context
        browser = self.browser
        self.context = None
        self.browser = None

        context_closed = await _close_resource_bounded(
            context,
            "legacy browser context",
            timeout_seconds=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
        )
        browser_closed = await _close_resource_bounded(
            browser,
            "Chromium",
            timeout_seconds=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
        )
        if not context_closed or not browser_closed:
            self.failed = "browser shutdown did not complete cleanly"
            log(f"[browser] WARN {self.failed}")


def _normalize_escaped_url(value: str) -> str:
    """Normalize slash-escaped and HTML-escaped URLs found in page source."""
    return value.replace("\\/", "/").replace("&amp;", "&")


def _host_matches_suffix(host: str, suffix: str) -> bool:
    """Return whether a hostname is exactly or transitively under a blocked suffix."""
    return host == suffix or host.endswith(f".{suffix}")


def _extract_hls_token(text: str, marker_start: int) -> str:
    """Extract one bounded URL-like token around an already located ``.m3u8`` marker."""
    marker_end = marker_start + len(_HLS_MARKER)
    left_limit = max(0, marker_start - _MAX_HLS_URL_TOKEN_LENGTH)
    right_limit = min(len(text), marker_end + _MAX_HLS_URL_TOKEN_LENGTH)

    left = marker_start
    while left > left_limit and text[left - 1] not in _LEFT_URL_BOUNDARIES:
        left -= 1

    right = marker_end
    while right < right_limit and text[right] not in _RIGHT_URL_BOUNDARIES:
        right += 1

    raw = _normalize_escaped_url(text[left:right]).strip()
    lowered = raw.lower()
    http_position = max(lowered.rfind("https://"), lowered.rfind("http://"))
    if http_position >= 0:
        # JavaScript can prefix an unquoted URL with a label such as ``src:https://...``.
        raw = raw[http_position:]
    return raw


def is_allowed_hls_url(url: str) -> bool:
    """Accept HTTP(S) HLS URLs while excluding unrelated Google/Facebook/YouTube services."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    if _HLS_MARKER not in url.lower():
        return False
    return not any(_host_matches_suffix(host, suffix) for suffix in _BLOCKED_STREAM_HOST_SUFFIXES)


def extract_m3u8_urls(text: str, base_url: str) -> set[str]:
    """Extract HLS playlist URLs in linear time without regex backtracking on large pages."""
    urls: set[str] = set()
    lowered = text.lower()
    cursor = 0
    while True:
        marker_start = lowered.find(_HLS_MARKER, cursor)
        if marker_start < 0:
            break

        raw = _extract_hls_token(text, marker_start)
        if raw.startswith("//"):
            raw = "https:" + raw
        candidate = urljoin(base_url, raw)
        if is_allowed_hls_url(candidate):
            urls.add(candidate)

        # Continue after this marker; token scanning is independently bounded to 4 KiB.
        cursor = marker_start + len(_HLS_MARKER)
    return urls


@asynccontextmanager
async def http_session(
    *,
    max_connections: int = DEFAULT_HTTP_CONNECTIONS,
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield one shared true-async HTTP client with explicit limits and timeouts."""
    if max_connections < 1:
        raise ValueError("max_connections must be at least 1")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")

    timeout = httpx.Timeout(
        timeout_seconds,
        connect=min(5.0, timeout_seconds),
        pool=min(5.0, timeout_seconds),
    )
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=min(max_connections, 16),
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/127 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/vnd.apple.mpegurl,*/*;q=0.8",
    }
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        limits=limits,
        headers=headers,
    ) as client:
        yield client


async def fetch_text(
    url: str,
    *,
    client: Any | None = None,
    total_timeout_seconds: float = DEFAULT_HTTP_TOTAL_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    """Fetch text through async HTTP with a strict total wall-clock request deadline."""
    if total_timeout_seconds <= 0:
        raise ValueError("total_timeout_seconds must be greater than 0")
    if client is None:
        # Standalone callers still get true async I/O; batch callers pass one shared client.
        async with http_session() as local_client:
            return await fetch_text(
                url,
                client=local_client,
                total_timeout_seconds=total_timeout_seconds,
            )

    response = await _await_bounded(
        client.get(url),
        timeout_seconds=total_timeout_seconds,
        operation=f"HTTP GET {url}",
    )
    response.raise_for_status()
    return response.text, str(response.url)


async def discover_http(
    url: str,
    *,
    client: Any | None = None,
    label: str | None = None,
) -> set[str]:
    """Discover HLS URLs visible directly in one source page response."""
    prefix = f"[http][{label}]" if label else "[http]"
    log(f"{prefix} GET {url}")
    started = time.monotonic()
    try:
        body, final_url = await fetch_text(url, client=client)
    except Exception as exc:
        elapsed = time.monotonic() - started
        log(f"{prefix} ERROR after {elapsed:.1f}s {url}: {type(exc).__name__}: {exc}")
        raise

    streams = extract_m3u8_urls(body, final_url)
    elapsed = time.monotonic() - started
    log(f"{prefix} done in {elapsed:.1f}s: {len(streams)} HLS candidate(s)")
    return streams


def _consume_background_task(task: asyncio.Task[Any]) -> None:
    """Consume a detached task result so abandoned transport calls do not emit warnings."""
    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        # Detached transport calls are diagnostic cleanup only after the caller timed out.
        return


async def _await_bounded(
    awaitable: Awaitable[_T],
    *,
    timeout_seconds: float,
    operation: str,
) -> _T:
    """Await an operation for a wall-clock bound without waiting for cancellation cooperation."""
    if timeout_seconds <= 0:
        raise _BoundedOperationTimeout(f"{operation} deadline exhausted")

    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
    if task not in done:
        # Crucially, cancel but DO NOT await this task. Some network/Playwright transport calls can
        # suppress or delay cancellation, which would defeat asyncio.timeout()/wait_for().
        task.cancel()
        task.add_done_callback(_consume_background_task)
        raise _BoundedOperationTimeout(
            f"{operation} exceeded {timeout_seconds:.3f}s"
        )
    return task.result()


def _remaining_seconds(deadline: float) -> float:
    """Return remaining monotonic seconds for one hard browser-page deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _BoundedOperationTimeout("browser source-page deadline exhausted")
    return remaining


async def _run_cleanup_bounded(
    action: Callable[[], Awaitable[Any]],
    label: str,
    *,
    timeout_seconds: float,
) -> bool:
    """Run one asynchronous cleanup action without waiting for cancellation cooperation."""
    try:
        await _await_bounded(
            action(),
            timeout_seconds=timeout_seconds,
            operation=label,
        )
    except _BoundedOperationTimeout:
        log(f"[browser] WARN {label} timeout after {timeout_seconds:.1f}s")
        return False
    except Exception as exc:
        log(f"[browser] WARN {label} failed: {type(exc).__name__}: {exc}")
        return False
    return True


async def _close_resource_bounded(
    resource: Any | None,
    label: str,
    *,
    timeout_seconds: float,
) -> bool:
    """Close a Playwright resource without allowing cleanup to block forever."""
    if resource is None:
        return True
    return await _run_cleanup_bounded(
        resource.close,
        label,
        timeout_seconds=timeout_seconds,
    )


async def _close_page_bounded(
    page: Any,
    url: str,
    *,
    timeout_seconds: float,
) -> bool:
    """Close one Playwright page and report whether the renderer was released."""
    closed = await _run_cleanup_bounded(
        page.close,
        f"page close {url}",
        timeout_seconds=timeout_seconds,
    )
    if closed:
        log(f"[browser] closed {url}")
    return closed


async def _create_source_context(
    session: BrowserSession,
    *,
    deadline: float,
    prefix: str,
) -> tuple[Any, bool]:
    """Create an isolated source context in real Chromium sessions or reuse a test seam."""
    if session.browser is not None:
        context = await _await_bounded(
            session.browser.new_context(),
            timeout_seconds=min(
                DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
                _remaining_seconds(deadline),
            ),
            operation="isolated source context creation",
        )
        log(f"{prefix} isolated browser context created")
        return context, True
    if session.context is not None:
        # Unit tests and lightweight callers can still inject a context without a real browser.
        return session.context, False
    raise RuntimeError("browser session has neither Chromium nor a compatibility context")


async def _activate_video_elements(page: Any, prefix: str, deadline: float) -> bool:
    """Attempt programmatic playback of videos in the existing source frame tree."""
    attempted = False
    for index, frame in enumerate(page.frames):
        try:
            videos = frame.locator("video")
            count = await _await_bounded(
                videos.count(),
                timeout_seconds=min(1.0, _remaining_seconds(deadline)),
                operation=f"video count frame {index}",
            )
            if not count:
                continue
            attempted = True
            await _await_bounded(
                videos.evaluate_all(
                    """elements => Promise.allSettled(elements.map(async video => {
                        video.muted = true;
                        video.playsInline = true;
                        await video.play();
                    }))"""
                ),
                timeout_seconds=min(1.5, _remaining_seconds(deadline)),
                operation=f"video play frame {index}",
            )
            frame_url = getattr(frame, "url", "unknown")
            log(f"{prefix} media activation: play() on {count} video(s) frame={frame_url}")
        except Exception as exc:
            # Cross-origin frames are still represented by Playwright Frame objects, but a broken
            # player must never abort discovery of the rest of the source page.
            log(f"{prefix} media activation warning frame={index}: {type(exc).__name__}: {exc}")
    return attempted


async def _click_play_controls(page: Any, prefix: str, deadline: float) -> bool:
    """Click one visible play control per frame before falling back to programmatic playback."""
    clicked = False
    for frame_index, frame in enumerate(page.frames):
        for selector in _PLAY_CONTROL_SELECTORS:
            try:
                locator = frame.locator(selector).first
                count = await _await_bounded(
                    locator.count(),
                    timeout_seconds=min(0.5, _remaining_seconds(deadline)),
                    operation=f"play control count frame {frame_index}",
                )
                if not count:
                    continue
                visible = await _await_bounded(
                    locator.is_visible(),
                    timeout_seconds=min(0.5, _remaining_seconds(deadline)),
                    operation=f"play control visibility frame {frame_index}",
                )
                if not visible:
                    continue
                await _await_bounded(
                    locator.click(timeout=1_000, force=True),
                    timeout_seconds=min(1.5, _remaining_seconds(deadline)),
                    operation=f"play control click frame {frame_index}",
                )
                frame_url = getattr(frame, "url", "unknown")
                log(
                    f"{prefix} media activation: clicked {selector!r} "
                    f"frame={frame_url}"
                )
                clicked = True
                break
            except Exception:
                # Selectors are best-effort across many unrelated third-party player libraries.
                continue
    return clicked


async def _wait_for_hls_after_activation(
    stream_seen: asyncio.Event,
    *,
    deadline: float,
) -> None:
    """Wait for a user-gesture/programmatic activation to produce an HLS request."""
    wait_seconds = min(_MEDIA_ACTIVATION_GRACE_MS / 1000, _remaining_seconds(deadline))
    try:
        await asyncio.wait_for(stream_seen.wait(), timeout=wait_seconds)
    except TimeoutError:
        return


async def _visit_browser_target(
    session: BrowserSession,
    url: str,
    *,
    timeout_ms: int = DEFAULT_BROWSER_NAVIGATION_TIMEOUT_MS,
    settle_ms: int = DEFAULT_BROWSER_SETTLE_MS,
    hard_timeout_ms: int = DEFAULT_BROWSER_TARGET_HARD_TIMEOUT_MS,
    close_timeout_seconds: float = DEFAULT_BROWSER_PAGE_CLOSE_TIMEOUT_SECONDS,
    label: str | None = None,
) -> set[str]:
    """Observe one source page/frame tree inside an isolated bounded browser session."""
    if hard_timeout_ms < 1:
        raise ValueError("hard_timeout_ms must be at least 1")
    if close_timeout_seconds <= 0:
        raise ValueError("close_timeout_seconds must be greater than 0")

    prefix = f"[browser][{label}]" if label else "[browser]"
    queued_at = time.monotonic()
    log(f"{prefix} queued {url}")

    # The semaphore bounds both isolated contexts and renderer memory. Iframe URLs are never
    # reopened as separate browser targets; their requests remain visible through the source page.
    async with session.page_semaphore:
        if session.failed:
            raise RuntimeError(session.failed)
        generation = session.generation
        waited = time.monotonic() - queued_at
        log(f"{prefix} slot acquired after {waited:.1f}s")
        source_context: Any | None = None
        isolated_context = False
        page: Any | None = None
        streams: set[str] = set()
        stream_seen = asyncio.Event()
        started = time.monotonic()
        deadline = started + (hard_timeout_ms / 1000)
        try:
            source_context, isolated_context = await _create_source_context(
                session,
                deadline=deadline,
                prefix=prefix,
            )
            page = await _await_bounded(
                source_context.new_page(),
                timeout_seconds=_remaining_seconds(deadline),
                operation=f"new page {url}",
            )
            log(f"{prefix} open {url}")

            def collect(request: Any) -> None:
                """Capture only allowed HLS requests from the whole page/frame tree."""
                candidate = request.url
                if not is_allowed_hls_url(candidate):
                    return
                if candidate not in streams:
                    log(f"{prefix} HLS {candidate}")
                streams.add(candidate)
                stream_seen.set()

            page.on("request", collect)
            navigation_bound = min(
                timeout_ms / 1000,
                _remaining_seconds(deadline),
            )
            try:
                await _await_bounded(
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms),
                    timeout_seconds=navigation_bound,
                    operation=f"navigation {url}",
                )
            except _BoundedOperationTimeout:
                raise
            except Exception as exc:
                # A Playwright navigation error can still leave a usable player that emitted HLS.
                log(f"{prefix} navigation warning: {type(exc).__name__}: {exc}")
            else:
                log(f"{prefix} loaded {url}")

            if session.generation != generation:
                raise RuntimeError("browser was recycled during page visit")

            if settle_ms > 0:
                remaining = _remaining_seconds(deadline)
                if streams:
                    await asyncio.sleep(
                        min(settle_ms / 1000, _BROWSER_STREAM_GRACE_MS / 1000, remaining)
                    )
                else:
                    wait_seconds = min(settle_ms / 1000, remaining)
                    try:
                        await asyncio.wait_for(stream_seen.wait(), timeout=wait_seconds)
                    except TimeoutError:
                        pass
                    else:
                        remaining = _remaining_seconds(deadline)
                        await asyncio.sleep(
                            min(
                                settle_ms / 1000,
                                _BROWSER_STREAM_GRACE_MS / 1000,
                                remaining,
                            )
                        )

            if not streams:
                log(f"{prefix} no HLS after initial load; trying user-gesture media activation")
                # Real third-party players can explicitly require a user gesture. Try visible play
                # controls first; only use HTMLMediaElement.play() if the click path produced no HLS.
                clicked = await _click_play_controls(page, prefix, deadline)
                if clicked and not streams:
                    await _wait_for_hls_after_activation(stream_seen, deadline=deadline)
                attempted = False
                if not streams:
                    attempted = await _activate_video_elements(page, prefix, deadline)
                    if attempted and not streams:
                        await _wait_for_hls_after_activation(stream_seen, deadline=deadline)
                if not streams and not clicked and not attempted:
                    # Some players attach their video/control asynchronously after DOMContentLoaded.
                    # Even when there is nothing to activate yet, keep the source page alive for one
                    # bounded grace window so late network HLS requests are not missed.
                    log(f"{prefix} media controls not ready; waiting for delayed HLS initialization")
                    await _wait_for_hls_after_activation(stream_seen, deadline=deadline)

            # Ensure the settle/activation window itself did not consume the deadline.
            _remaining_seconds(deadline)
            elapsed = time.monotonic() - started
            log(f"{prefix} done in {elapsed:.1f}s: {len(streams)} HLS candidate(s)")
            return streams
        except _BoundedOperationTimeout as exc:
            elapsed = time.monotonic() - started
            log(
                f"{prefix} HARD TIMEOUT after {elapsed:.1f}s "
                f"(limit={hard_timeout_ms / 1000:.1f}s): {url} ({exc})"
            )
            raise
        finally:
            page_closed = True
            if page is not None:
                page_closed = await _close_page_bounded(
                    page,
                    url,
                    timeout_seconds=close_timeout_seconds,
                )

            context_closed = True
            if isolated_context and source_context is not None:
                context_closed = await _close_resource_bounded(
                    source_context,
                    f"source context {url}",
                    timeout_seconds=close_timeout_seconds,
                )
                if context_closed:
                    log(f"{prefix} isolated browser context closed")

            if not page_closed or not context_closed:
                # Never release a slot as healthy after unconfirmed source-session cleanup. Closing
                # Chromium also terminates every renderer/context belonging to the poisoned process.
                await session.recycle(
                    expected_generation=generation,
                    reason=f"source cleanup failed for {url}",
                )
                if session.failed:
                    raise RuntimeError(session.failed)


@asynccontextmanager
async def browser_session(
    enabled: bool = True,
    *,
    max_pages: int = DEFAULT_BROWSER_CONCURRENCY,
) -> AsyncIterator[BrowserSession | None]:
    """Yield one Chromium process with bounded, isolated per-source browser contexts."""
    if not enabled:
        yield None
        return
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is required for browser-based stream discovery") from exc

    log(f"[browser] launching one Chromium process; max {max_pages} active source session(s)")
    manager = async_playwright()
    playwright = await manager.start()
    session: BrowserSession | None = None
    try:
        browser = await _await_bounded(
            playwright.chromium.launch(headless=True),
            timeout_seconds=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
            operation="Chromium launch",
        )
        session = BrowserSession(
            context=None,
            page_semaphore=asyncio.Semaphore(max_pages),
            browser=browser,
            browser_type=playwright.chromium,
        )
        log("[browser] Chromium ready; source browser contexts are isolated per active page")
        yield session
    finally:
        log("[browser] shutting down shared Chromium")
        if session is not None:
            await session.shutdown()
        await _run_cleanup_bounded(
            playwright.stop,
            "Playwright stop",
            timeout_seconds=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
        )
        log("[browser] shutdown finished")


async def discover_browser(
    url: str,
    *,
    browser: BrowserSession,
    timeout_ms: int = DEFAULT_BROWSER_NAVIGATION_TIMEOUT_MS,
    settle_ms: int = DEFAULT_BROWSER_SETTLE_MS,
    hard_timeout_ms: int = DEFAULT_BROWSER_TARGET_HARD_TIMEOUT_MS,
    label: str | None = None,
) -> set[str]:
    """Discover HLS requests from one source page including requests from its frame tree."""
    if browser is None:
        raise RuntimeError("browser discovery requires an active browser session")

    prefix = f"[browser][{label}]" if label else "[browser]"
    log(f"{prefix} discovery start {url}")
    for attempt in range(2):
        generation = browser.generation
        try:
            streams = await _visit_browser_target(
                browser,
                url,
                timeout_ms=timeout_ms,
                settle_ms=settle_ms,
                hard_timeout_ms=hard_timeout_ms,
                label=label,
            )
        except Exception as exc:
            # If another source forced a successful recycle, retry this source once on new Chromium.
            if attempt == 0 and browser.generation != generation and not browser.failed:
                log(f"{prefix} retry after shared browser recycle")
                continue
            log(f"{prefix} failed {url}: {type(exc).__name__}: {exc}")
            return set()
        if streams:
            log(f"{prefix} discovery done: {len(streams)} HLS candidate(s)")
            return streams
        if attempt == 0:
            # Some third-party players occasionally initialize without starting their HLS request.
            # A second visit gets a fresh isolated browser context while keeping the retry bounded.
            log(f"{prefix} clean miss; retrying once in a fresh source context")
            continue
        log(f"{prefix} discovery done: 0 HLS candidate(s) after one clean retry")
        return set()

    return set()


async def discover_page(
    url: str,
    *,
    use_browser: bool = True,
    browser: BrowserSession | None = None,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
    client: Any | None = None,
    label: str | None = None,
    browser_observed_urls: set[str] | None = None,
) -> set[str]:
    """Run staged discovery and optionally report HLS URLs actually requested by the browser."""
    if client is None:
        async with http_session() as local_client:
            return await discover_page(
                url,
                use_browser=use_browser,
                browser=browser,
                browser_concurrency=browser_concurrency,
                client=local_client,
                label=label,
                browser_observed_urls=browser_observed_urls,
            )
    if use_browser and browser is None:
        async with browser_session(max_pages=browser_concurrency) as local_browser:
            if local_browser is None:
                raise RuntimeError("browser session unexpectedly disabled")
            return await discover_page(
                url,
                use_browser=True,
                browser=local_browser,
                browser_concurrency=browser_concurrency,
                client=client,
                label=label,
                browser_observed_urls=browser_observed_urls,
            )

    prefix = f"[discover][{label}]" if label else "[discover]"
    log(f"{prefix} page start {url}")
    streams: set[str] = set()
    http_error: BaseException | None = None
    try:
        streams.update(await discover_http(url, client=client, label=label))
    except Exception as exc:
        http_error = exc

    # Browser observation is intentionally still run after an HTTP hit: the same page can host
    # additional dynamically loaded cameras that are absent from its static HTML.
    if use_browser:
        if browser is None:
            raise RuntimeError("browser discovery requires an active browser session")
        browser_streams = await discover_browser(url, browser=browser, label=label)
        streams.update(browser_streams)
        if browser_observed_urls is not None:
            # Keep provenance separate from static HTML extraction. A request actually emitted by
            # the rendered player is strong enough evidence to survive a transient HLS probe error.
            browser_observed_urls.update(browser_streams)

    if not streams and http_error is not None:
        raise http_error
    log(f"{prefix} page done: {len(streams)} unique HLS candidate(s)")
    return streams
