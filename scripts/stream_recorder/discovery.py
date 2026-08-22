"""Discover public HLS streams from HTML pages and browser network traffic."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from .log import log


DEFAULT_BROWSER_CONCURRENCY = 4
DEFAULT_BROWSER_NAVIGATION_TIMEOUT_MS = 10_000
DEFAULT_BROWSER_SETTLE_MS = 4_000
DEFAULT_BROWSER_TARGET_HARD_TIMEOUT_MS = 15_000
DEFAULT_BROWSER_PAGE_CLOSE_TIMEOUT_SECONDS = 2.0
DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
DEFAULT_HTTP_CONNECTIONS = 32
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
_BROWSER_STREAM_GRACE_MS = 750

_M3U8_RE = re.compile(
    r"(?P<url>(?:https?:)?(?:\\?/[^\s'\"<>]*)?[^\s'\"<>]*?\.m3u8(?:\?[^\s'\"<>]*)?)",
    re.IGNORECASE,
)
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


@dataclass(slots=True)
class BrowserSession:
    """Share one Chromium context while bounding active source pages."""

    context: Any
    page_semaphore: asyncio.Semaphore
    browser: Any | None = None
    browser_type: Any | None = None
    generation: int = 0
    failed: str | None = None
    restart_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def recycle(self, *, expected_generation: int, reason: str) -> None:
        """Recycle a poisoned browser/context before any new renderer is created."""
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

            context_closed = await _close_resource_bounded(
                old_context,
                "browser context",
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
                new_browser = await asyncio.wait_for(
                    self.browser_type.launch(headless=True),
                    timeout=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
                )
                new_context = await asyncio.wait_for(
                    new_browser.new_context(),
                    timeout=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                self.failed = f"browser restart failed: {type(exc).__name__}: {exc}"
                log(f"[browser] FATAL {self.failed}")
                return

            self.browser = new_browser
            self.context = new_context
            self.failed = None
            log(f"[browser] recycle complete; generation={self.generation}")

    async def shutdown(self) -> None:
        """Bound shared browser shutdown so command completion cannot hang forever."""
        context = self.context
        browser = self.browser
        self.context = None
        self.browser = None

        context_closed = await _close_resource_bounded(
            context,
            "browser context",
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


def is_allowed_hls_url(url: str) -> bool:
    """Accept HTTP(S) HLS URLs while excluding unrelated Google/Facebook/YouTube services."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    if ".m3u8" not in url.lower():
        return False
    return not any(_host_matches_suffix(host, suffix) for suffix in _BLOCKED_STREAM_HOST_SUFFIXES)


def extract_m3u8_urls(text: str, base_url: str) -> set[str]:
    """Extract allowed absolute HLS playlist URLs from arbitrary HTML or JavaScript text."""
    urls: set[str] = set()
    for match in _M3U8_RE.finditer(text):
        raw = _normalize_escaped_url(match.group("url")).strip()
        if raw.startswith("//"):
            raw = "https:" + raw
        candidate = urljoin(base_url, raw)
        if is_allowed_hls_url(candidate):
            urls.add(candidate)
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
) -> tuple[str, str]:
    """Fetch text through a supplied/shared async HTTP client."""
    if client is None:
        # Standalone callers still get true async I/O; batch callers pass one shared client.
        async with http_session() as local_client:
            return await fetch_text(url, client=local_client)

    response = await client.get(url)
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


async def _run_cleanup_bounded(
    action: Callable[[], Awaitable[Any]],
    label: str,
    *,
    timeout_seconds: float,
) -> bool:
    """Run one asynchronous cleanup action with a strict deadline."""
    try:
        await asyncio.wait_for(action(), timeout=timeout_seconds)
    except TimeoutError:
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
    """Observe one source page/frame tree with one bounded Playwright tab."""
    if hard_timeout_ms < 1:
        raise ValueError("hard_timeout_ms must be at least 1")
    if close_timeout_seconds <= 0:
        raise ValueError("close_timeout_seconds must be greater than 0")

    prefix = f"[browser][{label}]" if label else "[browser]"
    queued_at = time.monotonic()
    log(f"{prefix} queued {url}")

    # The semaphore bounds renderer memory; no iframe URL is ever reopened as another page.
    async with session.page_semaphore:
        if session.failed:
            raise RuntimeError(session.failed)
        generation = session.generation
        waited = time.monotonic() - queued_at
        log(f"{prefix} slot acquired after {waited:.1f}s")
        page: Any | None = None
        streams: set[str] = set()
        stream_seen = asyncio.Event()
        started = time.monotonic()
        try:
            async with asyncio.timeout(hard_timeout_ms / 1000):
                if session.context is None:
                    raise RuntimeError("browser context is unavailable")
                page = await session.context.new_page()
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
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                except Exception as exc:
                    # A navigation timeout/error can still leave a usable player that emitted HLS.
                    log(f"{prefix} navigation warning: {type(exc).__name__}: {exc}")
                else:
                    log(f"{prefix} loaded {url}")

                if session.generation != generation:
                    raise RuntimeError("browser was recycled during page visit")

                if settle_ms > 0:
                    if streams:
                        await page.wait_for_timeout(min(settle_ms, _BROWSER_STREAM_GRACE_MS))
                    else:
                        try:
                            await asyncio.wait_for(stream_seen.wait(), timeout=settle_ms / 1000)
                        except TimeoutError:
                            pass
                        else:
                            await page.wait_for_timeout(min(settle_ms, _BROWSER_STREAM_GRACE_MS))

                elapsed = time.monotonic() - started
                log(f"{prefix} done in {elapsed:.1f}s: {len(streams)} HLS candidate(s)")
                return streams
        except TimeoutError:
            elapsed = time.monotonic() - started
            log(
                f"{prefix} HARD TIMEOUT after {elapsed:.1f}s "
                f"(limit={hard_timeout_ms / 1000:.1f}s): {url}"
            )
            raise
        finally:
            if page is not None:
                closed = await _close_page_bounded(
                    page,
                    url,
                    timeout_seconds=close_timeout_seconds,
                )
                if not closed:
                    # Never release a slot as healthy after an unconfirmed renderer cleanup.
                    await session.recycle(
                        expected_generation=generation,
                        reason=f"page cleanup failed for {url}",
                    )
                    if session.failed:
                        raise RuntimeError(session.failed)


@asynccontextmanager
async def browser_session(
    enabled: bool = True,
    *,
    max_pages: int = DEFAULT_BROWSER_CONCURRENCY,
) -> AsyncIterator[BrowserSession | None]:
    """Yield one Chromium process/context with bounded concurrent source tabs."""
    if not enabled:
        yield None
        return
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is required for browser-based stream discovery") from exc

    log(f"[browser] launching one Chromium process; max {max_pages} active source tab(s)")
    manager = async_playwright()
    playwright = await manager.start()
    session: BrowserSession | None = None
    try:
        browser = await asyncio.wait_for(
            playwright.chromium.launch(headless=True),
            timeout=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
        )
        context = await asyncio.wait_for(
            browser.new_context(),
            timeout=DEFAULT_BROWSER_SHUTDOWN_TIMEOUT_SECONDS,
        )
        session = BrowserSession(
            context=context,
            page_semaphore=asyncio.Semaphore(max_pages),
            browser=browser,
            browser_type=playwright.chromium,
        )
        log("[browser] Chromium ready; one shared browser context created")
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
            # If another tab forced a successful recycle, retry this source once on the new context.
            if attempt == 0 and browser.generation != generation and not browser.failed:
                log(f"{prefix} retry after shared browser recycle")
                continue
            log(f"{prefix} failed {url}: {type(exc).__name__}: {exc}")
            return set()
        log(f"{prefix} discovery done: {len(streams)} HLS candidate(s)")
        return streams

    return set()


async def discover_page(
    url: str,
    *,
    use_browser: bool = True,
    browser: BrowserSession | None = None,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
    client: Any | None = None,
    label: str | None = None,
) -> set[str]:
    """Run staged HTTP then one-page browser discovery for a single source."""
    if client is None:
        async with http_session() as local_client:
            return await discover_page(
                url,
                use_browser=use_browser,
                browser=browser,
                browser_concurrency=browser_concurrency,
                client=local_client,
                label=label,
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
        streams.update(await discover_browser(url, browser=browser, label=label))

    if not streams and http_error is not None:
        raise http_error
    log(f"{prefix} page done: {len(streams)} unique HLS candidate(s)")
    return streams
