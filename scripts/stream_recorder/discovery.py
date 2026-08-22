"""Discover public HLS streams from HTML pages and browser network traffic."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import re
import ssl
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BROWSER_CONCURRENCY = 2

_M3U8_RE = re.compile(
    r"(?P<url>(?:https?:)?(?:\\?/[^\s'\"<>]*)?[^\s'\"<>]*?\.m3u8(?:\?[^\s'\"<>]*)?)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class BrowserSession:
    """Share one browser context and bound the number of simultaneously active tabs."""

    context: Any
    page_semaphore: asyncio.Semaphore


def _normalize_escaped_url(value: str) -> str:
    """Normalize slash-escaped and HTML-escaped URLs found in page source."""
    return value.replace("\\/", "/").replace("&amp;", "&")


def extract_m3u8_urls(text: str, base_url: str) -> set[str]:
    """Extract absolute HLS playlist URLs from arbitrary HTML or JavaScript text."""
    urls: set[str] = set()
    for match in _M3U8_RE.finditer(text):
        raw = _normalize_escaped_url(match.group("url")).strip()
        if raw.startswith("//"):
            raw = "https:" + raw
        urls.add(urljoin(base_url, raw))
    return urls


def _fetch_text_sync(url: str, timeout: float) -> tuple[str, str]:
    """Fetch text with urllib for execution in a worker thread."""
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/127 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/vnd.apple.mpegurl,*/*;q=0.8",
        },
    )
    context = ssl.create_default_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset, errors="replace")
        final_url = response.geturl()
    return body, final_url


async def fetch_text(url: str, timeout: float = 20.0) -> tuple[str, str]:
    """Fetch text without blocking the asyncio event loop."""
    # urllib is intentionally kept to avoid another runtime dependency; move it to a thread.
    return await asyncio.to_thread(_fetch_text_sync, url, timeout)


async def discover_http(url: str) -> set[str]:
    """Discover HLS URLs visible directly in the source page response."""
    body, final_url = await fetch_text(url)
    return extract_m3u8_urls(body, final_url)


async def crawl_browser_targets(
    root_url: str,
    visit: Callable[[str], Awaitable[tuple[set[str], set[str]]]],
    *,
    max_iframe_depth: int = 2,
) -> set[str]:
    """Visit browser targets breadth-first while scheduling siblings concurrently."""
    streams: set[str] = set()
    seen: set[str] = set()
    current_level = {root_url}

    for depth in range(max_iframe_depth + 1):
        targets = sorted(url for url in current_level if url not in seen)
        if not targets:
            break
        seen.update(targets)

        # Siblings are scheduled together; BrowserSession applies the global tab limit.
        results = await asyncio.gather(*(visit(target) for target in targets), return_exceptions=True)
        next_level: set[str] = set()
        for result in results:
            if isinstance(result, BaseException):
                continue
            target_streams, iframe_urls = result
            streams.update(target_streams)
            if depth < max_iframe_depth:
                next_level.update(url for url in iframe_urls if url not in seen)
        current_level = next_level

    return streams


async def _iframe_urls(page: Any) -> set[str]:
    """Return absolute iframe source URLs exposed by a rendered Playwright page."""
    urls = await page.locator("iframe").evaluate_all(
        "elements => elements.map(element => element.src).filter(Boolean)"
    )
    return {
        url
        for url in urls
        if isinstance(url, str) and url.startswith(("http://", "https://"))
    }


async def _visit_browser_target(
    session: BrowserSession,
    url: str,
    *,
    timeout_ms: int,
    settle_ms: int,
) -> tuple[set[str], set[str]]:
    """Observe one target in a shared context while respecting the global tab limit."""
    streams: set[str] = set()

    # A renderer can use hundreds of MB. Keep only a small bounded number alive at once.
    async with session.page_semaphore:
        page = await session.context.new_page()
        try:

            def collect(request: Any) -> None:
                """Capture HLS requests emitted while the target page is running."""
                if ".m3u8" in request.url.lower():
                    streams.add(request.url)

            page.on("request", collect)
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_timeout(settle_ms)
            return streams, await _iframe_urls(page)
        finally:
            await page.close()


@asynccontextmanager
async def browser_session(
    enabled: bool = True,
    *,
    max_pages: int = DEFAULT_BROWSER_CONCURRENCY,
) -> AsyncIterator[BrowserSession | None]:
    """Yield one Chromium/context pair with bounded concurrent tabs."""
    if not enabled:
        yield None
        return
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is required for browser-based stream discovery") from exc

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        try:
            yield BrowserSession(
                context=context,
                page_semaphore=asyncio.Semaphore(max_pages),
            )
        finally:
            await context.close()
            await browser.close()


async def discover_browser(
    url: str,
    *,
    browser: BrowserSession,
    timeout_ms: int = 20_000,
    settle_ms: int = 5_000,
    max_iframe_depth: int = 2,
) -> set[str]:
    """Discover HLS requests from a page and its isolated iframe/player targets."""
    if browser is None:
        raise RuntimeError("browser discovery requires an active browser session")

    async def visit(target_url: str) -> tuple[set[str], set[str]]:
        """Visit one target while converting target-specific failures to an empty result."""
        try:
            return await _visit_browser_target(
                browser,
                target_url,
                timeout_ms=timeout_ms,
                settle_ms=settle_ms,
            )
        except Exception:
            # A broken third-party player must not prevent other cameras from being discovered.
            return set(), set()

    return await crawl_browser_targets(url, visit, max_iframe_depth=max_iframe_depth)


async def discover_page(
    url: str,
    *,
    use_browser_fallback: bool = True,
    browser: BrowserSession | None = None,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
) -> set[str]:
    """Run direct HTTP and browser discovery concurrently for one source page."""
    if use_browser_fallback and browser is None:
        async with browser_session(max_pages=browser_concurrency) as local_browser:
            return await discover_page(
                url,
                use_browser_fallback=True,
                browser=local_browser,
                browser_concurrency=browser_concurrency,
            )

    coroutines: list[Awaitable[set[str]]] = [discover_http(url)]
    if use_browser_fallback:
        if browser is None:
            raise RuntimeError("browser discovery requires an active browser session")
        coroutines.append(discover_browser(url, browser=browser))

    # HTTP source inspection and browser observation are independent I/O paths.
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    streams: set[str] = set()
    errors: list[BaseException] = []
    for result in results:
        if isinstance(result, BaseException):
            errors.append(result)
        else:
            streams.update(result)

    if not streams and len(errors) == len(results):
        raise errors[0]
    return streams
