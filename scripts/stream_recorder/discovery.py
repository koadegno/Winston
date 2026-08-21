from __future__ import annotations

from collections import deque
from collections.abc import Callable
import re
import ssl
from urllib.parse import urljoin
from urllib.request import Request, urlopen


_M3U8_RE = re.compile(
    r"(?P<url>(?:https?:)?(?:\\?/[^\s'\"<>]*)?[^\s'\"<>]*?\.m3u8(?:\?[^\s'\"<>]*)?)",
    re.IGNORECASE,
)


def _normalize_escaped_url(value: str) -> str:
    return value.replace("\\/", "/").replace("&amp;", "&")


def extract_m3u8_urls(text: str, base_url: str) -> set[str]:
    urls: set[str] = set()
    for match in _M3U8_RE.finditer(text):
        raw = _normalize_escaped_url(match.group("url")).strip()
        if raw.startswith("//"):
            raw = "https:" + raw
        urls.add(urljoin(base_url, raw))
    return urls


def crawl_browser_targets(
    root_url: str,
    visit: Callable[[str], tuple[set[str], set[str]]],
    *,
    max_iframe_depth: int = 2,
) -> set[str]:
    streams: set[str] = set()
    seen: set[str] = set()
    pending = deque([(root_url, 0)])

    while pending:
        target_url, depth = pending.popleft()
        if target_url in seen:
            continue
        seen.add(target_url)

        target_streams, iframe_urls = visit(target_url)
        streams.update(target_streams)

        if depth >= max_iframe_depth:
            continue
        for iframe_url in sorted(iframe_urls):
            if iframe_url not in seen:
                pending.append((iframe_url, depth + 1))

    return streams


def fetch_text(url: str, timeout: float = 20.0) -> tuple[str, str]:
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


def discover_http(url: str) -> set[str]:
    body, final_url = fetch_text(url)
    return extract_m3u8_urls(body, final_url)


def _iframe_urls(page) -> set[str]:
    urls = page.locator("iframe").evaluate_all(
        "elements => elements.map(element => element.src).filter(Boolean)"
    )
    return {url for url in urls if isinstance(url, str) and url.startswith(("http://", "https://"))}


def _visit_browser_target(browser, url: str, *, timeout_ms: int, settle_ms: int) -> tuple[set[str], set[str]]:
    streams: set[str] = set()
    context = browser.new_context()
    try:
        page = context.new_page()

        def collect(request) -> None:
            if ".m3u8" in request.url.lower():
                streams.add(request.url)

        page.on("request", collect)
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(settle_ms)
        return streams, _iframe_urls(page)
    finally:
        context.close()


def discover_browser(
    url: str,
    timeout_ms: int = 20_000,
    settle_ms: int = 5_000,
    max_iframe_depth: int = 2,
) -> set[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is required for browser-based stream discovery") from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            def visit(target_url: str) -> tuple[set[str], set[str]]:
                try:
                    return _visit_browser_target(
                        browser,
                        target_url,
                        timeout_ms=timeout_ms,
                        settle_ms=settle_ms,
                    )
                except Exception:
                    return set(), set()

            return crawl_browser_targets(url, visit, max_iframe_depth=max_iframe_depth)
        finally:
            browser.close()


def discover_page(url: str, use_browser_fallback: bool = True) -> set[str]:
    streams: set[str] = set()
    try:
        streams.update(discover_http(url))
    except Exception:
        pass

    if use_browser_fallback:
        try:
            streams.update(discover_browser(url))
        except Exception:
            if not streams:
                raise
    return streams
