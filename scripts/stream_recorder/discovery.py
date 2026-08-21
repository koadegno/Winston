from __future__ import annotations

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


def extract_rendered_frame_m3u8_urls(frames) -> set[str]:
    streams: set[str] = set()
    for frame in frames:
        try:
            streams.update(extract_m3u8_urls(frame.content(), frame.url))
        except Exception:
            continue
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


def discover_browser(url: str, timeout_ms: int = 20_000, settle_ms: int = 5_000) -> set[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is required for browser-based stream discovery") from exc

    streams: set[str] = set()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()

            def collect(request) -> None:
                if ".m3u8" in request.url.lower():
                    streams.add(request.url)

            page.on("request", collect)
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(settle_ms)
            streams.update(extract_rendered_frame_m3u8_urls(page.frames))
        finally:
            browser.close()
    return streams


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
