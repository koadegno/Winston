"""Performance regressions for static HLS URL extraction."""

from __future__ import annotations

import time

from scripts.stream_recorder.discovery import extract_m3u8_urls


def test_static_hls_extraction_stays_fast_on_large_pages_without_matches() -> None:
    """Scanning a large page without HLS must be linear enough to keep the event loop responsive."""
    html = "a" * 20_000
    started = time.monotonic()

    assert extract_m3u8_urls(html, "https://example.test/page") == set()

    assert time.monotonic() - started < 0.25
