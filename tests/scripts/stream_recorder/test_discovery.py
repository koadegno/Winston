import scripts.stream_recorder.discovery as discovery
from scripts.stream_recorder.discovery import discover_http, extract_m3u8_urls


def test_extracts_absolute_relative_and_escaped_m3u8_urls():
    html = r'''
    <video data-src="https://cdn.example/live/master.m3u8"></video>
    <script>const x = "\/streams\/cam-2.m3u8?token=abc";</script>
    <a href="relative/cam-3.m3u8">x</a>
    '''
    urls = extract_m3u8_urls(html, "https://example.test/page/")
    assert urls == {
        "https://cdn.example/live/master.m3u8",
        "https://example.test/streams/cam-2.m3u8?token=abc",
        "https://example.test/page/relative/cam-3.m3u8",
    }


def test_http_discovery_leaves_iframe_loading_to_browser(monkeypatch):
    calls: list[str] = []
    page = '<iframe src="https://third-party.example/player"></iframe>'

    def fake_fetch(url: str, timeout: float = 20.0):
        calls.append(url)
        if url != "https://example.test/page":
            raise AssertionError("HTTP discovery must not fetch third-party iframes")
        return page, url

    monkeypatch.setattr("scripts.stream_recorder.discovery.fetch_text", fake_fetch)

    assert discover_http("https://example.test/page") == set()
    assert calls == ["https://example.test/page"]


def test_browser_discovery_visits_iframe_urls_as_independent_targets():
    root_url = "https://example.test/webcam"
    player_url = "https://player.example/embed/123"
    stream_url = "https://cdn.example/live/camera.m3u8"
    visited: list[str] = []

    def visit(url: str) -> tuple[set[str], set[str]]:
        visited.append(url)
        if url == root_url:
            return set(), {player_url}
        if url == player_url:
            return {stream_url}, set()
        raise AssertionError(f"unexpected target: {url}")

    crawl = getattr(discovery, "crawl_browser_targets", None)
    assert crawl is not None, "isolated browser target crawler is not implemented"
    assert crawl(root_url, visit, max_iframe_depth=1) == {stream_url}
    assert visited == [root_url, player_url]
