from scripts.stream_recorder.discovery import (
    discover_http,
    extract_m3u8_urls,
    extract_rendered_frame_m3u8_urls,
)


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


class _FakeLocator:
    def __init__(self, content: str):
        self._content = content

    def inner_html(self, *, timeout: float) -> str:
        assert timeout <= 2_000
        return self._content


class _FakeFrame:
    def __init__(self, url: str, content: str):
        self.url = url
        self._content = content

    def content(self) -> str:
        raise AssertionError("unbounded frame.content() must not be used")

    def locator(self, selector: str) -> _FakeLocator:
        assert selector == "html"
        return _FakeLocator(self._content)


def test_extracts_m3u8_from_rendered_child_frames_with_bounded_dom_reads():
    frames = [
        _FakeFrame("https://example.test/", "<body></body>"),
        _FakeFrame(
            "https://player.example/embed/123",
            '<script>window.stream="https://cdn.example/live/camera.m3u8?token=abc"</script>',
        ),
    ]

    assert extract_rendered_frame_m3u8_urls(frames) == {
        "https://cdn.example/live/camera.m3u8?token=abc"
    }
