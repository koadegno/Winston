from threading import Barrier

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


def test_discover_http_fetches_iframes_concurrently(monkeypatch):
    barrier = Barrier(2)
    page = '<iframe src="/cam-a"></iframe><iframe src="/cam-b"></iframe>'

    def fake_fetch(url: str, timeout: float = 20.0):
        if url == "https://example.test/page":
            return page, url
        barrier.wait(timeout=0.5)
        camera_name = url.rsplit("/", 1)[-1]
        return f'https://cdn.example/{camera_name}.m3u8', url

    monkeypatch.setattr("scripts.stream_recorder.discovery.fetch_text", fake_fetch)

    assert discover_http("https://example.test/page") == {
        "https://cdn.example/cam-a.m3u8",
        "https://cdn.example/cam-b.m3u8",
    }
