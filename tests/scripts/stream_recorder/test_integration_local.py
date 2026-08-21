from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from scripts.stream_recorder.orchestrator import discover_source_cameras
from scripts.stream_recorder.sources import Source


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/page":
            body = b'<script>const stream="/master.m3u8";</script>'
            content_type = "text/html"
        elif self.path == "/master.m3u8":
            body = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1000,RESOLUTION=640x360\n/low.m3u8\n#EXT-X-STREAM-INF:BANDWIDTH=5000,RESOLUTION=1920x1080\n/high.m3u8\n"
            content_type = "application/vnd.apple.mpegurl"
        elif self.path in {"/low.m3u8", "/high.m3u8"}:
            body = b"#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4,\nsegment.ts\n"
            content_type = "application/vnd.apple.mpegurl"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


@contextmanager
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        thread.join()


def test_discovers_master_and_selects_best_variant():
    with server() as port:
        source = Source("1", "Square", "City", "Country", f"http://127.0.0.1:{port}/page")
        cameras = discover_source_cameras(source, use_browser_fallback=False)
    assert len(cameras) == 1
    assert cameras[0].stream.resolution == (1920, 1080)
    assert cameras[0].stream.url.endswith("/high.m3u8")
