"""Regression tests for HLS URL classification during stream discovery."""

from scripts.stream_recorder.discovery import extract_m3u8_urls, is_allowed_hls_url


def test_rejects_telemetry_url_when_m3u8_only_appears_inside_query_value() -> None:
    """Analytics requests embedding a stream URL in their query are not HLS playlists."""
    telemetry_url = (
        "https://prd.jwpltx.com/v1/jwplayer6/ping.gif?"
        "mu=https%3A%2F%2Fwww.crtvg.es%3A1555%2Fhls%2Fcelanova%2Findex.m3u8"
    )

    assert not is_allowed_hls_url(telemetry_url)
    assert extract_m3u8_urls(
        f'<script>fetch("{telemetry_url}")</script>',
        "https://www.g24.gal/-/celanova",
    ) == set()


def test_accepts_real_hls_path_with_query_token() -> None:
    """A real playlist remains valid when authentication data follows in the query string."""
    stream_url = "https://cdn.example.test/live/master.m3u8?token=abc123"

    assert is_allowed_hls_url(stream_url)
    assert extract_m3u8_urls(
        f'<video src="{stream_url}"></video>',
        "https://example.test/page",
    ) == {stream_url}
