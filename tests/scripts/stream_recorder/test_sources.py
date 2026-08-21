from scripts.stream_recorder.sources import Source, is_youtube_url


def test_youtube_urls_are_excluded():
    assert is_youtube_url("https://youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("https://www.ipcamlive.com/622f38aa34665")


def test_source_slug_is_deterministic():
    source = Source(id="61", place="Markt", city="Sittard", country="Netherlands", url="https://example.test/cam")
    assert source.slug == "netherlands/sittard/markt"
