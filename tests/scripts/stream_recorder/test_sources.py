from pathlib import Path

from scripts.stream_recorder.sources import Source, is_youtube_url, load_sources


def test_youtube_urls_are_excluded():
    """YouTube sources remain out of scope regardless of source-file format."""
    assert is_youtube_url("https://youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("https://www.ipcamlive.com/622f38aa34665")


def test_source_slug_is_deterministic():
    """Storage slugs stay deterministic for stable source metadata."""
    source = Source(
        id="61",
        place="Markt",
        city="Sittard",
        country="Netherlands",
        url="https://example.test/cam",
    )
    assert source.slug == "netherlands/sittard/markt"


def test_load_sources_reads_csv_and_skips_disabled_and_youtube(tmp_path):
    """The maintainable CSV loads enabled non-YouTube sources only."""
    path = tmp_path / "Place_Overview.csv"
    path.write_text(
        "id,place,city,country,url,description,enabled,disabled_reason\n"
        "90-1,Namesti T. G. M.,Pribram,Czech republic,https://example.test/cam,,true,\n"
        "61,Markt,Sittard,Netherlands,https://webcamsittard.nl/index.php/nl/camera1,,false,youtube-backed\n"
        "8,Legacy YouTube,Biberach,Germany,https://www.youtube.com/watch?v=abc,,true,\n",
        encoding="utf-8",
    )

    sources = load_sources(path)

    assert [source.id for source in sources] == ["90-1"]
    assert sources[0].url == "https://example.test/cam"


def test_repository_csv_contains_new_pribram_and_skyline_sources():
    """The maintained repository source list contains all newly validated public camera pages."""
    sources = load_sources(Path("Place_Overview.csv"))
    by_id = {source.id: source for source in sources}

    assert len(sources) == 34
    assert "61" not in by_id
    assert {
        by_id[source_id].url
        for source_id in ("90-1", "90-2", "90-3", "90-4", "90-5", "90-6")
    } == {
        "https://pribram.eu/pribram-zive-namesti-t-g-m.html",
        "https://pribram.eu/pribram-zive-namesti-17-listopadu.html",
        "https://pribram.eu/pribram-zive-dvorakovo-nabrezi.html",
        "https://pribram.eu/pribram-zive-sk-spartak.html",
        "https://pribram.eu/pribram-zive-sk-spartak-3.html",
        "https://pribram.eu/pribram-zive-namesti-j-a-alise.html",
    }
    assert by_id["131"].url.endswith("/santander/playa-del-sardinero.html")
    assert by_id["132"].url.endswith("/birmingham/sutton-coldfield.html")
    assert by_id["133"].url.endswith("/genova/boccadasse-genova.html")
