import asyncio
import json
from pathlib import Path

import pytest

from scripts.stream_recorder import main as main_module
from scripts.stream_recorder.main import main
from scripts.stream_recorder.sources import Source


@pytest.mark.asyncio
async def test_discover_url_prints_json(monkeypatch, capsys):
    """Single-URL discovery keeps machine-readable JSON on stdout."""
    async def fake_discover_url(*_args, **_kwargs):
        """Return one deterministic resolved camera."""
        return {
            "url": "https://example.test/page",
            "candidates": ["https://example.test/master.m3u8"],
            "cameras": [{"camera_id": "camera-001", "stream_url": "https://example.test/high.m3u8"}],
        }

    monkeypatch.setattr(main_module, "discover_url", fake_discover_url)
    assert await main(["discover", "--url", "https://example.test/page"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cameras"][0]["camera_id"] == "camera-001"


@pytest.mark.asyncio
async def test_candidates_only_skips_hls_resolution(monkeypatch, capsys):
    """Candidates-only discovery never resolves HLS master playlists."""
    async def fake_discover_page(*_args, **_kwargs):
        """Return one discovered HLS candidate."""
        return {"https://example.test/master.m3u8"}

    async def fail_resolve(*_args, **_kwargs):
        """Fail if candidate-only mode incorrectly attempts HLS resolution."""
        raise AssertionError("must not resolve")

    monkeypatch.setattr(main_module, "discover_page", fake_discover_page)
    monkeypatch.setattr(main_module, "resolve_cameras", fail_resolve)
    assert await main([
        "discover",
        "--url",
        "https://example.test/page",
        "--candidates-only",
        "--no-browser",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidates"] == ["https://example.test/master.m3u8"]
    assert payload["cameras"] == []


@pytest.mark.asyncio
async def test_xlsb_sources_discover_concurrently_and_report_progress(monkeypatch, tmp_path: Path, capsys):
    """Every XLSB source starts concurrently and emits progress on stderr."""
    sources = [
        Source("1", "A", "City", "Country", "https://example.test/a"),
        Source("2", "B", "City", "Country", "https://example.test/b"),
    ]
    both_started = asyncio.Event()
    started: set[str] = set()
    monkeypatch.setattr(main_module, "load_sources", lambda _path: sources)

    async def fake_discover_url(url: str, **_kwargs):
        """Block each source until both source tasks have started."""
        started.add(url)
        if len(started) == len(sources):
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)
        return {"url": url, "candidates": [f"{url}/stream.m3u8"], "cameras": []}

    monkeypatch.setattr(main_module, "discover_url", fake_discover_url)
    results = await main_module._discover_xlsb(
        tmp_path / "places.xlsb",
        use_browser_fallback=False,
        resolve=False,
    )
    captured = capsys.readouterr()
    assert [result["source"]["id"] for result in results] == ["1", "2"]
    assert "starting 2 sources in parallel" in captured.err


@pytest.mark.asyncio
async def test_xlsb_candidates_only_passes_resolve_false(monkeypatch, tmp_path: Path, capsys):
    """The XLSB CLI path honors candidates-only mode just like single-URL discovery."""
    observed: dict[str, bool] = {}

    async def fake_discover_xlsb(_path, *, use_browser_fallback: bool, resolve: bool):
        """Capture the resolve flag passed by the CLI."""
        observed["resolve"] = resolve
        observed["browser"] = use_browser_fallback
        return []

    monkeypatch.setattr(main_module, "_discover_xlsb", fake_discover_xlsb)
    assert await main([
        "discover",
        "--xlsb",
        str(tmp_path / "places.xlsb"),
        "--candidates-only",
        "--no-browser",
    ]) == 0
    json.loads(capsys.readouterr().out)
    assert observed == {"resolve": False, "browser": False}


@pytest.mark.asyncio
async def test_record_sources_start_concurrently(monkeypatch, tmp_path: Path):
    """Every selected source recorder starts concurrently."""
    sources = [
        Source("1", "A", "City", "Country", "https://example.test/a"),
        Source("2", "B", "City", "Country", "https://example.test/b"),
    ]
    both_started = asyncio.Event()
    started: set[str] = set()

    async def fake_record(source, _output, **_kwargs):
        """Block each source recorder until both have started."""
        started.add(source.id)
        if len(started) == len(sources):
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)

    monkeypatch.setattr(main_module, "record_source_forever", fake_record)
    await main_module._record_sources(sources, tmp_path, use_browser=False)
    assert started == {"1", "2"}
