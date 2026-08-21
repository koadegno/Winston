import json

from scripts.stream_recorder.main import main


def test_discover_url_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "scripts.stream_recorder.main.discover_url",
        lambda *_args, **_kwargs: {
            "url": "https://example.test/page",
            "candidates": ["https://example.test/master.m3u8"],
            "cameras": [{"camera_id": "camera-001", "stream_url": "https://example.test/high.m3u8"}],
        },
    )
    assert main(["discover", "--url", "https://example.test/page"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cameras"][0]["camera_id"] == "camera-001"
