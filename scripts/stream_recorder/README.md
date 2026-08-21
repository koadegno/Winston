# Public stream recorder

One-shot dataset collection tooling for Winston. It discovers public HLS webcam streams and records them into UTC hour-aligned MKV files.

YouTube and YouTube Live are intentionally unsupported.

## Requirements

- Python 3.13+
- FFmpeg available on `PATH`
- the `script` optional dependencies from `pyproject.toml`
- Playwright Chromium for JavaScript-driven players

Install the project and the one-shot script dependencies with `uv`:

```bash
uv sync --extra script
uv run python -m playwright install chromium
```

## Concurrency model

Independent I/O is concurrent:

- all source pages from `Place_Overview.xlsb` are discovered concurrently;
- direct HTTP inspection and Playwright observation for one page run concurrently;
- iframe/player URLs at the same depth are opened concurrently as isolated Playwright contexts;
- all HLS candidates for a page are resolved concurrently;
- all sources record concurrently;
- all cameras belonging to a source record concurrently;
- FFmpeg processes are awaited asynchronously, so one camera never blocks another.

One Chromium process is shared across a multi-source command. Each source/player target gets an isolated browser context.

## Discover one page

```bash
uv run python -m scripts.stream_recorder.main discover \
  --url 'https://example.com/public-webcam'
```

The command prints progress to stderr and JSON to stdout. The JSON contains every candidate `.m3u8` and every resolved logical camera.

To stop after finding `.m3u8` candidates without fetching the HLS playlists:

```bash
uv run python -m scripts.stream_recorder.main discover \
  --url 'https://example.com/public-webcam' \
  --candidates-only
```

## Discover every supported source in Place_Overview.xlsb

```bash
uv run python -m scripts.stream_recorder.main discover \
  --xlsb Place_Overview.xlsb
```

All non-YouTube rows start concurrently. Completion progress is written immediately to stderr while the final deterministic JSON array remains on stdout.

## Record

```bash
uv run python -m scripts.stream_recorder.main record \
  --xlsb Place_Overview.xlsb \
  --output datasets/recordings
```

Record only selected source IDs:

```bash
uv run python -m scripts.stream_recorder.main record \
  --xlsb Place_Overview.xlsb \
  --source-id 61 \
  --source-id 82
```

The output layout is:

```text
datasets/recordings/
└── country/city/place/
    ├── source.json
    └── cameras/
        └── camera-001/
            ├── camera.json
            └── YYYY/MM/DD/YYYY-MM-DDTHH-00-00Z.mkv
```

The first file after startup may be shorter than one hour so that subsequent files align with real UTC clock hours.
