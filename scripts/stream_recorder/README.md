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

The recorder schedules independent I/O concurrently without allowing browser renderers to exhaust system memory:

- all source tasks from `Place_Overview.xlsb` are scheduled concurrently;
- direct HTTP inspection is concurrent and does not consume browser tabs;
- all HLS candidates for a page are resolved concurrently;
- all sources record concurrently;
- all cameras belonging to a source record concurrently;
- FFmpeg processes are awaited asynchronously, so one camera never blocks another;
- Playwright uses exactly one Chromium process and one shared browser context;
- source pages and iframe/player targets use tabs in that shared context;
- a global semaphore limits active Playwright tabs to **2 by default** across the entire command.

The browser limit is configurable with `--browser-concurrency N`. Increasing it can speed up discovery but also increases RAM use because each active Chromium renderer can consume hundreds of MB.

Playwright listens to the complete page network, including requests emitted by normally loaded frames. If a page already exposes one or more HLS streams, its iframe URLs are **not reopened as separate browser targets**. Separate iframe crawling is only used when the parent page exposed no HLS at all. This avoids duplicating expensive browser work while preserving the iframe fallback needed by third-party players.

## Progress logs

Machine-readable discovery JSON is written to stdout. Detailed progress is written immediately to stderr, so a long batch never looks frozen.

The logs show:

- every source task starting and completing;
- direct HTTP request start, completion time, candidate count, and errors;
- Chromium startup and shutdown;
- every browser target entering the queue;
- every browser tab opening, loading, finding HLS requests, and closing;
- iframe fallback depth and whether iframe targets are skipped;
- HLS master/variant resolution and selected quality;
- source/camera metadata creation;
- each FFmpeg process starting, its PID, output file, duration, and exit code;
- retry and HLS rediscovery after FFmpeg failures.

Typical browser progress looks like:

```text
[00:50:45] [browser] queued https://example.com/webcam
[00:50:45] [browser] open https://example.com/webcam
[00:50:48] [browser] HLS https://cdn.example.com/live/stream.m3u8
[00:50:48] [browser] done https://example.com/webcam: 1 HLS, 2 iframe(s) in 3.2s
[00:50:48] [browser] closed https://example.com/webcam
[00:50:48] [browser] skip 2 iframe target(s) under https://example.com/webcam: HLS already observed on parent page
```

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

All non-YouTube source tasks are scheduled immediately, but only two Playwright tabs are active at once by default. HTTP work continues concurrently while browser work waits for a tab slot. Completion progress is written immediately to stderr while the final deterministic JSON array remains on stdout.

Use a different browser limit when needed:

```bash
uv run python -m scripts.stream_recorder.main discover \
  --xlsb Place_Overview.xlsb \
  --browser-concurrency 1
```

## Record

```bash
uv run python -m scripts.stream_recorder.main record \
  --xlsb Place_Overview.xlsb \
  --output datasets/recordings
```

The same browser limit applies during initial discovery and later stream rediscovery. FFmpeg recording processes themselves remain independent of this Playwright limit.

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
