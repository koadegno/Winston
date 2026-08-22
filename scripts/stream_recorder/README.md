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

## Source list

`Place_Overview.csv` is the maintained source list used for discovery and recording. It contains enabled and disabled rows so known exclusions remain documented instead of disappearing from history.

The columns are:

```text
id,place,city,country,url,description,enabled,disabled_reason
```

Disabled rows and YouTube URLs are skipped before discovery. The legacy `Place_Overview.xlsb` loader remains supported for compatibility, but new source maintenance belongs in the CSV.

## Concurrency model

The recorder schedules independent I/O concurrently without allowing browser renderers to exhaust system memory:

- all enabled source tasks are scheduled concurrently;
- direct HTTP inspection is concurrent and does not consume browser tabs;
- all HLS candidates for a page are resolved concurrently;
- all sources record concurrently;
- all cameras belonging to a source record concurrently;
- FFmpeg processes are awaited asynchronously, so one camera never blocks another;
- Playwright uses exactly one Chromium process and one shared browser context;
- each source is opened once as a tab in that shared context;
- a global semaphore limits active Playwright tabs to **4 by default** across the entire command.

The browser limit is configurable with `--browser-concurrency N`. Lower it on RAM-constrained machines; increasing it can speed up discovery but each active Chromium renderer can consume hundreds of MB.

Playwright listens to network requests from the complete loaded frame tree. Embedded iframe URLs are **not reopened recursively**. If a third-party player does not start automatically, discovery tries a visible Play interaction inside the already loaded frame tree and only then falls back to programmatic `video.play()`.

## Progress logs

Machine-readable discovery JSON is written to stdout. Detailed progress is written immediately to stderr, so a long batch never looks frozen.

The logs show:

- each HTTP source request starting, completing, timing out, or failing;
- Chromium startup and shutdown;
- each browser source entering the queue and acquiring a tab slot;
- browser page open/load/close events;
- media activation attempts when a player needs interaction;
- every accepted HLS request as it is observed;
- HLS master/variant resolution and selected quality;
- TLS-only HLS fallback when an already discovered stream has a broken certificate;
- source/camera metadata creation;
- each FFmpeg process starting, output path, duration, and exit code;
- retry and HLS rediscovery after FFmpeg failures.

Typical browser progress looks like:

```text
[00:50:45] [browser][source 112] queued https://example.com/webcam
[00:50:45] [browser][source 112] slot acquired after 0.0s
[00:50:45] [browser][source 112] open https://example.com/webcam
[00:50:48] [browser][source 112] no HLS after initial load; trying user-gesture media activation
[00:50:49] [browser][source 112] media activation: clicked 'video' frame=https://player.example/embed
[00:50:50] [browser][source 112] HLS https://cdn.example.com/live/stream.m3u8
[00:50:50] [browser][source 112] done in 5.2s: 1 HLS candidate(s)
[00:50:50] [browser] closed https://example.com/webcam
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

## Discover every enabled source

```bash
uv run python -m scripts.stream_recorder.main discover \
  --sources Place_Overview.csv
```

All enabled non-YouTube source tasks are scheduled immediately. HTTP discovery runs concurrently, while Playwright allows at most four active source tabs by default. Completion progress is written immediately to stderr while the final deterministic JSON array remains on stdout.

Use a lower browser limit when needed:

```bash
uv run python -m scripts.stream_recorder.main discover \
  --sources Place_Overview.csv \
  --browser-concurrency 1
```

The old CLI spelling remains accepted for legacy workbooks:

```bash
uv run python -m scripts.stream_recorder.main discover \
  --xlsb Place_Overview.xlsb
```

## Record

```bash
uv run python -m scripts.stream_recorder.main record \
  --sources Place_Overview.csv \
  --output datasets/recordings
```

The same browser limit applies during initial discovery and later stream rediscovery. FFmpeg recording processes themselves remain independent of this Playwright limit.

Record only selected source IDs:

```bash
uv run python -m scripts.stream_recorder.main record \
  --sources Place_Overview.csv \
  --source-id 112 \
  --source-id 113
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
