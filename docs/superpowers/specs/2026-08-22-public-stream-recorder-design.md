# Public Stream Recorder Design

## Scope

Add a one-shot tooling package under `scripts/stream_recorder/`. It is not part of the Winston runtime package under `src/winston`.

The recorder reads `Place_Overview.xlsb`, ignores every YouTube source, visits each remaining public webcam page, discovers every HLS (`.m3u8`) stream loaded by that page, resolves HLS master playlists to the highest-quality variant, and records every logical camera concurrently.

## Discovery

Discovery uses two layers:

1. HTTP scan of the page source and first-level iframes for `.m3u8` URLs.
2. Playwright browser observation of network requests plus rendered HTML.

YouTube URLs are rejected at source-loading time and are never opened by the browser discovery path.

If a browser exposes a master playlist and its variants, the recorder treats the variants as one logical camera and selects the highest resolution, then highest bandwidth variant.

## Recording

Each camera is handled by a dedicated worker. FFmpeg uses stream copy (`-c copy`) and writes Matroska (`.mkv`) without re-encoding. Files are aligned to UTC clock hours. If FFmpeg exits with a non-zero status, the worker re-runs discovery for the source page and retries with the newly discovered URL for the same camera index.

## Storage

```text
datasets/recordings/
└── <country>/<city>/<place>/
    ├── source.json
    └── cameras/
        └── camera-001/
            ├── camera.json
            └── YYYY/MM/DD/YYYY-MM-DDTHH-00-00Z.mkv
```

`source.json` preserves source identity and origin. `camera.json` preserves the current HLS URL, resolution, bandwidth, first-seen time, and last-seen time. `datasets/recordings/` is gitignored.

## Verification

Unit tests cover source filtering, URL extraction, HLS resolution, camera IDs, metadata, FFmpeg command construction, and retry/re-discovery. A local HTTP integration test verifies page -> master playlist -> best variant. GitHub Actions installs Chromium and performs a network smoke test against a real public webcam page; success requires at least one resolved live `.m3u8`.
