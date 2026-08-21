# Public Stream Recorder Design

## Scope

Add a one-shot tooling package under `scripts/stream_recorder/`. It is not part of the Winston runtime package under `src/winston`.

The recorder reads `Place_Overview.xlsb`, ignores every YouTube source, visits each remaining public webcam page, discovers every HLS (`.m3u8`) stream loaded by that page, resolves HLS master playlists to the highest-quality variant, and records every logical camera concurrently.

## Async and concurrency model

`asyncio` is the orchestration layer. Independent work runs concurrently; only true data dependencies remain sequential.

- All non-YouTube source pages are discovered concurrently.
- Direct HTTP source inspection and Playwright network observation run concurrently for each page.
- One Chromium process is shared by a multi-source command.
- Each source page or third-party iframe/player target uses its own isolated Playwright browser context.
- Independent iframe/player targets at the same recursion depth are visited concurrently.
- HLS candidate playlists are fetched/resolved concurrently.
- Every selected source records concurrently, and every camera inside a source records concurrently.
- FFmpeg is spawned with asyncio subprocess APIs and awaited without blocking other cameras.
- Blocking standard-library HTTP calls are isolated with `asyncio.to_thread`; no extra HTTP client dependency is required.

Async tests use `pytest-asyncio` and directly `await` async functions. Test code does not create event loops with `asyncio.run()` or `asyncio.Runner`.

## Discovery

Discovery uses two layers:

1. Direct HTTP scan of the source page for `.m3u8` URLs.
2. Playwright observation of `.m3u8` network requests.

Third-party iframe URLs are treated as independent browser targets. Their internal DOM is not read from the parent page; each target is opened in an isolated browser context with bounded recursion depth.

YouTube URLs are rejected at source-loading time and are never opened by the browser discovery path.

If a browser exposes a master playlist and its variants, the recorder treats the variants as one logical camera and selects the highest resolution, then highest bandwidth variant.

For XLSB discovery, progress is emitted to stderr as sources complete. Final machine-readable JSON stays on stdout and keeps spreadsheet ordering even though sources execute concurrently.

## Recording

Each camera is handled by an async task. FFmpeg uses stream copy (`-c copy`) and writes Matroska (`.mkv`) without re-encoding. Files are aligned to UTC clock hours. If FFmpeg exits with a non-zero status, the worker asynchronously backs off, re-runs discovery for the source page, and retries with the newly discovered URL for the same camera index.

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

## Code documentation

Every production function, including private and nested helpers, has a docstring. Non-obvious concurrency, fallback, and durability decisions also have inline comments. An AST-based test enforces the function-docstring rule.

## Verification

Unit tests cover source filtering, URL extraction, concurrent discovery layers, concurrent iframe targets, concurrent HLS resolution, camera IDs, metadata, async FFmpeg execution, concurrent camera/source orchestration, retry/re-discovery, and CLI progress. A local HTTP integration test verifies page -> master playlist -> best variant.

GitHub Actions installs the project from `pyproject.toml`, installs Chromium, runs the full stream-recorder test suite, and smoke-tests two real public pages: Brussels Grand Place (`livecam.brucity.be`) and Prijedor (`ipcamlive.com`). Success requires live `.m3u8` candidates from both sources.
