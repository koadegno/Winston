# Stream Recorder Async Parallel Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the public stream recorder so source discovery, iframe/player discovery, HLS resolution, camera recording, and source recording run concurrently without blocking the CLI, while preserving deterministic output and adding immediate progress feedback.

**Architecture:** Use `asyncio` as the orchestration layer. Playwright moves to its async API with one shared Chromium browser and one isolated context per browser target; HTTP/HLS urllib calls remain standard-library code but are moved off the event loop with `asyncio.to_thread`. FFmpeg is spawned with `asyncio.create_subprocess_exec`, and source/camera tasks are coordinated with `asyncio.gather` / `asyncio.as_completed`.

**Tech Stack:** Python >=3.13, asyncio, Playwright async API, urllib, FFmpeg, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-08-22-public-stream-recorder-design.md`

## Global Constraints

- `scripts/stream_recorder/` remains one-shot tooling and must not move into `src/winston`.
- YouTube and YouTube Live remain unsupported and filtered before discovery.
- All independent sources, browser targets, HLS candidates, and camera recorders run concurrently.
- Use async APIs for Playwright, process execution, sleeps, and orchestration; use `asyncio.to_thread` only for unavoidable blocking standard-library I/O.
- Async tests use `pytest-asyncio` and `await`; tests must not call `asyncio.run()` or `asyncio.Runner`.
- Every production function, including private helpers and properties, must have a docstring. Non-obvious concurrency and fallback behavior must also have comments.
- `discover --xlsb` must print progress to stderr immediately while keeping final JSON on stdout.
- Keep output recording format as hour-aligned UTC MKV with FFmpeg stream copy.
- Preserve the user's `pyproject.toml` / `uv.lock` dependency updates.
- If `main` integration becomes necessary, rebase the feature branch; do not merge `main` into the feature branch.

---

### Task 1: Async browser and page discovery

**Files:**
- Modify: `scripts/stream_recorder/discovery.py`
- Test: `tests/scripts/stream_recorder/test_discovery.py`

**Interfaces:**
- Produces: `async fetch_text(...)`, `async discover_http(...)`, `async crawl_browser_targets(...)`, `async discover_browser(...)`, `async discover_page(...)`, `browser_session()`.

- [ ] Write async regression tests proving sibling iframe targets start concurrently and use `pytest.mark.asyncio`.
- [ ] Run the targeted tests and verify they fail because the current discovery functions are synchronous/sequential.
- [ ] Implement async Playwright discovery with a shared browser and isolated context per target.
- [ ] Run the targeted tests and verify they pass.

### Task 2: Parallel HLS candidate resolution

**Files:**
- Modify: `scripts/stream_recorder/hls.py`
- Test: `tests/scripts/stream_recorder/test_hls.py`

**Interfaces:**
- Produces: `async resolve_candidate(url)` and `async resolve_cameras(candidate_urls)`.

- [ ] Write an async regression test proving multiple candidate playlists are resolved concurrently.
- [ ] Verify RED.
- [ ] Resolve candidates with `asyncio.gather` while preserving master/variant deduplication.
- [ ] Verify GREEN.

### Task 3: Async FFmpeg and camera/source orchestration

**Files:**
- Modify: `scripts/stream_recorder/recorder.py`
- Modify: `scripts/stream_recorder/orchestrator.py`
- Test: `tests/scripts/stream_recorder/test_recorder.py`
- Test: `tests/scripts/stream_recorder/test_orchestrator.py`
- Test: `tests/scripts/stream_recorder/test_integration_local.py`

**Interfaces:**
- Produces: `async record_one_hour_slice(...)`, `async discover_source_cameras(...)`, `async record_camera_loop(...)`, `async record_source_forever(...)`.

- [ ] Write async tests for FFmpeg subprocess awaiting, retry sleep/re-discovery, and concurrent camera workers.
- [ ] Verify RED.
- [ ] Replace blocking `subprocess.run`, `time.sleep`, and thread pools with asyncio equivalents.
- [ ] Convert the local HTTP integration test to `pytest-asyncio` and await discovery.
- [ ] Verify GREEN.

### Task 4: Parallel XLSB CLI with progress

**Files:**
- Modify: `scripts/stream_recorder/main.py`
- Test: `tests/scripts/stream_recorder/test_main.py`

**Interfaces:**
- Produces: async CLI `main(argv)` and concurrent `_discover_xlsb(...)` / `_record_sources(...)`.

- [ ] Write async tests proving XLSB sources start concurrently, `--candidates-only` is respected for XLSB, and progress is emitted to stderr before final JSON.
- [ ] Verify RED.
- [ ] Implement concurrent source tasks with one shared browser session and deterministic result ordering.
- [ ] Keep the module entrypoint as the only production boundary allowed to call `asyncio.run(main())`.
- [ ] Verify GREEN.

### Task 5: Documentation policy and live smoke tests

**Files:**
- Modify: all Python files under `scripts/stream_recorder/` as needed for docstrings/comments.
- Create: `tests/scripts/stream_recorder/test_documentation.py`
- Modify: `scripts/stream_recorder/README.md`
- Modify: `.github/workflows/stream-recorder.yml`
- Modify: `docs/superpowers/specs/2026-08-22-public-stream-recorder-design.md`

**Interfaces:**
- Enforces: production functions, async functions, methods, and properties have docstrings.

- [ ] Add an AST test that fails for any undocumented production function/helper in `scripts/stream_recorder`.
- [ ] Verify RED against the current code.
- [ ] Add missing docstrings and comments, then verify GREEN.
- [ ] Update docs to describe shared-browser async parallelism and live progress.
- [ ] Update CI to install project dependencies from `pyproject.toml` and smoke-test both Brussels Grand Place and Prijedor public pages.
- [ ] Run the full test suite and inspect the final GitHub Actions logs; require both public smoke tests to discover `.m3u8` URLs before declaring the PR ready.
