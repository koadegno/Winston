# Public Stream Recorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-shot public webcam collector that discovers non-YouTube HLS streams and records every camera into hourly MKV files with traceable metadata.

**Architecture:** `scripts/stream_recorder` owns source loading, HTTP/Playwright discovery, HLS resolution, metadata, FFmpeg recording, and orchestration. Recording output is external data under `datasets/recordings`, not Winston product code.

**Tech Stack:** Python 3.13, standard library, pyxlsb, Playwright/Chromium, FFmpeg, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-22-public-stream-recorder-design.md`

## Global Constraints

- Do not support YouTube or YouTube Live.
- Discover all cameras loaded by the supplied page; do not crawl the rest of the domain.
- Prefer the highest-resolution HLS variant, then highest bandwidth.
- Record with FFmpeg stream copy to MKV.
- Align files to UTC clock hours.
- Re-discover stream URLs after FFmpeg failures.
- Keep all implementation under `scripts/stream_recorder/`.

---

### Task 1: Source loading and discovery

- [x] Write failing tests for YouTube exclusion, source slugging, and `.m3u8` extraction.
- [x] Verify the tests fail before implementation.
- [x] Implement XLSB source loading plus HTTP/iframe and Playwright discovery.
- [x] Verify the focused tests pass.

### Task 2: HLS resolution

- [x] Write failing tests for master parsing and best-quality selection.
- [x] Implement HLS master/variant resolution and de-duplication.
- [x] Verify the focused tests pass.

### Task 3: Recording and orchestration

- [x] Write failing tests for hourly paths, stream-copy command, metadata, camera IDs, and failure re-discovery.
- [x] Implement recording, metadata, and multi-camera supervision.
- [x] Verify the focused tests pass.

### Task 4: CLI and local integration

- [x] Write a failing CLI test.
- [x] Implement `discover` and `record` commands.
- [x] Verify local page -> master HLS -> 1080p variant integration.
- [x] Verify the supplied `Place_Overview.xlsb` loads 27 non-YouTube sources.

### Task 5: CI smoke test and PR

- [ ] Run the complete recorder test suite in GitHub Actions.
- [ ] Install Playwright Chromium in CI.
- [ ] Discover and resolve at least one real public `.m3u8` from a webcam page.
- [ ] Inspect logs and correct failures until green.
- [ ] Open the pull request to `main` only after fresh verification.
