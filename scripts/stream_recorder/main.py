"""Command-line entry point for discovering and recording public HLS webcams."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .discovery import (
    DEFAULT_BROWSER_CONCURRENCY,
    BrowserSession,
    browser_session,
    discover_page,
)
from .hls import resolve_cameras
from .orchestrator import assign_camera_ids, record_source_forever
from .sources import Source, load_sources


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for bounded concurrency arguments."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


async def discover_url(
    url: str,
    *,
    use_browser_fallback: bool = True,
    resolve: bool = True,
    browser: BrowserSession | None = None,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
) -> dict[str, object]:
    """Discover HLS candidates and optionally resolve logical cameras for one URL."""
    candidates = await discover_page(
        url,
        use_browser_fallback=use_browser_fallback,
        browser=browser,
        browser_concurrency=browser_concurrency,
    )
    cameras = assign_camera_ids(await resolve_cameras(candidates)) if resolve else []
    return {
        "url": url,
        "candidates": sorted(candidates),
        "cameras": [
            {
                "camera_id": camera.id,
                "stream_url": camera.stream.url,
                "resolution": list(camera.stream.resolution) if camera.stream.resolution else None,
                "bandwidth": camera.stream.bandwidth,
            }
            for camera in cameras
        ],
    }


async def _discover_one_source(
    index: int,
    source: Source,
    *,
    use_browser_fallback: bool,
    resolve: bool,
    browser: BrowserSession | None,
    browser_concurrency: int,
) -> tuple[int, dict[str, object]]:
    """Discover one XLSB source and capture failures as structured output."""
    try:
        result = await discover_url(
            source.url,
            use_browser_fallback=use_browser_fallback,
            resolve=resolve,
            browser=browser,
            browser_concurrency=browser_concurrency,
        )
        result["source"] = source.to_dict()
        result["error"] = None
    except Exception as exc:
        result = {
            "url": source.url,
            "source": source.to_dict(),
            "candidates": [],
            "cameras": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return index, result


async def _discover_xlsb(
    path: Path,
    *,
    use_browser_fallback: bool,
    resolve: bool,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
) -> list[dict[str, object]]:
    """Schedule every XLSB source concurrently with bounded browser resource use."""
    sources = load_sources(path)
    print(
        (
            f"[discover] scheduling {len(sources)} sources; "
            f"max {browser_concurrency} active browser tab(s)"
        ),
        file=sys.stderr,
        flush=True,
    )
    if not sources:
        return []

    ordered_results: list[dict[str, object] | None] = [None] * len(sources)
    async with browser_session(
        enabled=use_browser_fallback,
        max_pages=browser_concurrency,
    ) as browser:
        # All sources are async tasks, but browser work queues behind one global semaphore.
        tasks = [
            asyncio.create_task(
                _discover_one_source(
                    index,
                    source,
                    use_browser_fallback=use_browser_fallback,
                    resolve=resolve,
                    browser=browser,
                    browser_concurrency=browser_concurrency,
                )
            )
            for index, source in enumerate(sources)
        ]
        for completed in asyncio.as_completed(tasks):
            index, result = await completed
            ordered_results[index] = result
            source = sources[index]
            status = result.get("error") or f"{len(result['candidates'])} candidate(s)"
            print(
                f"[discover] {source.id} {source.place}: {status}",
                file=sys.stderr,
                flush=True,
            )

    return [result for result in ordered_results if result is not None]


async def _record_one_source(
    source: Source,
    output_root: Path,
    *,
    browser: BrowserSession | None,
) -> None:
    """Run one source recorder without allowing its failure to stop other sources."""
    try:
        await record_source_forever(source, output_root, browser=browser)
    except Exception as exc:
        print(
            f"[{source.id}] {source.url}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


async def _record_sources(
    sources: list[Source],
    output_root: Path,
    *,
    use_browser: bool = True,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
) -> None:
    """Start all source recorders while bounding concurrent Playwright tabs."""
    if not sources:
        raise RuntimeError("No non-YouTube sources found")

    print(
        (
            f"[record] scheduling {len(sources)} sources; "
            f"max {browser_concurrency} active browser tab(s)"
        ),
        file=sys.stderr,
        flush=True,
    )
    async with browser_session(
        enabled=use_browser,
        max_pages=browser_concurrency,
    ) as browser:
        # Recorders are independent; only their browser discovery work is rate-limited.
        await asyncio.gather(
            *(
                _record_one_source(source, output_root, browser=browser)
                for source in sources
            )
        )


def _add_browser_concurrency_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared Playwright tab-limit option to a subcommand parser."""
    parser.add_argument(
        "--browser-concurrency",
        type=_positive_int,
        default=DEFAULT_BROWSER_CONCURRENCY,
        metavar="N",
        help=(
            "Maximum number of Playwright tabs active at once across all sources "
            f"(default: {DEFAULT_BROWSER_CONCURRENCY})."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for discovery and recording commands."""
    parser = argparse.ArgumentParser(
        description="Discover and record public non-YouTube HLS webcam streams."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover",
        help="Discover HLS streams without recording them.",
    )
    target = discover.add_mutually_exclusive_group(required=True)
    target.add_argument("--url")
    target.add_argument("--xlsb", type=Path)
    discover.add_argument(
        "--no-browser",
        action="store_true",
        help="Disable Playwright fallback.",
    )
    discover.add_argument(
        "--candidates-only",
        action="store_true",
        help="Only discover .m3u8 URLs; do not fetch/resolve the HLS playlists.",
    )
    _add_browser_concurrency_argument(discover)

    record = subparsers.add_parser(
        "record",
        help="Record all non-YouTube sources from Place_Overview.xlsb.",
    )
    record.add_argument("--xlsb", type=Path, required=True)
    record.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/recordings"),
    )
    record.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Optional source id filter; repeatable.",
    )
    _add_browser_concurrency_argument(record)
    return parser


async def main(argv: Sequence[str] | None = None) -> int:
    """Execute the stream-recorder CLI asynchronously."""
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        use_browser = not args.no_browser
        resolve = not args.candidates_only
        if args.url:
            print(f"[discover] {args.url}", file=sys.stderr, flush=True)
            payload: dict[str, object] | list[dict[str, object]] = await discover_url(
                args.url,
                use_browser_fallback=use_browser,
                resolve=resolve,
                browser_concurrency=args.browser_concurrency,
            )
        else:
            payload = await _discover_xlsb(
                args.xlsb,
                use_browser_fallback=use_browser,
                resolve=resolve,
                browser_concurrency=args.browser_concurrency,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    sources = load_sources(args.xlsb)
    if args.source_id:
        selected = set(args.source_id)
        sources = [source for source in sources if source.id in selected]
    await _record_sources(
        sources,
        args.output,
        browser_concurrency=args.browser_concurrency,
    )
    return 0


if __name__ == "__main__":
    # The CLI boundary owns the event loop; async tests await main() directly via pytest-asyncio.
    raise SystemExit(asyncio.run(main()))
