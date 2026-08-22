"""Command-line entry point for discovering and recording public HLS webcams."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from .discovery import (
    DEFAULT_BROWSER_CONCURRENCY,
    BrowserSession,
    browser_session,
    discover_browser,
    discover_http,
    discover_page,
    http_session,
)
from .hls import resolve_cameras
from .log import log
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
    use_browser: bool = True,
    resolve: bool = True,
    browser: BrowserSession | None = None,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
    client: Any | None = None,
) -> dict[str, object]:
    """Discover HLS candidates and optionally resolve logical cameras for one URL."""
    if client is None:
        async with http_session() as local_client:
            return await discover_url(
                url,
                use_browser=use_browser,
                resolve=resolve,
                browser=browser,
                browser_concurrency=browser_concurrency,
                client=local_client,
            )
    if use_browser and browser is None:
        async with browser_session(max_pages=browser_concurrency) as local_browser:
            if local_browser is None:
                raise RuntimeError("browser session unexpectedly disabled")
            return await discover_url(
                url,
                use_browser=True,
                resolve=resolve,
                browser=local_browser,
                browser_concurrency=browser_concurrency,
                client=client,
            )

    browser_observed_urls: set[str] = set()
    candidates = await discover_page(
        url,
        use_browser=use_browser,
        browser=browser,
        browser_concurrency=browser_concurrency,
        client=client,
        browser_observed_urls=browser_observed_urls,
    )
    cameras = (
        assign_camera_ids(
            await resolve_cameras(
                candidates,
                client=client,
                browser_observed_urls=browser_observed_urls,
            )
        )
        if resolve
        else []
    )
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


async def _discover_http_source(
    index: int,
    source: Source,
    *,
    client: Any,
) -> tuple[int, set[str], BaseException | None]:
    """Run stage-1 HTTP discovery for one configured source."""
    try:
        streams = await discover_http(
            source.url,
            client=client,
            label=f"source {source.id}",
        )
    except Exception as exc:
        return index, set(), exc
    return index, streams, None


async def _discover_browser_source(
    index: int,
    source: Source,
    *,
    browser: BrowserSession,
) -> tuple[int, set[str]]:
    """Run stage-2 one-page browser discovery for one configured source."""
    streams = await discover_browser(
        source.url,
        browser=browser,
        label=f"source {source.id}",
    )
    return index, streams


async def _resolve_source(
    index: int,
    candidates: set[str],
    browser_observed_urls: set[str],
    *,
    client: Any,
) -> tuple[int, list[dict[str, object]]]:
    """Resolve one source's HLS candidates into deterministic camera dictionaries."""
    cameras = assign_camera_ids(
        await resolve_cameras(
            candidates,
            client=client,
            browser_observed_urls=browser_observed_urls,
        )
    )
    return index, [
        {
            "camera_id": camera.id,
            "stream_url": camera.stream.url,
            "resolution": list(camera.stream.resolution) if camera.stream.resolution else None,
            "bandwidth": camera.stream.bandwidth,
        }
        for camera in cameras
    ]


async def _discover_xlsb(
    path: Path,
    *,
    use_browser: bool,
    resolve: bool,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
) -> list[dict[str, object]]:
    """Discover all configured sources in explicit HTTP, browser, and HLS stages."""
    sources = load_sources(path)
    log(
        f"[discover] loaded {len(sources)} enabled non-YouTube source(s) from {path}; "
        f"browser tabs={browser_concurrency}"
    )
    if not sources:
        return []

    candidates_by_source: list[set[str]] = [set() for _ in sources]
    browser_observed_by_source: list[set[str]] = [set() for _ in sources]
    errors_by_source: list[BaseException | None] = [None for _ in sources]
    cameras_by_source: list[list[dict[str, object]]] = [[] for _ in sources]

    async with http_session() as client:
        log(f"[discover] STAGE 1/3 HTTP start: {len(sources)} source(s) in parallel")
        http_tasks = [
            asyncio.create_task(_discover_http_source(index, source, client=client))
            for index, source in enumerate(sources)
        ]
        for completed in asyncio.as_completed(http_tasks):
            index, streams, error = await completed
            candidates_by_source[index].update(streams)
            errors_by_source[index] = error
            source = sources[index]
            status = (
                f"ERROR {type(error).__name__}: {error}"
                if error is not None
                else f"{len(streams)} candidate(s)"
            )
            log(f"[discover] HTTP source {source.id} DONE {source.place}: {status}")
        log("[discover] STAGE 1/3 HTTP complete")

        if use_browser:
            log(
                f"[discover] STAGE 2/3 BROWSER start: one page/source, "
                f"max {browser_concurrency} active tab(s)"
            )
            async with browser_session(max_pages=browser_concurrency) as browser:
                if browser is None:
                    raise RuntimeError("browser session unexpectedly disabled")

                # Every source is visited in Playwright even after an HTTP hit because dynamic
                # frame traffic can expose additional cameras absent from static page content.
                browser_tasks = [
                    asyncio.create_task(
                        _discover_browser_source(index, source, browser=browser)
                    )
                    for index, source in enumerate(sources)
                ]
                for completed in asyncio.as_completed(browser_tasks):
                    index, streams = await completed
                    candidates_by_source[index].update(streams)
                    browser_observed_by_source[index].update(streams)
                    source = sources[index]
                    log(
                        f"[discover] BROWSER source {source.id} DONE {source.place}: "
                        f"{len(streams)} candidate(s)"
                    )
            log("[discover] STAGE 2/3 BROWSER complete")
        else:
            log("[discover] STAGE 2/3 BROWSER skipped (--no-browser)")

        if resolve:
            log("[discover] STAGE 3/3 HLS resolve start")
            resolve_tasks = [
                asyncio.create_task(
                    _resolve_source(
                        index,
                        candidates,
                        browser_observed_by_source[index],
                        client=client,
                    )
                )
                for index, candidates in enumerate(candidates_by_source)
                if candidates
            ]
            for completed in asyncio.as_completed(resolve_tasks):
                index, cameras = await completed
                cameras_by_source[index] = cameras
                source = sources[index]
                log(
                    f"[discover] HLS source {source.id} DONE {source.place}: "
                    f"{len(cameras)} logical camera(s)"
                )
            log("[discover] STAGE 3/3 HLS resolve complete")
        else:
            log("[discover] STAGE 3/3 HLS resolve skipped (--candidates-only)")

    results: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        candidates = candidates_by_source[index]
        error = errors_by_source[index] if not candidates else None
        result = {
            "url": source.url,
            "source": source.to_dict(),
            "candidates": sorted(candidates),
            "cameras": cameras_by_source[index],
            "error": f"{type(error).__name__}: {error}" if error is not None else None,
        }
        results.append(result)
        status = result["error"] or f"{len(candidates)} candidate(s)"
        log(f"[discover] source {source.id} FINAL {source.place}: {status}")

    return results


async def _record_one_source(
    source: Source,
    output_root: Path,
    *,
    client: Any,
    browser: BrowserSession | None,
) -> None:
    """Run one source recorder without allowing its failure to stop other sources."""
    log(f"[record] source {source.id} START {source.place}: {source.url}")
    try:
        await record_source_forever(
            source,
            output_root,
            client=client,
            browser=browser,
        )
    except Exception as exc:
        log(f"[record] source {source.id} ERROR {source.url}: {type(exc).__name__}: {exc}")


async def _record_sources(
    sources: list[Source],
    output_root: Path,
    *,
    use_browser: bool = True,
    browser_concurrency: int = DEFAULT_BROWSER_CONCURRENCY,
) -> None:
    """Start all source recorders with shared async HTTP and bounded Playwright resources."""
    if not sources:
        raise RuntimeError("No enabled non-YouTube sources found")

    log(
        f"[record] scheduling {len(sources)} sources; "
        f"max {browser_concurrency} active browser tab(s); output={output_root}"
    )
    async with http_session() as client:
        async with browser_session(
            enabled=use_browser,
            max_pages=browser_concurrency,
        ) as browser:
            # Long-lived source recorders are independent; only browser access is bounded.
            await asyncio.gather(
                *(
                    _record_one_source(
                        source,
                        output_root,
                        client=client,
                        browser=browser,
                    )
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
            "Maximum number of Playwright source tabs active at once "
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
    target.add_argument(
        "--sources",
        "--xlsb",
        dest="sources",
        type=Path,
        help="CSV source list (preferred) or legacy XLSB workbook.",
    )
    discover.add_argument(
        "--no-browser",
        action="store_true",
        help="Disable Playwright discovery.",
    )
    discover.add_argument(
        "--candidates-only",
        action="store_true",
        help="Only discover .m3u8 URLs; do not fetch/resolve the HLS playlists.",
    )
    _add_browser_concurrency_argument(discover)

    record = subparsers.add_parser(
        "record",
        help="Record all enabled non-YouTube sources from the configured source list.",
    )
    record.add_argument(
        "--sources",
        "--xlsb",
        dest="sources",
        type=Path,
        required=True,
        help="CSV source list (preferred) or legacy XLSB workbook.",
    )
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
            log(f"[discover] single URL {args.url}")
            payload: dict[str, object] | list[dict[str, object]] = await discover_url(
                args.url,
                use_browser=use_browser,
                resolve=resolve,
                browser_concurrency=args.browser_concurrency,
            )
        else:
            payload = await _discover_xlsb(
                args.sources,
                use_browser=use_browser,
                resolve=resolve,
                browser_concurrency=args.browser_concurrency,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    sources = load_sources(args.sources)
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
