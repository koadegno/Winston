from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Sequence

from .discovery import discover_page
from .hls import resolve_cameras
from .orchestrator import assign_camera_ids, record_source_forever
from .sources import Source, load_sources


def discover_url(
    url: str,
    *,
    use_browser_fallback: bool = True,
    resolve: bool = True,
) -> dict[str, object]:
    candidates = discover_page(url, use_browser_fallback=use_browser_fallback)
    cameras = assign_camera_ids(resolve_cameras(candidates)) if resolve else []
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


def _discover_xlsb(path: Path, *, use_browser_fallback: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for source in load_sources(path):
        try:
            result = discover_url(source.url, use_browser_fallback=use_browser_fallback)
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
        results.append(result)
    return results


def _record_sources(sources: list[Source], output_root: Path) -> None:
    if not sources:
        raise RuntimeError("No non-YouTube sources found")
    with ThreadPoolExecutor(max_workers=len(sources), thread_name_prefix="webcam-source") as pool:
        futures = {pool.submit(record_source_forever, source, output_root): source for source in sources}
        for future in as_completed(futures):
            source = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[{source.id}] {source.url}: {type(exc).__name__}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and record public non-YouTube HLS webcam streams.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover HLS streams without recording them.")
    target = discover.add_mutually_exclusive_group(required=True)
    target.add_argument("--url")
    target.add_argument("--xlsb", type=Path)
    discover.add_argument("--no-browser", action="store_true", help="Disable Playwright fallback.")
    discover.add_argument(
        "--candidates-only",
        action="store_true",
        help="Only discover .m3u8 URLs; do not fetch/resolve the HLS playlists.",
    )

    record = subparsers.add_parser("record", help="Record all non-YouTube sources from Place_Overview.xlsb.")
    record.add_argument("--xlsb", type=Path, required=True)
    record.add_argument("--output", type=Path, default=Path("datasets/recordings"))
    record.add_argument("--source-id", action="append", default=[], help="Optional source id filter; repeatable.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        use_browser = not args.no_browser
        payload = (
            discover_url(
                args.url,
                use_browser_fallback=use_browser,
                resolve=not args.candidates_only,
            )
            if args.url
            else _discover_xlsb(args.xlsb, use_browser_fallback=use_browser)
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    sources = load_sources(args.xlsb)
    if args.source_id:
        selected = set(args.source_id)
        sources = [source for source in sources if source.id in selected]
    _record_sources(sources, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
