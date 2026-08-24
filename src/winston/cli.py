"""Command-line entry points for Winston."""

import argparse
import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import logging
from pathlib import Path
import sys

from winston.config import Settings, get_config
from winston.indexing.models import IndexRunResult
from winston.indexing.pipeline import run_indexing
from winston.ingest.models import ImageMetadata, VideoMetadata
from winston.ingest.probe import probe_media
from winston.ingest.scanner import scan_media
from winston.search.models import SearchResult
from winston.search.pipeline import run_search

PROGRESS_LOGGER = logging.getLogger("winston.progress")


def _format_duration(seconds: float) -> str:
    """Format a media duration as HH:MM:SS.mmm."""
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _positive_int(value: str) -> int:
    """Parse one strictly positive integer for argparse result-count options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


@contextmanager
def _index_progress_logging() -> Iterator[None]:
    """Surface Winston INFO progress on stderr without enabling third-party INFO loggers."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = PROGRESS_LOGGER.level
    previous_propagate = PROGRESS_LOGGER.propagate
    PROGRESS_LOGGER.addHandler(handler)
    PROGRESS_LOGGER.setLevel(logging.INFO)
    PROGRESS_LOGGER.propagate = False
    try:
        yield
    finally:
        PROGRESS_LOGGER.removeHandler(handler)
        PROGRESS_LOGGER.setLevel(previous_level)
        PROGRESS_LOGGER.propagate = previous_propagate


def _print_metadata(metadata: VideoMetadata | ImageMetadata, root: Path) -> None:
    """Print one scanned media record relative to the requested root when possible."""
    try:
        display_path = metadata.path.relative_to(root)
    except ValueError:
        display_path = metadata.path

    if isinstance(metadata, VideoMetadata):
        print(f"VIDEO  {display_path}")
        print(f"       {metadata.width}x{metadata.height}")
        print(f"       {metadata.codec}")
        print(f"       {metadata.fps:g} fps")
        print(f"       {_format_duration(metadata.duration_seconds)}")
    else:
        print(f"IMAGE  {display_path}")
        print(f"       {metadata.width}x{metadata.height}")


def scan_command(root: Path) -> int:
    """Discover and synchronously print metadata for supported local media."""
    resolved_root = root.resolve()
    for media in scan_media(resolved_root):
        _print_metadata(probe_media(media), resolved_root)
    return 0


def _print_index_result(result: IndexRunResult) -> None:
    """Print a concise indexing summary plus actionable per-asset failures."""
    print(f"Indexed: {result.indexed}")
    print(f"Skipped: {result.skipped}")
    print(f"Failed: {result.failed}")
    for failure in result.failures:
        print()
        print(f"FAILED {failure.source_path} [{failure.stage.value}]")
        print(f"  {failure.message}")


def index_command(root: Path, settings: Settings) -> int:
    """Run restartable asynchronous indexing and map the result to a shell exit status."""
    with _index_progress_logging():
        try:
            resolved_root = root.resolve(strict=True)
            if not resolved_root.is_dir():
                raise ValueError(f"indexing root is not a directory: {resolved_root}")
            PROGRESS_LOGGER.info("Winston index: %s", resolved_root)
            result = asyncio.run(run_indexing(resolved_root, settings))
        except Exception as exc:
            print(f"Indexing failed: {exc}", file=sys.stderr)
            return 1

    _print_index_result(result)
    return result.exit_code


def _print_search_result(position: int, result: SearchResult) -> None:
    """Print one ranked semantic result with raw score and exact visual provenance."""
    print(f"{position}. {result.source_path}")
    if result.start_timestamp_seconds is not None:
        end = result.end_timestamp_seconds
        representative = result.representative_timestamp_seconds
        if end is None or representative is None:
            raise ValueError("video search result is missing temporal provenance")
        print(
            "   passage: "
            f"{_format_duration(result.start_timestamp_seconds)} -> {_format_duration(end)}"
        )
        print(f"   representative: {_format_duration(representative)}")
    region = result.region
    print(
        f"   region: {result.region_kind.value} "
        f"x={region.x} y={region.y} width={region.width} height={region.height} "
        f"scale={region.scale:g}"
    )
    # Keep the cosine exactly as a raw score. It is not calibrated confidence and must
    # not be rendered as a percentage.
    print(f"   raw score: {result.raw_score:.6f}")


def search_command(query: str, limit: int, settings: Settings) -> int:
    """Run one asynchronous semantic search and map results or failures to CLI output."""
    normalized_query = query.strip()
    if not normalized_query:
        print("Search failed: search query must not be blank", file=sys.stderr)
        return 1

    try:
        results = asyncio.run(
            run_search(
                normalized_query,
                limit=limit,
                settings=settings,
            )
        )
    except Exception as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    for position, result in enumerate(results, start=1):
        if position > 1:
            print()
        _print_search_result(position, result)
    return 0


def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """Build Winston's CLI parser from one runtime settings snapshot."""
    config = settings or get_config()
    parser = argparse.ArgumentParser(prog="winston")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Discover and inspect local media files",
    )
    scan_parser.add_argument("path", nargs="?", type=Path, default=config.data_dir)

    index_parser = subparsers.add_parser(
        "index",
        help="Index local videos and photos into the configured visual index",
    )
    index_parser.add_argument("path", nargs="?", type=Path, default=config.data_dir)

    search_parser = subparsers.add_parser(
        "search",
        help="Search the configured visual index with natural language",
    )
    search_parser.add_argument("query", help="Natural-language semantic search query")
    search_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=int(config.search.result_limit),
        help="Maximum number of grouped results to print",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse Winston CLI arguments and dispatch one supported command."""
    config = get_config()
    args = build_parser(config).parse_args(argv)
    if args.command == "scan":
        return scan_command(args.path)
    if args.command == "index":
        return index_command(args.path, config)
    if args.command == "search":
        return search_command(args.query, args.limit, config)
    raise AssertionError(f"Unhandled command: {args.command}")
