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

PROGRESS_LOGGER = logging.getLogger("winston.progress")


def _format_duration(seconds: float) -> str:
    """Format a media duration as HH:MM:SS.mmm."""
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse Winston CLI arguments and dispatch one supported command."""
    config = get_config()
    args = build_parser(config).parse_args(argv)
    if args.command == "scan":
        return scan_command(args.path)
    if args.command == "index":
        return index_command(args.path, config)
    raise AssertionError(f"Unhandled command: {args.command}")
