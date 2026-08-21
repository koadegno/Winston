import argparse
from pathlib import Path
from typing import Sequence

from winston.config import get_config
from winston.ingest.models import ImageMetadata, VideoMetadata
from winston.ingest.probe import probe_media
from winston.ingest.scanner import scan_media


def _format_duration(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _print_metadata(metadata: VideoMetadata | ImageMetadata, root: Path) -> None:
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
    root = root.resolve()
    for media in scan_media(root):
        _print_metadata(probe_media(media), root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    config = get_config()
    parser = argparse.ArgumentParser(prog="winston")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Discover and inspect local media files")
    scan_parser.add_argument("path", nargs="?", type=Path, default=config.data_dir)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        return scan_command(args.path)
    raise AssertionError(f"Unhandled command: {args.command}")
