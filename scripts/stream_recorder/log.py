"""Small stderr logger for the one-shot stream recorder."""

from __future__ import annotations

from datetime import datetime, timezone
import sys


def log(message: str) -> None:
    """Write one timestamped diagnostic line to stderr immediately."""
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)
