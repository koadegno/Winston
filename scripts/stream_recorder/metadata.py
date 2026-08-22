"""Persist recorder metadata atomically to avoid partial JSON files."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON through a temporary sibling file and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # A sibling temp file keeps the final replace atomic on the same filesystem.
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    temp_path.replace(path)
