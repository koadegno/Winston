"""Deterministic portable identity for source media assets."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

ASSET_IDENTITY_PREFIX = "winston:asset:v1:"


@dataclass(frozen=True, slots=True)
class AssetIdentity:
    """Stable identity and portable relative path for one concrete source media revision."""

    asset_id: str
    source_path: str


def identify_asset(*, index_root: Path, source_path: Path) -> AssetIdentity:
    """Return a deterministic asset ID from canonical path and filesystem revision metadata."""
    root = Path(index_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("indexing root must reference a directory")

    source = Path(source_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("source path must reference a file")

    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError("source path must be inside the indexing root") from exc

    relative_path = relative.as_posix()
    stat = source.stat()
    material = {
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "relative_path": relative_path,
    }
    canonical = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(
        f"{ASSET_IDENTITY_PREFIX}{canonical}".encode("utf-8")
    ).hexdigest()
    return AssetIdentity(asset_id=digest, source_path=relative_path)
