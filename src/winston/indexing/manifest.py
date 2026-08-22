"""Durable dataset identity and asset-level restart state for indexing."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import os
import re
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

DATASET_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
ASSET_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(RuntimeError):
    """Raised when Winston restart state cannot be trusted or persisted."""


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Stable logical identity stored with one indexing root."""

    dataset_instance_id: str


class _DatasetDocument(BaseModel):
    """Strict serialized representation of `.winston/dataset.json`."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    dataset_instance_id: UUID


class _CompletionRecord(BaseModel):
    """One completed asset tied to one exact Qdrant collection instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    index_instance_id: UUID
    asset_id: str
    source_path: str
    status: Literal["completed"]

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, asset_id: str) -> str:
        """Require the canonical lowercase SHA-256 asset identifier."""
        if ASSET_ID_PATTERN.fullmatch(asset_id) is None:
            raise ValueError(
                "asset_id must be a 64-character lowercase SHA-256 hexadecimal string"
            )
        return asset_id

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, source_path: str) -> str:
        """Require a normalized portable path relative to the indexing root."""
        if not source_path or "\\" in source_path:
            raise ValueError("source_path must be a non-empty relative POSIX path")
        if source_path.startswith("/") or source_path.endswith("/") or "//" in source_path:
            raise ValueError("source_path must be a normalized relative POSIX path")
        if any(part in {"", ".", ".."} for part in source_path.split("/")):
            raise ValueError(
                "source_path must be a normalized relative POSIX path without '.' or '..'"
            )
        if PurePosixPath(source_path).is_absolute():
            raise ValueError("source_path must be a normalized relative POSIX path")
        return source_path


def _resolve_root(root: Path) -> Path:
    """Resolve and validate one existing indexing-root directory."""
    try:
        resolved = Path(root).resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"Indexing root does not exist or cannot be resolved: {root}") from exc
    if not resolved.is_dir():
        raise ManifestError(f"Indexing root must be a directory: {resolved}")
    return resolved


def _parse_uuid(value: str, *, field_name: str) -> UUID:
    """Parse one UUID string and expose a manifest-specific validation error."""
    try:
        return UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ManifestError(f"{field_name} must be a valid UUID, got {value!r}") from exc


def load_or_create_dataset_identity(root: Path) -> DatasetIdentity:
    """Load one stable dataset UUID or atomically create it on first indexing."""
    resolved_root = _resolve_root(root)
    state_dir = resolved_root / ".winston"
    dataset_path = state_dir / "dataset.json"

    try:
        state_dir.mkdir(exist_ok=True)
    except OSError as exc:
        raise ManifestError(f"Failed to create Winston state directory {state_dir}") from exc

    if dataset_path.exists():
        try:
            document = _DatasetDocument.model_validate_json(
                dataset_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError) as exc:
            raise ManifestError(f"Invalid Winston dataset identity file {dataset_path}") from exc
        return DatasetIdentity(dataset_instance_id=str(document.dataset_instance_id))

    document = _DatasetDocument(
        schema_version=DATASET_SCHEMA_VERSION,
        dataset_instance_id=uuid4(),
    )
    temporary_path: Path | None = None
    try:
        # Write beside the destination so os.replace() is an atomic rename on the same filesystem.
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_dir,
            prefix=".dataset-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(document.model_dump_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, dataset_path)
    except OSError as exc:
        raise ManifestError(f"Failed to persist Winston dataset identity file {dataset_path}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass

    return DatasetIdentity(dataset_instance_id=str(document.dataset_instance_id))


class IndexManifest:
    """Append-only durable completion journal stored under one indexing root."""

    def __init__(self, root: Path) -> None:
        """Bind restart history to `<root>/.winston/index-state.jsonl`."""
        resolved_root = _resolve_root(root)
        self._state_dir = resolved_root / ".winston"
        self._path = self._state_dir / "index-state.jsonl"

    def completed_asset_ids(self, index_instance_id: str) -> set[str]:
        """Return completed asset IDs for exactly one Qdrant collection instance."""
        target_instance = _parse_uuid(
            index_instance_id,
            field_name="index_instance_id",
        )
        if not self._path.exists():
            return set()

        try:
            raw = self._path.read_bytes()
        except OSError as exc:
            raise ManifestError(f"Failed to read Winston index manifest {self._path}") from exc

        # Preserve physical line numbers while ignoring blank lines between valid JSONL records.
        non_empty_lines = [
            (line_number, line)
            for line_number, line in enumerate(raw.split(b"\n"), start=1)
            if line.strip()
        ]
        completed: set[str] = set()
        for position, (line_number, raw_line) in enumerate(non_empty_lines):
            is_final_non_empty = position == len(non_empty_lines) - 1
            try:
                text = raw_line.decode("utf-8")
                record = _CompletionRecord.model_validate_json(text)
            except (UnicodeDecodeError, ValidationError) as exc:
                # A crash may tear only the append currently at the end of the file. It cannot
                # represent a trusted completion, so ignoring that one final record is safe.
                if is_final_non_empty:
                    continue
                raise ManifestError(
                    f"Corrupt Winston index manifest {self._path} at line {line_number}"
                ) from exc

            if record.index_instance_id == target_instance:
                completed.add(record.asset_id)
        return completed

    def mark_completed(
        self,
        *,
        index_instance_id: str,
        asset_id: str,
        source_path: str,
    ) -> None:
        """Append one completed record and fsync it before reporting success."""
        try:
            record = _CompletionRecord(
                schema_version=MANIFEST_SCHEMA_VERSION,
                index_instance_id=_parse_uuid(
                    index_instance_id,
                    field_name="index_instance_id",
                ),
                asset_id=asset_id,
                source_path=source_path,
                status="completed",
            )
        except (ManifestError, ValidationError) as exc:
            if isinstance(exc, ManifestError):
                raise
            raise ManifestError(
                f"Invalid completion record for asset {source_path!r}"
            ) from exc

        try:
            self._state_dir.mkdir(exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json())
                handle.write("\n")
                handle.flush()
                # Completion becomes durable only after the append reaches the filesystem boundary.
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ManifestError(
                f"Failed to persist completed asset {source_path!r} in {self._path}"
            ) from exc
