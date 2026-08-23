import json
from pathlib import Path
from uuid import UUID

import pytest

from winston.indexing.manifest import (
    IndexManifest,
    ManifestError,
    load_or_create_dataset_identity,
)

CURRENT_INSTANCE = "ddbbd067-036e-4f45-a36c-69f702377c97"
OLD_INSTANCE = "d44b56ad-b945-4dc7-afc3-5e9af7e19e26"


def test_dataset_identity_is_created_once_and_reused(tmp_path: Path) -> None:
    """One indexing root keeps one stable logical dataset UUID."""
    first = load_or_create_dataset_identity(tmp_path)
    second = load_or_create_dataset_identity(tmp_path)

    assert first == second
    assert str(UUID(first.dataset_instance_id)) == first.dataset_instance_id
    assert (tmp_path / ".winston" / "dataset.json").is_file()


def test_dataset_identity_file_has_exact_versioned_shape(tmp_path: Path) -> None:
    """Dataset ownership state must remain small, explicit, and versioned."""
    identity = load_or_create_dataset_identity(tmp_path)

    payload = json.loads(
        (tmp_path / ".winston" / "dataset.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "schema_version": 1,
        "dataset_instance_id": identity.dataset_instance_id,
    }


def test_invalid_dataset_document_is_fatal(tmp_path: Path) -> None:
    """Corrupt ownership state is never silently replaced with a new identity."""
    state_dir = tmp_path / ".winston"
    state_dir.mkdir()
    (state_dir / "dataset.json").write_text(
        '{"schema_version":1,"dataset_instance_id":"broken"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="dataset.json"):
        load_or_create_dataset_identity(tmp_path)


def test_dataset_identity_rejects_non_directory_root(tmp_path: Path) -> None:
    """Restart state may only be attached to a real indexing directory."""
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(ManifestError, match="directory"):
        load_or_create_dataset_identity(file_path)


def test_missing_manifest_returns_no_completed_assets(tmp_path: Path) -> None:
    """A dataset without prior journal history starts with an empty completed set."""
    manifest = IndexManifest(tmp_path)

    assert manifest.completed_asset_ids(CURRENT_INSTANCE) == set()


def test_manifest_filters_completion_by_index_instance(tmp_path: Path) -> None:
    """Only completion records for the exact current Qdrant collection instance are reusable."""
    manifest = IndexManifest(tmp_path)
    manifest.mark_completed(
        index_instance_id=OLD_INSTANCE,
        asset_id="a" * 64,
        source_path="old.jpg",
    )
    manifest.mark_completed(
        index_instance_id=CURRENT_INSTANCE,
        asset_id="b" * 64,
        source_path="current.jpg",
    )

    assert manifest.completed_asset_ids(CURRENT_INSTANCE) == {"b" * 64}


def test_manifest_ignores_only_a_corrupt_final_non_empty_line(tmp_path: Path) -> None:
    """A crash-truncated final append must not invalidate earlier durable completions."""
    path = tmp_path / ".winston" / "index-state.jsonl"
    path.parent.mkdir()
    path.write_text(
        '{"schema_version":1,"index_instance_id":"'
        + CURRENT_INSTANCE
        + '","asset_id":"'
        + "b" * 64
        + '","source_path":"current.jpg","status":"completed"}\n'
        + '{"schema_version":1,"index_instance_id":',
        encoding="utf-8",
    )

    assert IndexManifest(tmp_path).completed_asset_ids(CURRENT_INSTANCE) == {"b" * 64}


def test_manifest_ignores_invalid_utf8_only_on_final_non_empty_line(tmp_path: Path) -> None:
    """A torn final UTF-8 code point is treated like any other incomplete final append."""
    path = tmp_path / ".winston" / "index-state.jsonl"
    path.parent.mkdir()
    valid = (
        '{"schema_version":1,"index_instance_id":"'
        + CURRENT_INSTANCE
        + '","asset_id":"'
        + "b" * 64
        + '","source_path":"current.jpg","status":"completed"}\n'
    ).encode("utf-8")
    path.write_bytes(valid + b"\xff")

    assert IndexManifest(tmp_path).completed_asset_ids(CURRENT_INSTANCE) == {"b" * 64}


def test_manifest_rejects_corruption_before_final_record(tmp_path: Path) -> None:
    """Corruption inside durable history must be reported with its physical line number."""
    path = tmp_path / ".winston" / "index-state.jsonl"
    path.parent.mkdir()
    path.write_text(
        '{broken}\n'
        + '{"schema_version":1,"index_instance_id":"'
        + CURRENT_INSTANCE
        + '","asset_id":"'
        + "b" * 64
        + '","source_path":"current.jpg","status":"completed"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="line 1"):
        IndexManifest(tmp_path).completed_asset_ids(CURRENT_INSTANCE)


def test_manifest_rejects_invalid_record_before_final_record(tmp_path: Path) -> None:
    """Structurally invalid durable history must not be silently skipped."""
    path = tmp_path / ".winston" / "index-state.jsonl"
    path.parent.mkdir()
    path.write_text(
        '{"schema_version":1,"index_instance_id":"'
        + CURRENT_INSTANCE
        + '","asset_id":"bad","source_path":"bad.jpg","status":"completed"}\n'
        + '{"schema_version":1,"index_instance_id":"'
        + CURRENT_INSTANCE
        + '","asset_id":"'
        + "b" * 64
        + '","source_path":"current.jpg","status":"completed"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="line 1"):
        IndexManifest(tmp_path).completed_asset_ids(CURRENT_INSTANCE)


def test_mark_completed_writes_only_completed_event(tmp_path: Path) -> None:
    """The journal stores no transient running or failed state."""
    manifest = IndexManifest(tmp_path)

    manifest.mark_completed(
        index_instance_id=CURRENT_INSTANCE,
        asset_id="c" * 64,
        source_path="cameras/cam01.mkv",
    )

    payload = json.loads(
        (tmp_path / ".winston" / "index-state.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert payload == {
        "schema_version": 1,
        "index_instance_id": CURRENT_INSTANCE,
        "asset_id": "c" * 64,
        "source_path": "cameras/cam01.mkv",
        "status": "completed",
    }


def test_mark_completed_fsyncs_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completion is acknowledged only after the append is flushed through fsync."""
    calls: list[int] = []

    def fake_fsync(file_descriptor: int) -> None:
        calls.append(file_descriptor)

    monkeypatch.setattr("winston.indexing.manifest.os.fsync", fake_fsync)

    IndexManifest(tmp_path).mark_completed(
        index_instance_id=CURRENT_INSTANCE,
        asset_id="d" * 64,
        source_path="photo.jpg",
    )

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("asset_id", "source_path"),
    [
        ("BAD", "photo.jpg"),
        ("a" * 64, "/absolute.jpg"),
        ("a" * 64, "a/../photo.jpg"),
        ("a" * 64, "a\\photo.jpg"),
    ],
)
def test_mark_completed_rejects_invalid_asset_identity(
    tmp_path: Path,
    asset_id: str,
    source_path: str,
) -> None:
    """Invalid durable identity data must fail before it can enter the journal."""
    with pytest.raises(ManifestError):
        IndexManifest(tmp_path).mark_completed(
            index_instance_id=CURRENT_INSTANCE,
            asset_id=asset_id,
            source_path=source_path,
        )


def test_completed_asset_ids_rejects_invalid_index_instance_id(tmp_path: Path) -> None:
    """Manifest lookup must be scoped by a valid collection instance UUID."""
    with pytest.raises(ManifestError, match="index_instance_id"):
        IndexManifest(tmp_path).completed_asset_ids("broken")
