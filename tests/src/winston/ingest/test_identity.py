import os
from pathlib import Path

import pytest

from winston.ingest.identity import identify_asset


def test_identify_asset_is_stable_for_unchanged_file(tmp_path: Path) -> None:
    """The same relative path, size, and mtime must produce exactly the same asset identity."""
    root = tmp_path / "data"
    source = root / "cameras" / "cam01.mkv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"camera-data")
    timestamp_ns = 1_787_412_345_678_900_000
    os.utime(source, ns=(timestamp_ns, timestamp_ns))

    first = identify_asset(index_root=root, source_path=source)
    second = identify_asset(index_root=root, source_path=source)

    assert first == second
    assert first.source_path == "cameras/cam01.mkv"
    assert len(first.asset_id) == 64
    assert first.asset_id == first.asset_id.lower()


def test_identify_asset_changes_when_file_size_changes(tmp_path: Path) -> None:
    """Changing source size must create a new concrete media revision identity."""
    root = tmp_path / "data"
    source = root / "cam01.mkv"
    root.mkdir()
    source.write_bytes(b"a")
    before = identify_asset(index_root=root, source_path=source)

    source.write_bytes(b"changed-size")
    after = identify_asset(index_root=root, source_path=source)

    assert before.asset_id != after.asset_id


def test_identify_asset_changes_when_mtime_changes(tmp_path: Path) -> None:
    """Changing nanosecond modification time must create a new asset identity."""
    root = tmp_path / "data"
    source = root / "cam01.mkv"
    root.mkdir()
    source.write_bytes(b"same-data")
    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    before = identify_asset(index_root=root, source_path=source)

    os.utime(source, ns=(2_000_000_000, 2_000_000_000))
    after = identify_asset(index_root=root, source_path=source)

    assert before.asset_id != after.asset_id


def test_identify_asset_changes_when_relative_path_changes(tmp_path: Path) -> None:
    """Moving identical file metadata within the dataset must change the asset identity."""
    root = tmp_path / "data"
    first_path = root / "camera-a.mkv"
    second_path = root / "nested" / "camera-a.mkv"
    second_path.parent.mkdir(parents=True)
    first_path.write_bytes(b"same-data")
    second_path.write_bytes(b"same-data")
    timestamp_ns = 3_000_000_000
    os.utime(first_path, ns=(timestamp_ns, timestamp_ns))
    os.utime(second_path, ns=(timestamp_ns, timestamp_ns))

    first = identify_asset(index_root=root, source_path=first_path)
    second = identify_asset(index_root=root, source_path=second_path)

    assert first.asset_id != second.asset_id
    assert first.source_path == "camera-a.mkv"
    assert second.source_path == "nested/camera-a.mkv"


def test_identify_asset_rejects_source_outside_index_root(tmp_path: Path) -> None:
    """Asset identity must never silently encode an absolute path outside the dataset root."""
    root = tmp_path / "data"
    root.mkdir()
    source = tmp_path / "outside.mkv"
    source.write_bytes(b"x")

    with pytest.raises(ValueError, match="inside the indexing root"):
        identify_asset(index_root=root, source_path=source)


def test_identify_asset_rejects_directory_source(tmp_path: Path) -> None:
    """Only concrete media files can receive asset identities."""
    root = tmp_path / "data"
    source = root / "camera-directory"
    source.mkdir(parents=True)

    with pytest.raises(ValueError, match="source path must reference a file"):
        identify_asset(index_root=root, source_path=source)
