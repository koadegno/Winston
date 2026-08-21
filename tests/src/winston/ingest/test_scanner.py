from pathlib import Path

from winston.ingest.models import MediaType
from winston.ingest.scanner import scan_media


def test_scan_media_discovers_supported_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "camera.MKV").touch()
    (tmp_path / "photo.jpg").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "still.JPEG").touch()
    (nested / "notes.txt").touch()

    media = scan_media(tmp_path)

    assert [(item.path.relative_to(tmp_path).as_posix(), item.media_type) for item in media] == [
        ("camera.MKV", MediaType.VIDEO),
        ("nested/still.JPEG", MediaType.IMAGE),
        ("photo.jpg", MediaType.IMAGE),
    ]


def test_scan_media_rejects_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    try:
        scan_media(missing)
    except FileNotFoundError as exc:
        assert exc.filename == str(missing)
    else:
        raise AssertionError("scan_media should reject a missing root")
