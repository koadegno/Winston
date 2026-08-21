import errno
from pathlib import Path

from winston.ingest.models import MediaFile, MediaType

VIDEO_SUFFIXES = frozenset({".mkv"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg"})


def classify_media(path: Path) -> MediaType | None:
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return MediaType.VIDEO
    if suffix in IMAGE_SUFFIXES:
        return MediaType.IMAGE
    return None


def scan_media(root: Path) -> list[MediaFile]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(root))
    if not root.is_dir():
        raise NotADirectoryError(errno.ENOTDIR, "Not a directory", str(root))

    media: list[MediaFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        media_type = classify_media(path)
        if media_type is not None:
            media.append(MediaFile(path=path, media_type=media_type))
    return media
