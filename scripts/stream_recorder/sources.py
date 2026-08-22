"""Load and normalize public webcam sources from CSV or legacy XLSB files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping
import unicodedata
from urllib.parse import urlparse


def slugify(value: str) -> str:
    """Convert a human-readable location component into a filesystem-safe slug."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-") or "unknown"


def is_youtube_url(url: str) -> bool:
    """Return whether a URL points to YouTube, which this recorder intentionally excludes."""
    host = urlparse(url).netloc.lower().split(":", 1)[0]
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


@dataclass(frozen=True, slots=True)
class Source:
    """Describe one public webcam source page from the source list."""

    id: str
    place: str
    city: str
    country: str
    url: str
    description: str | None = None

    @property
    def slug(self) -> str:
        """Return the deterministic country/city/place storage path."""
        return "/".join((slugify(self.country), slugify(self.city), slugify(self.place)))

    def to_dict(self) -> dict[str, str | None]:
        """Serialize source metadata for JSON output and persistence."""
        return {
            "id": self.id,
            "place": self.place,
            "city": self.city,
            "country": self.country,
            "url": self.url,
            "description": self.description,
            "slug": self.slug,
        }


def _cell_to_id(value: object) -> str:
    """Normalize legacy spreadsheet numeric identifiers without a decimal suffix."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _is_enabled(value: object | None) -> bool:
    """Interpret an optional source-list enabled flag with enabled as the safe default."""
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if not normalized:
        return True
    return normalized not in {"0", "false", "no", "off", "disabled"}


def _source_from_mapping(row: Mapping[str, object | None]) -> Source | None:
    """Build one enabled, non-YouTube source from canonical or legacy column names."""
    if not _is_enabled(row.get("enabled")):
        return None

    url_value = row.get("url") or row.get("Link")
    if not url_value:
        return None
    url = str(url_value).strip()
    if is_youtube_url(url):
        # YouTube, including YouTube Live, is explicitly out of scope.
        return None

    identifier = row.get("id") if row.get("id") is not None else row.get("No.")
    description_value = row.get("description") or row.get("Short description")
    return Source(
        id=_cell_to_id(identifier),
        place=str(row.get("place") or row.get("Place Name") or "unknown").strip(),
        city=str(row.get("city") or row.get("City") or "unknown").strip(),
        country=str(row.get("country") or row.get("Country") or "unknown").strip(),
        url=url,
        description=str(description_value).strip() if description_value else None,
    )


def _load_csv_sources(path: Path) -> list[Source]:
    """Load enabled sources from the maintainable UTF-8 CSV source list."""
    result: list[Source] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "place", "city", "country", "url"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV source list must contain id, place, city, country, and url columns"
            )
        for row in reader:
            source = _source_from_mapping(row)
            if source is not None:
                result.append(source)
    return result


def _load_xlsb_sources(path: Path) -> list[Source]:
    """Load enabled non-YouTube sources from the legacy XLSB dataset workbook."""
    try:
        from pyxlsb import open_workbook
    except ImportError as exc:
        raise RuntimeError("pyxlsb is required to read Place_Overview.xlsb") from exc

    with open_workbook(str(path)) as workbook:
        if not workbook.sheets:
            return []
        with workbook.get_sheet(workbook.sheets[0]) as sheet:
            rows = [[cell.v for cell in row] for row in sheet.rows()]

    result: list[Source] = []
    header_index: dict[str, int] | None = None
    for values in rows:
        candidate = {
            str(value).strip(): index
            for index, value in enumerate(values)
            if value is not None
        }
        if {"No.", "Place Name", "City", "Country", "Link"}.issubset(candidate):
            header_index = candidate
            continue
        if header_index is None:
            continue

        row = {
            name: values[index] if index < len(values) else None
            for name, index in header_index.items()
        }
        source = _source_from_mapping(row)
        if source is not None:
            result.append(source)
    return result


def load_sources(path: str | Path) -> list[Source]:
    """Load enabled public webcam sources from CSV or the legacy XLSB workbook."""
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_sources(source_path)
    if suffix == ".xlsb":
        return _load_xlsb_sources(source_path)
    raise ValueError(f"Unsupported source-list format: {source_path.suffix or '<none>'}")
