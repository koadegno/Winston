from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-") or "unknown"


def is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().split(":", 1)[0]
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    place: str
    city: str
    country: str
    url: str
    description: str | None = None

    @property
    def slug(self) -> str:
        return "/".join((slugify(self.country), slugify(self.city), slugify(self.place)))

    def to_dict(self) -> dict[str, str | None]:
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
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_sources(path: str | Path) -> list[Source]:
    try:
        from pyxlsb import open_workbook
    except ImportError as exc:
        raise RuntimeError("pyxlsb is required to read Place_Overview.xlsb") from exc

    result: list[Source] = []
    with open_workbook(str(path)) as workbook:
        if not workbook.sheets:
            return result
        with workbook.get_sheet(workbook.sheets[0]) as sheet:
            rows = [[cell.v for cell in row] for row in sheet.rows()]

    header_index: dict[str, int] | None = None
    for values in rows:
        candidate = {str(value).strip(): index for index, value in enumerate(values) if value is not None}
        if {"No.", "Place Name", "City", "Country", "Link"}.issubset(candidate):
            header_index = candidate
            continue
        if header_index is None:
            continue

        def get(name: str) -> object | None:
            index = header_index[name]
            return values[index] if index < len(values) else None

        url_value = get("Link")
        if not url_value:
            continue
        url = str(url_value).strip()
        if is_youtube_url(url):
            continue

        place = str(get("Place Name") or "unknown").strip()
        city = str(get("City") or "unknown").strip()
        country = str(get("Country") or "unknown").strip()
        description = None
        if "Short description" in header_index:
            raw_description = get("Short description")
            if raw_description:
                description = str(raw_description).strip()
        result.append(
            Source(
                id=_cell_to_id(get("No.")),
                place=place,
                city=city,
                country=country,
                url=url,
                description=description,
            )
        )

    return result
