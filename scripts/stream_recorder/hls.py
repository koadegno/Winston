from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urljoin

from .discovery import fetch_text


@dataclass(frozen=True, slots=True)
class HlsVariant:
    url: str
    bandwidth: int = 0
    resolution: tuple[int, int] | None = None

    @property
    def pixels(self) -> int:
        if not self.resolution:
            return 0
        return self.resolution[0] * self.resolution[1]


_ATTRIBUTE_RE = re.compile(r'([A-Z0-9-]+)=((?:"[^"]*")|[^,]*)')


def _parse_attributes(line: str) -> dict[str, str]:
    _, _, raw = line.partition(":")
    result: dict[str, str] = {}
    for match in _ATTRIBUTE_RE.finditer(raw):
        value = match.group(2).strip().strip('"')
        result[match.group(1)] = value
    return result


def parse_master_playlist(text: str, playlist_url: str) -> list[HlsVariant]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    variants: list[HlsVariant] = []
    for index, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF:"):
            continue
        attrs = _parse_attributes(line)
        next_index = index + 1
        while next_index < len(lines) and lines[next_index].startswith("#"):
            next_index += 1
        if next_index >= len(lines):
            continue

        resolution = None
        if "RESOLUTION" in attrs and "x" in attrs["RESOLUTION"].lower():
            width, height = attrs["RESOLUTION"].lower().split("x", 1)
            try:
                resolution = (int(width), int(height))
            except ValueError:
                resolution = None

        try:
            bandwidth = int(attrs.get("AVERAGE-BANDWIDTH") or attrs.get("BANDWIDTH") or 0)
        except ValueError:
            bandwidth = 0

        variants.append(
            HlsVariant(
                url=urljoin(playlist_url, lines[next_index]),
                bandwidth=bandwidth,
                resolution=resolution,
            )
        )
    return variants


def choose_best_stream(variants: list[HlsVariant]) -> HlsVariant:
    if not variants:
        raise ValueError("No HLS variants available")
    return max(variants, key=lambda item: (item.pixels, item.bandwidth))


def resolve_candidate(url: str) -> tuple[HlsVariant, set[str]]:
    text, final_url = fetch_text(url)
    variants = parse_master_playlist(text, final_url)
    if not variants:
        return HlsVariant(url=final_url), set()
    return choose_best_stream(variants), {variant.url for variant in variants}


def resolve_cameras(candidate_urls: set[str]) -> list[HlsVariant]:
    resolved: dict[str, tuple[HlsVariant, set[str]]] = {}
    referenced_variants: set[str] = set()
    for url in sorted(candidate_urls):
        try:
            best, variants = resolve_candidate(url)
        except Exception:
            continue
        resolved[url] = (best, variants)
        referenced_variants.update(variants)

    cameras: list[HlsVariant] = []
    seen_stream_urls: set[str] = set()
    for candidate_url, (best, _) in resolved.items():
        if candidate_url in referenced_variants:
            continue
        if best.url in seen_stream_urls:
            continue
        seen_stream_urls.add(best.url)
        cameras.append(best)
    return cameras
