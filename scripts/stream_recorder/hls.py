"""Resolve HLS master playlists into one highest-quality stream per camera."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import urljoin

from .discovery import fetch_text
from .log import log


@dataclass(frozen=True, slots=True)
class HlsVariant:
    """Describe one concrete HLS playlist variant."""

    url: str
    bandwidth: int = 0
    resolution: tuple[int, int] | None = None

    @property
    def pixels(self) -> int:
        """Return total pixel count for quality ordering."""
        if not self.resolution:
            return 0
        return self.resolution[0] * self.resolution[1]


_ATTRIBUTE_RE = re.compile(r'([A-Z0-9-]+)=((?:"[^"]*")|[^,]*)')


def _parse_attributes(line: str) -> dict[str, str]:
    """Parse attributes from an EXT-X-STREAM-INF line."""
    _, _, raw = line.partition(":")
    result: dict[str, str] = {}
    for match in _ATTRIBUTE_RE.finditer(raw):
        value = match.group(2).strip().strip('"')
        result[match.group(1)] = value
    return result


def parse_master_playlist(text: str, playlist_url: str) -> list[HlsVariant]:
    """Parse HLS master playlist variants and resolve relative URLs."""
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
    """Choose highest resolution, then highest bandwidth HLS variant."""
    if not variants:
        raise ValueError("No HLS variants available")
    return max(variants, key=lambda item: (item.pixels, item.bandwidth))


async def resolve_candidate(
    url: str,
    *,
    client: Any | None = None,
) -> tuple[HlsVariant, set[str]]:
    """Resolve one HLS candidate into its best concrete stream and known variants."""
    log(f"[hls] resolve {url}")
    try:
        text, final_url = await fetch_text(url, client=client)
    except Exception as exc:
        log(f"[hls] ERROR {url}: {type(exc).__name__}: {exc}")
        raise

    variants = parse_master_playlist(text, final_url)
    if not variants:
        log(f"[hls] media playlist {final_url}")
        return HlsVariant(url=final_url), set()

    best = choose_best_stream(variants)
    quality = (
        f"{best.resolution[0]}x{best.resolution[1]}"
        if best.resolution
        else "resolution unknown"
    )
    log(
        f"[hls] master {final_url}: {len(variants)} variant(s); "
        f"selected {quality}, bandwidth={best.bandwidth}, url={best.url}"
    )
    return best, {variant.url for variant in variants}


async def resolve_cameras(
    candidate_urls: set[str],
    *,
    client: Any | None = None,
) -> list[HlsVariant]:
    """Resolve all HLS candidates concurrently and deduplicate logical cameras."""
    ordered_urls = sorted(candidate_urls)
    log(f"[hls] resolving {len(ordered_urls)} candidate playlist(s) in parallel")

    async def resolve_one(url: str) -> tuple[HlsVariant, set[str]]:
        """Resolve one candidate while preserving monkeypatch-friendly optional clients."""
        if client is None:
            return await resolve_candidate(url)
        return await resolve_candidate(url, client=client)

    # Candidate playlists are independent network requests, so resolve them together.
    results = await asyncio.gather(
        *(resolve_one(url) for url in ordered_urls),
        return_exceptions=True,
    )

    resolved: dict[str, tuple[HlsVariant, set[str]]] = {}
    referenced_variants: set[str] = set()
    for url, result in zip(ordered_urls, results, strict=True):
        if isinstance(result, BaseException):
            log(f"[hls] candidate failed {url}: {type(result).__name__}: {result}")
            continue
        best, variants = result
        resolved[url] = (best, variants)
        referenced_variants.update(variants)

    cameras: list[HlsVariant] = []
    seen_stream_urls: set[str] = set()
    for candidate_url, (best, _) in resolved.items():
        # Ignore variants also observed by the browser when their master is present.
        if candidate_url in referenced_variants:
            continue
        if best.url in seen_stream_urls:
            continue
        seen_stream_urls.add(best.url)
        cameras.append(best)

    log(f"[hls] resolved {len(cameras)} logical camera(s)")
    return cameras
