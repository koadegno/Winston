"""Resolve HLS master playlists into one highest-quality stream per camera."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse

import httpx

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
_CHUNKLIST_RE = re.compile(r"^chunklist(?:_[^.]+)?\.m3u8$", re.IGNORECASE)
_OUTPUT_RE = re.compile(r"^(?P<base>.+?)_output_\d+\.m3u8$", re.IGNORECASE)
_TRACKS_DIR_RE = re.compile(r"^tracks-v\d+$", re.IGNORECASE)
_UUID_PLAYLIST_RE = re.compile(
    r"^(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.m3u8$",
    re.IGNORECASE,
)
_UUID_IN_PATH_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_CDN_MIRROR_PREFIX_RE = re.compile(r"^cdn\d+\.", re.IGNORECASE)
_GENERIC_PLAYLIST_NAMES = {
    "index.m3u8",
    "master.m3u8",
    "playlist.m3u8",
    "playlists.m3u8",
    "stream.m3u8",
    "chunks.m3u8",
}
_EPHEMERAL_QUERY_KEYS = {
    "auth",
    "expires",
    "hdnts",
    "policy",
    "schash",
    "scendtime",
    "session",
    "sig",
    "signature",
    "token",
}
_EPHEMERAL_QUERY_PREFIXES = (
    "_hls_",
    "wowzatoken",
    "x-amz-",
)


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


def _is_ephemeral_query_key(key: str) -> bool:
    """Return whether a query key represents a short-lived HLS delivery token."""
    lowered = key.lower()
    return lowered in _EPHEMERAL_QUERY_KEYS or any(
        lowered.startswith(prefix) for prefix in _EPHEMERAL_QUERY_PREFIXES
    )


def _stable_query_string(url: str) -> str:
    """Keep only query parameters that can identify a camera rather than one session."""
    pairs = [
        (key, value)
        for key, value in parse_qsl(urlparse(url).query, keep_blank_values=True)
        if not _is_ephemeral_query_key(key)
    ]
    return urlencode(sorted(pairs))


def _normalized_origin(url: str, family_path: str) -> str:
    """Normalize default ports and equivalent numbered CDN mirror hostnames."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    # CDN01/CDN02 style mirrors serve the same stream identity from equivalent hosts.
    # Only collapse them when the path contains a strong UUID identity to avoid merging
    # unrelated generic /live/playlist.m3u8 cameras from different servers.
    if _UUID_IN_PATH_RE.search(family_path):
        host = _CDN_MIRROR_PREFIX_RE.sub("", host)

    port = parsed.port
    if port and not (
        (parsed.scheme.lower() == "https" and port == 443)
        or (parsed.scheme.lower() == "http" and port == 80)
    ):
        host = f"{host}:{port}"
    return f"{parsed.scheme.lower()}://{host}"


def _camera_family_path(url: str) -> str:
    """Derive a stable path shared by master/media playlists of one physical camera."""
    path = urlparse(url).path
    parts = [part for part in path.split("/") if part]
    if not parts:
        return "/"

    filename = parts[-1]
    lowered = filename.lower()

    # Wowza/Nimble-style master and chunklist filenames live in the same stream directory.
    if _CHUNKLIST_RE.match(filename) or lowered in _GENERIC_PLAYLIST_NAMES:
        return "/" + "/".join(parts[:-1]) + "/"

    # LL-HLS players commonly expose video1_stream.m3u8 beside index.m3u8.
    if re.fullmatch(r"video\d*_stream\.m3u8", lowered):
        return "/" + "/".join(parts[:-1]) + "/"

    # HLS masters can reference tracks-v1/mono.ts.m3u8 below the camera directory.
    if len(parts) >= 2 and _TRACKS_DIR_RE.match(parts[-2]) and lowered.endswith(".m3u8"):
        return "/" + "/".join(parts[:-2]) + "/"

    output_match = _OUTPUT_RE.match(filename)
    if output_match:
        parts[-1] = output_match.group("base")
        return "/" + "/".join(parts)

    uuid_match = _UUID_PLAYLIST_RE.match(filename)
    if uuid_match:
        parts[-1] = uuid_match.group("uuid")
        return "/" + "/".join(parts)

    # Preserve arbitrary filenames such as front.m3u8/back.m3u8 so two cameras in the
    # same directory are never merged merely because they share a parent path.
    return path


def _camera_family_key(url: str) -> str:
    """Return a stable logical-camera key for one observed HLS candidate URL."""
    family_path = _camera_family_path(url)
    origin = _normalized_origin(url, family_path)
    stable_query = _stable_query_string(url)
    if stable_query:
        return f"{origin}{family_path}?{stable_query}"
    return f"{origin}{family_path}"


def _candidate_priority(url: str) -> tuple[int, int, int, str]:
    """Prefer stable master playlists over session-scoped media/chunk playlists."""
    parsed = urlparse(url)
    filename = parsed.path.rsplit("/", 1)[-1]
    lowered = filename.lower()

    if lowered in {"index.m3u8", "master.m3u8", "playlist.m3u8", "playlists.m3u8"}:
        rank = 0
    elif _UUID_PLAYLIST_RE.match(filename):
        rank = 0
    elif (
        _CHUNKLIST_RE.match(filename)
        or _OUTPUT_RE.match(filename)
        or re.fullmatch(r"video\d*_stream\.m3u8", lowered)
        or ("/tracks-v" in parsed.path.lower())
    ):
        rank = 2
    else:
        rank = 1

    # Stable URLs without volatile query strings are preferred when two mirrors are equal.
    return (rank, bool(parsed.query), len(url), url)


def _group_camera_candidates(candidate_urls: set[str]) -> list[list[str]]:
    """Group observed HLS URLs into deterministic physical-camera candidate families."""
    groups: dict[str, list[str]] = {}
    for url in sorted(candidate_urls):
        groups.setdefault(_camera_family_key(url), []).append(url)
    ordered_groups = [sorted(group, key=_candidate_priority) for group in groups.values()]
    ordered_groups.sort(key=lambda group: group[0])
    if len(ordered_groups) != len(candidate_urls):
        log(
            f"[hls] grouped {len(candidate_urls)} observed playlist(s) into "
            f"{len(ordered_groups)} camera family candidate(s)"
        )
    return ordered_groups


def _select_camera_candidates(candidate_urls: set[str]) -> list[str]:
    """Select one preferred HLS URL per inferred physical-camera family."""
    return [group[0] for group in _group_camera_candidates(candidate_urls)]


def _is_certificate_verification_error(exc: BaseException) -> bool:
    """Return whether an HTTP failure is specifically a TLS certificate verification error."""
    if not isinstance(exc, httpx.ConnectError):
        return False
    current: BaseException | None = exc
    while current is not None:
        text = str(current).upper()
        if "CERTIFICATE_VERIFY_FAILED" in text or "CERTIFICATE VERIFY FAILED" in text:
            return True
        current = current.__cause__
    return False


async def _fetch_hls_text(url: str, *, client: Any | None) -> tuple[str, str]:
    """Fetch a discovered HLS playlist, retrying only broken-certificate endpoints insecurely."""
    try:
        return await fetch_text(url, client=client)
    except Exception as exc:
        if not _is_certificate_verification_error(exc):
            raise

    # This is intentionally scoped to a URL already observed as an HLS playlist. Normal source
    # pages and all other HTTP traffic keep certificate verification enabled.
    log(f"[hls] WARN TLS certificate verification failed; retrying HLS only: {url}")
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as insecure_client:
        return await fetch_text(url, client=insecure_client)


async def resolve_candidate(
    url: str,
    *,
    client: Any | None = None,
) -> tuple[HlsVariant, set[str]]:
    """Resolve one HLS candidate into its best concrete stream and known variants."""
    log(f"[hls] resolve {url}")
    try:
        text, final_url = await _fetch_hls_text(url, client=client)
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
    browser_observed_urls: set[str] | None = None,
) -> list[HlsVariant]:
    """Resolve each camera family, preserving only failed URLs observed in live browser traffic."""
    groups = _group_camera_candidates(candidate_urls)
    observed = browser_observed_urls or set()
    log(f"[hls] resolving {len(groups)} camera candidate playlist(s) in parallel")

    async def resolve_url(url: str) -> tuple[HlsVariant, set[str]]:
        """Resolve one candidate while preserving monkeypatch-friendly optional clients."""
        if client is None:
            return await resolve_candidate(url)
        return await resolve_candidate(url, client=client)

    async def resolve_family(group: list[str]) -> HlsVariant | None:
        """Resolve one camera family and use a browser-observed URL as a transient fallback."""
        preferred = group[0]
        retry_urls = [preferred]
        retry_urls.extend(
            url
            for url in group
            if url != preferred and url in observed
        )

        for url in retry_urls:
            try:
                best, _ = await resolve_url(url)
            except Exception as exc:
                log(f"[hls] candidate failed {url}: {type(exc).__name__}: {exc}")
                continue
            return best

        browser_fallbacks = [url for url in group if url in observed]
        if browser_fallbacks:
            fallback = min(browser_fallbacks, key=_candidate_priority)
            # A request seen in the rendered player's network is stronger evidence than a stale
            # HTML string. Keep it so FFmpeg can attempt the stream and normal rediscovery can
            # recover when an upstream playlist is temporarily 404/unavailable.
            log(
                f"[hls] WARN probe unavailable; keeping browser-observed camera candidate: "
                f"{fallback}"
            )
            return HlsVariant(url=fallback)
        return None

    # Camera families are independent network operations, so resolve all families concurrently.
    results = await asyncio.gather(*(resolve_family(group) for group in groups))

    cameras: list[HlsVariant] = []
    seen_stream_urls: set[str] = set()
    for result in results:
        if result is None or result.url in seen_stream_urls:
            continue
        seen_stream_urls.add(result.url)
        cameras.append(result)

    log(f"[hls] resolved {len(cameras)} logical camera(s)")
    return cameras
