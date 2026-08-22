"""Coordinate stream discovery, metadata persistence, retries, and camera recorders."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .discovery import discover_page
from .hls import HlsVariant, resolve_cameras
from .log import log
from .metadata import write_json_atomic
from .recorder import record_one_hour_slice
from .sources import Source


@dataclass(frozen=True, slots=True)
class Camera:
    """Bind a deterministic camera identifier to one resolved HLS stream."""

    id: str
    stream: HlsVariant


@dataclass(slots=True)
class SourceRediscovery:
    """Coalesce concurrent rediscovery requests for all cameras of one source."""

    source: Source
    http: Any | None
    browser: Any | None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _in_flight: asyncio.Task[list[Camera]] | None = field(default=None, init=False)

    async def _cameras(self) -> list[Camera]:
        """Return one shared in-flight source rediscovery result."""
        async with self._lock:
            task = self._in_flight
            if task is None:
                log(f"[source {self.source.id}] starting shared rediscovery")
                task = asyncio.create_task(
                    discover_source_cameras(
                        self.source,
                        client=self.http,
                        browser=self.browser,
                    )
                )
                self._in_flight = task
            else:
                log(f"[source {self.source.id}] joining shared rediscovery")

        try:
            return await task
        finally:
            # Clear only the task this caller actually awaited; a later cycle may already exist.
            async with self._lock:
                if self._in_flight is task and task.done():
                    self._in_flight = None

    async def camera(self, camera_id: str) -> Camera | None:
        """Return one camera from the shared rediscovery result by deterministic ID."""
        cameras = await self._cameras()
        replacement = next((camera for camera in cameras if camera.id == camera_id), None)
        if replacement is None:
            log(f"[source {self.source.id}/{camera_id}] rediscovery found no matching camera")
        else:
            log(
                f"[source {self.source.id}/{camera_id}] rediscovery selected "
                f"{replacement.stream.url}"
            )
        return replacement


def assign_camera_ids(streams: list[HlsVariant]) -> list[Camera]:
    """Assign stable-in-order camera IDs after sorting resolved stream URLs."""
    ordered = sorted(streams, key=lambda stream: stream.url)
    return [
        Camera(id=f"camera-{index:03d}", stream=stream)
        for index, stream in enumerate(ordered, start=1)
    ]


async def discover_source_cameras(
    source: Source,
    use_browser: bool = True,
    *,
    client: Any | None = None,
    browser: Any | None = None,
) -> list[Camera]:
    """Discover and resolve every logical HLS camera exposed by one source page."""
    log(f"[source {source.id}] discovery start: {source.place} - {source.url}")
    browser_observed_urls: set[str] = set()
    candidates = await discover_page(
        source.url,
        use_browser=use_browser,
        browser=browser,
        client=client,
        label=f"source {source.id}",
        browser_observed_urls=browser_observed_urls,
    )
    log(f"[source {source.id}] {len(candidates)} HLS candidate(s); resolving cameras")
    cameras = assign_camera_ids(
        await resolve_cameras(
            candidates,
            client=client,
            browser_observed_urls=browser_observed_urls,
        )
    )
    log(f"[source {source.id}] discovery complete: {len(cameras)} camera(s)")
    return cameras


def _source_dir(output_root: Path, source: Source) -> Path:
    """Return the root directory used to persist one source's recordings."""
    return output_root / Path(source.slug)


def write_source_metadata(
    output_root: Path,
    source: Source,
    cameras: list[Camera],
) -> None:
    """Persist source metadata and initialize metadata for all discovered cameras."""
    now = datetime.now(timezone.utc).isoformat()
    source_dir = _source_dir(output_root, source)
    write_json_atomic(
        source_dir / "source.json",
        {
            **source.to_dict(),
            "source_url": source.url,
            "discovered_at": now,
            "camera_count": len(cameras),
        },
    )
    for camera in cameras:
        write_camera_metadata(output_root, source, camera, first_seen=now)
    log(f"[source {source.id}] metadata written to {source_dir}")


def write_camera_metadata(
    output_root: Path,
    source: Source,
    camera: Camera,
    *,
    first_seen: str | None = None,
) -> None:
    """Persist current camera stream metadata while preserving its first-seen time."""
    camera_path = _source_dir(output_root, source) / "cameras" / camera.id / "camera.json"
    existing_first_seen = first_seen
    if camera_path.exists():
        try:
            existing_first_seen = (
                json.loads(camera_path.read_text(encoding="utf-8")).get("first_seen")
                or first_seen
            )
        except (OSError, ValueError, TypeError):
            # Corrupt metadata should be replaced rather than stopping a live recorder.
            existing_first_seen = first_seen

    now = datetime.now(timezone.utc).isoformat()
    write_json_atomic(
        camera_path,
        {
            "camera_id": camera.id,
            "source_id": source.id,
            "page_url": source.url,
            "stream_url": camera.stream.url,
            "resolution": list(camera.stream.resolution) if camera.stream.resolution else None,
            "bandwidth": camera.stream.bandwidth,
            "first_seen": existing_first_seen or now,
            "last_seen": now,
        },
    )


async def record_camera_loop(
    source: Source,
    camera_id: str,
    initial_stream: HlsVariant,
    output_root: Path,
    *,
    rediscovery: SourceRediscovery | None = None,
    client: Any | None = None,
    browser: Any | None = None,
    retry_seconds: float = 10.0,
    max_cycles: int | None = None,
) -> None:
    """Continuously record one camera and rediscover its HLS URL after failures."""
    stream = initial_stream
    cycles = 0
    coordinator = rediscovery or SourceRediscovery(source, http=client, browser=browser)
    log(f"[source {source.id}/{camera_id}] recorder started: {stream.url}")
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        camera = Camera(camera_id, stream)
        write_camera_metadata(output_root, source, camera)
        returncode = await record_one_hour_slice(
            stream.url,
            output_root,
            source.slug,
            camera_id,
        )
        if returncode == 0:
            log(f"[source {source.id}/{camera_id}] hour slice completed successfully")
            continue

        log(
            f"[source {source.id}/{camera_id}] FFmpeg failed with returncode={returncode}; "
            f"retrying discovery in {retry_seconds}s"
        )
        # Async sleep keeps every other source and camera recording during backoff.
        await asyncio.sleep(retry_seconds)
        replacement = await coordinator.camera(camera_id)
        if replacement is not None:
            stream = replacement.stream


async def record_source_forever(
    source: Source,
    output_root: Path,
    *,
    client: Any | None = None,
    browser: Any | None = None,
) -> None:
    """Discover one source and run all of its camera recorders concurrently."""
    log(f"[source {source.id}] recorder source task started: {source.place}")
    cameras = await discover_source_cameras(source, client=client, browser=browser)
    if not cameras:
        raise RuntimeError(f"No HLS stream found for {source.url}")

    write_source_metadata(output_root, source, cameras)
    rediscovery = SourceRediscovery(source, http=client, browser=browser)
    log(f"[source {source.id}] starting {len(cameras)} FFmpeg camera task(s)")
    # Camera loops are independent long-lived jobs and share one source rediscovery coordinator.
    await asyncio.gather(
        *(
            record_camera_loop(
                source,
                camera.id,
                camera.stream,
                output_root,
                rediscovery=rediscovery,
                client=client,
                browser=browser,
            )
            for camera in cameras
        )
    )
