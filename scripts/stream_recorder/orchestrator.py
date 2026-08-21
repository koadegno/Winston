from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

from .discovery import discover_page
from .hls import HlsVariant, resolve_cameras
from .metadata import write_json_atomic
from .recorder import record_one_hour_slice
from .sources import Source


@dataclass(frozen=True, slots=True)
class Camera:
    id: str
    stream: HlsVariant


def assign_camera_ids(streams: list[HlsVariant]) -> list[Camera]:
    ordered = sorted(streams, key=lambda stream: stream.url)
    return [Camera(id=f"camera-{index:03d}", stream=stream) for index, stream in enumerate(ordered, start=1)]


def discover_source_cameras(source: Source, use_browser_fallback: bool = True) -> list[Camera]:
    candidates = discover_page(source.url, use_browser_fallback=use_browser_fallback)
    return assign_camera_ids(resolve_cameras(candidates))


def _source_dir(output_root: Path, source: Source) -> Path:
    return output_root / Path(source.slug)


def write_source_metadata(output_root: Path, source: Source, cameras: list[Camera]) -> None:
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


def write_camera_metadata(
    output_root: Path,
    source: Source,
    camera: Camera,
    *,
    first_seen: str | None = None,
) -> None:
    camera_path = _source_dir(output_root, source) / "cameras" / camera.id / "camera.json"
    existing_first_seen = first_seen
    if camera_path.exists():
        try:
            import json

            existing_first_seen = json.loads(camera_path.read_text(encoding="utf-8")).get("first_seen") or first_seen
        except Exception:
            pass
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


def _rediscover_camera(source: Source, camera_id: str) -> Camera | None:
    cameras = discover_source_cameras(source)
    for camera in cameras:
        if camera.id == camera_id:
            return camera
    return None


def record_camera_loop(
    source: Source,
    camera_id: str,
    initial_stream: HlsVariant,
    output_root: Path,
    *,
    retry_seconds: float = 10.0,
    max_cycles: int | None = None,
) -> None:
    stream = initial_stream
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        camera = Camera(camera_id, stream)
        write_camera_metadata(output_root, source, camera)
        returncode = record_one_hour_slice(stream.url, output_root, source.slug, camera_id)
        if returncode == 0:
            continue

        time.sleep(retry_seconds)
        replacement = _rediscover_camera(source, camera_id)
        if replacement is not None:
            stream = replacement.stream


def record_source_forever(source: Source, output_root: Path) -> None:
    cameras = discover_source_cameras(source)
    if not cameras:
        raise RuntimeError(f"No HLS stream found for {source.url}")
    write_source_metadata(output_root, source, cameras)
    with ThreadPoolExecutor(max_workers=len(cameras), thread_name_prefix=f"source-{source.id}") as pool:
        futures = [
            pool.submit(record_camera_loop, source, camera.id, camera.stream, output_root)
            for camera in cameras
        ]
        for future in futures:
            future.result()
