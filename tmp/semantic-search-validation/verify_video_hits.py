"""Temporary real-video validation for Phase 1F semantic-search candidates."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from qdrant_client import AsyncQdrantClient, models

from winston.config import Settings
from winston.embeddings.base import RGBImage
from winston.embeddings.factory import create_embedder
from winston.index.models import VisualVector
from winston.search.pipeline import run_search
from winston.search.temporal import cosine_similarity

SOURCE_PATH = "videos/2026-08-22T14-00-00Z.mkv"
QUERIES = (
    "a person walking on the street",
    "a bicycle or cyclist",
    "people crossing the road",
    "a white car",
)


@dataclass(frozen=True, slots=True)
class ExtractedRgbFrame:
    """Minimal in-memory RGB frame accepted by Winston's embedder contract."""

    width: int
    height: int
    rgb24: bytes


@dataclass(frozen=True, slots=True)
class StoredRegionVector:
    """One stored Qdrant vector matched to the returned Winston region."""

    vector: VisualVector
    point_id: str


def _release_video(release_root: Path) -> Path:
    """Resolve the indexed source path inside the extracted release directory."""
    candidate = release_root / SOURCE_PATH
    if not candidate.is_file():
        raise RuntimeError(f"release video not found: {candidate}")
    return candidate


def _extract_crop(
    video: Path,
    *,
    timestamp_seconds: float,
    x: int,
    y: int,
    width: int,
    height: int,
    output: Path,
) -> None:
    """Decode the returned representative timestamp and exact indexed region from the release."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp_seconds:.6f}",
            "-i",
            str(video),
            "-vf",
            f"crop={width}:{height}:{x}:{y}",
            "-frames:v",
            "1",
            "-y",
            str(output),
        ],
        check=True,
    )


def _load_rgb(path: Path) -> ExtractedRgbFrame:
    """Load an extracted crop as deterministic packed RGB24 bytes."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return ExtractedRgbFrame(
            width=rgb.width,
            height=rgb.height,
            rgb24=rgb.tobytes(),
        )


def _dense_vector(record: models.Record, vector_name: str) -> VisualVector:
    """Extract one named dense Qdrant vector as Winston float32 storage."""
    vectors = record.vector
    if not isinstance(vectors, dict):
        raise RuntimeError(f"point {record.id} did not return named vectors")
    raw = vectors.get(vector_name)
    if not isinstance(raw, list):
        raise RuntimeError(f"point {record.id} did not return dense vector {vector_name!r}")
    vector = np.asarray(raw, dtype=np.float32)
    if vector.ndim != 1:
        raise RuntimeError(f"point {record.id} returned invalid vector shape {vector.shape}")
    return vector


async def _stored_region_vector(
    client: AsyncQdrantClient,
    *,
    settings: Settings,
    timestamp_us: int,
    region_kind: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> StoredRegionVector:
    """Find the exact stored region selected by Winston at one representative timestamp."""
    records, next_offset = await client.scroll(
        settings.qdrant.collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_path",
                    match=models.MatchValue(value=SOURCE_PATH),
                ),
                models.FieldCondition(
                    key="timestamp_us",
                    match=models.MatchValue(value=timestamp_us),
                ),
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=[settings.qdrant.vector_name],
    )
    if next_offset is not None:
        raise RuntimeError("unexpectedly more than 100 regions at one timestamp")

    for record in records:
        payload = record.payload
        if payload is None or payload.get("region_kind") != region_kind:
            continue
        region = payload.get("region")
        if not isinstance(region, dict):
            continue
        if (
            region.get("x") == x
            and region.get("y") == y
            and region.get("width") == width
            and region.get("height") == height
        ):
            return StoredRegionVector(
                vector=_dense_vector(record, settings.qdrant.vector_name),
                point_id=str(record.id),
            )
    raise RuntimeError(
        f"stored representative region not found at {timestamp_us}us: "
        f"{region_kind} x={x} y={y} width={width} height={height}"
    )


async def main() -> None:
    """Verify real CLI hits against freshly decoded and re-embedded release crops."""
    release_root = Path("/tmp/winston-release")
    output_root = Path("/tmp/winston-hit-crops")
    output_root.mkdir(parents=True, exist_ok=True)
    video = _release_video(release_root)
    settings = Settings()
    verifier = create_embedder(settings)
    client = AsyncQdrantClient(url=settings.qdrant.url)
    try:
        for query_index, query in enumerate(QUERIES, start=1):
            results = await run_search(query, limit=3, settings=settings)
            result = next((item for item in results if item.source_path == SOURCE_PATH), None)
            if result is None:
                raise RuntimeError(f"query did not return {SOURCE_PATH}: {query!r}")
            timestamp = result.representative_timestamp_seconds
            if timestamp is None:
                raise RuntimeError(f"video result has no representative timestamp: {query!r}")

            region = result.region
            crop_path = output_root / f"hit-{query_index}.png"
            _extract_crop(
                video,
                timestamp_seconds=timestamp,
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
                output=crop_path,
            )
            fresh_image: RGBImage = _load_rgb(crop_path)
            fresh_batch = await verifier.embed_images([fresh_image])
            query_batch = await verifier.embed_texts([query])
            fresh_vector = fresh_batch.vectors[0]
            query_vector = query_batch.vectors[0]

            stored = await _stored_region_vector(
                client,
                settings=settings,
                timestamp_us=round(timestamp * 1_000_000),
                region_kind=result.region_kind.value,
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
            )
            stored_vs_fresh = cosine_similarity(stored.vector, fresh_vector)
            stored_query = cosine_similarity(query_vector, stored.vector)
            fresh_query = cosine_similarity(query_vector, fresh_vector)

            print(f"QUERY={query}")
            print(f"SOURCE={result.source_path}")
            print(f"TIMESTAMP={timestamp:.6f}")
            print(
                "REGION="
                f"{result.region_kind.value} x={region.x} y={region.y} "
                f"width={region.width} height={region.height} scale={region.scale}"
            )
            print(f"POINT={stored.point_id}")
            print(f"SEARCH_RAW_SCORE={result.raw_score:.6f}")
            print(f"QUERY_VS_STORED={stored_query:.6f}")
            print(f"QUERY_VS_FRESH_CROP={fresh_query:.6f}")
            print(f"STORED_VS_FRESH_CROP={stored_vs_fresh:.6f}")
            print()

            if abs(stored_query - result.raw_score) > 1e-5:
                raise RuntimeError(
                    f"search raw score does not match stored-vector cosine for {query!r}"
                )
            if stored_vs_fresh < 0.95:
                raise RuntimeError(
                    f"release crop does not reproduce stored vector for {query!r}: "
                    f"cosine={stored_vs_fresh:.6f}"
                )
            if abs(fresh_query - result.raw_score) > 0.05:
                raise RuntimeError(
                    f"fresh release crop changed semantic score too much for {query!r}: "
                    f"search={result.raw_score:.6f}, fresh={fresh_query:.6f}"
                )
    finally:
        await client.close()
        await verifier.close()


if __name__ == "__main__":
    asyncio.run(main())
