"""Deterministic identity helpers for indexed visual candidates."""

from decimal import Decimal, ROUND_HALF_UP
import json
import uuid
from uuid import UUID

from winston.index.models import IndexedVisual

MICROSECONDS_PER_SECOND = Decimal(1_000_000)
VISUAL_IDENTITY_PREFIX = "winston:visual:v1:"


def timestamp_to_microseconds(timestamp_seconds: float | None) -> int | None:
    """Convert a non-negative finite timestamp to deterministic integer microseconds."""
    if timestamp_seconds is None:
        return None
    value = Decimal(str(timestamp_seconds)) * MICROSECONDS_PER_SECOND
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def visual_point_id(visual: IndexedVisual) -> UUID:
    """Return the deterministic UUIDv5 for one logical visual candidate."""
    material = {
        "asset_id": visual.asset_id,
        "dimension": visual.embedding_identity.dimension,
        "height": visual.region.height,
        "model_id": visual.embedding_identity.model_id,
        "preprocessing_version": visual.embedding_identity.preprocessing_version,
        "region_kind": visual.region_kind.value,
        "sample_kind": visual.sample_kind.value,
        "timestamp_us": timestamp_to_microseconds(visual.timestamp_seconds),
        "width": visual.region.width,
        "x": visual.region.x,
        "y": visual.region.y,
    }
    canonical = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{VISUAL_IDENTITY_PREFIX}{canonical}")
