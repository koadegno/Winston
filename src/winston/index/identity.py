"""Deterministic identity helpers for indexed visual candidates."""

import uuid
from uuid import UUID

from winston.index.models import IndexedVisual
from winston.utils.canonical import canonical_json, timestamp_to_microseconds

VISUAL_IDENTITY_PREFIX = "winston:visual:v1:"


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
    canonical = canonical_json(material)
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{VISUAL_IDENTITY_PREFIX}{canonical}")


__all__ = ["timestamp_to_microseconds", "visual_point_id"]
