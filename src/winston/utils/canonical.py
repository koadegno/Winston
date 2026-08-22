"""Canonical serialization helpers shared by deterministic Winston identities."""

from collections.abc import Mapping
from decimal import Decimal, ROUND_HALF_UP
import json


type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]

MICROSECONDS_PER_SECOND = Decimal(1_000_000)


def canonical_json(material: Mapping[str, JsonValue]) -> str:
    """Serialize JSON material deterministically for hashes and UUID names."""
    return json.dumps(
        dict(material),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def timestamp_to_microseconds(timestamp_seconds: float | None) -> int | None:
    """Convert seconds to deterministic integer microseconds using half-up rounding."""
    if timestamp_seconds is None:
        return None
    value = Decimal(str(timestamp_seconds)) * MICROSECONDS_PER_SECOND
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
