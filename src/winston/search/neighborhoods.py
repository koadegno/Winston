"""Semantic video-neighborhood grouping kept separate from local refinement budgets."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

from winston.index.models import ASSET_ID_PATTERN
from winston.ingest.models import MediaType
from winston.search.temporal import (
    MICROSECONDS_PER_SECOND,
    TemporalObservation,
    TemporalWindow,
)


@dataclass(frozen=True, slots=True)
class LogicalTemporalNeighborhood:
    """One connected coarse semantic opportunity before any runtime capping is applied."""

    asset_id: str
    source_path: str
    start_timestamp_us: int
    end_timestamp_us: int
    anchor: TemporalObservation

    def __post_init__(self) -> None:
        """Require one valid stream interval whose strongest coarse seed lies inside it."""
        if ASSET_ID_PATTERN.fullmatch(self.asset_id) is None:
            raise ValueError("asset_id must be a lowercase SHA-256 hexadecimal string")
        if not self.source_path:
            raise ValueError("source_path must not be empty")
        if self.start_timestamp_us < 0 or self.end_timestamp_us < self.start_timestamp_us:
            raise ValueError("logical neighborhood timestamps must be non-negative and ordered")

        visual = self.anchor.match.visual
        if visual.media_type is not MediaType.VIDEO:
            raise ValueError("logical neighborhood anchor must be a video observation")
        if visual.asset_id != self.asset_id or visual.source_path != self.source_path:
            raise ValueError("logical neighborhood anchor must belong to the same video stream")
        if not self.start_timestamp_us <= self.anchor.timestamp_us <= self.end_timestamp_us:
            raise ValueError("logical neighborhood anchor must lie inside the neighborhood")


def build_logical_temporal_neighborhoods(
    seeds: Sequence[TemporalObservation],
    *,
    context_seconds: float,
) -> tuple[LogicalTemporalNeighborhood, ...]:
    """Merge touching coarse seed contexts without letting runtime caps change diversity.

    The returned count is semantic: one connected transitive chain remains one logical
    opportunity no matter how long it becomes. Runtime work is bounded later by
    ``bounded_refinement_window()`` around the strongest coarse seed in that chain.
    """
    context_us = _positive_duration_us(context_seconds, name="context_seconds")

    expanded: list[tuple[str, str, int, int, TemporalObservation]] = []
    for seed in seeds:
        visual = seed.match.visual
        expanded.append(
            (
                visual.asset_id,
                visual.source_path,
                max(0, seed.timestamp_us - context_us),
                seed.timestamp_us + context_us,
                seed,
            )
        )
    expanded.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    if not expanded:
        return ()

    neighborhoods: list[LogicalTemporalNeighborhood] = []
    asset_id, source_path, start_us, end_us, anchor = expanded[0]
    for (
        candidate_asset_id,
        candidate_source_path,
        candidate_start_us,
        candidate_end_us,
        candidate_anchor,
    ) in expanded[1:]:
        same_stream = (
            candidate_asset_id == asset_id and candidate_source_path == source_path
        )
        if same_stream and candidate_start_us <= end_us:
            end_us = max(end_us, candidate_end_us)
            if _anchor_order_key(candidate_anchor) < _anchor_order_key(anchor):
                anchor = candidate_anchor
            continue

        neighborhoods.append(
            LogicalTemporalNeighborhood(
                asset_id=asset_id,
                source_path=source_path,
                start_timestamp_us=start_us,
                end_timestamp_us=end_us,
                anchor=anchor,
            )
        )
        asset_id = candidate_asset_id
        source_path = candidate_source_path
        start_us = candidate_start_us
        end_us = candidate_end_us
        anchor = candidate_anchor

    neighborhoods.append(
        LogicalTemporalNeighborhood(
            asset_id=asset_id,
            source_path=source_path,
            start_timestamp_us=start_us,
            end_timestamp_us=end_us,
            anchor=anchor,
        )
    )
    return tuple(neighborhoods)


def bounded_refinement_window(
    neighborhood: LogicalTemporalNeighborhood,
    *,
    max_window_seconds: float,
) -> TemporalWindow:
    """Choose at most one bounded local scan around a logical neighborhood's best seed.

    Short neighborhoods retain their complete semantic union. If a transitive chain is
    longer than the configured runtime budget, Winston scans one maximum-sized interval
    centered on the strongest ANN seed as far as the logical boundaries allow. Thus a
    single semantic opportunity can never multiply into several runtime chunks or cause
    the whole long union to be rescanned piece by piece.
    """
    max_window_us = _positive_duration_us(
        max_window_seconds,
        name="max_window_seconds",
    )
    if neighborhood.end_timestamp_us - neighborhood.start_timestamp_us <= max_window_us:
        return TemporalWindow(
            asset_id=neighborhood.asset_id,
            source_path=neighborhood.source_path,
            start_timestamp_us=neighborhood.start_timestamp_us,
            end_timestamp_us=neighborhood.end_timestamp_us,
        )

    half_window_us = max_window_us // 2
    start_us = max(
        neighborhood.start_timestamp_us,
        neighborhood.anchor.timestamp_us - half_window_us,
    )
    end_us = start_us + max_window_us
    if end_us > neighborhood.end_timestamp_us:
        end_us = neighborhood.end_timestamp_us
        start_us = max(
            neighborhood.start_timestamp_us,
            end_us - max_window_us,
        )

    return TemporalWindow(
        asset_id=neighborhood.asset_id,
        source_path=neighborhood.source_path,
        start_timestamp_us=start_us,
        end_timestamp_us=end_us,
    )


def _positive_duration_us(value: float, *, name: str) -> int:
    """Convert one finite positive duration to a representable integer microsecond count."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    value_us = round(value * MICROSECONDS_PER_SECOND)
    if value_us <= 0:
        raise ValueError(f"{name} is too small to represent in microseconds")
    return value_us


def _anchor_order_key(
    observation: TemporalObservation,
) -> tuple[float, int, str, str, int, int, int, int, float]:
    """Prefer stronger coarse score, then stable earliest visual provenance for exact ties."""
    match = observation.match
    visual = match.visual
    region = visual.region
    return (
        -match.score,
        observation.timestamp_us,
        visual.source_path,
        visual.region_kind.value,
        region.x,
        region.y,
        region.width,
        region.height,
        region.scale,
    )
