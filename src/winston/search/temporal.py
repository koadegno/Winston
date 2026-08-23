"""Pure temporal-search helpers shared by coarse discovery and video refinement."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from winston.index.models import ASSET_ID_PATTERN, ScoredVisual, VisualVector
from winston.ingest.models import MediaType

MICROSECONDS_PER_SECOND = 1_000_000


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """Strongest visual match retained for one sampled video timestamp."""

    timestamp_us: int
    match: ScoredVisual

    def __post_init__(self) -> None:
        """Keep an observation tied exactly to the keyframe represented by its visual."""
        if (
            isinstance(self.timestamp_us, bool)
            or not isinstance(self.timestamp_us, int)
            or self.timestamp_us < 0
        ):
            raise ValueError("timestamp_us must be a non-negative integer")
        visual = self.match.visual
        if visual.media_type is not MediaType.VIDEO or visual.timestamp_us is None:
            raise ValueError("temporal observations require a timestamped video visual")
        if visual.timestamp_us != self.timestamp_us:
            raise ValueError("timestamp_us must match the representative visual timestamp")


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    """One merged video interval whose complete indexed neighborhood must be retrieved."""

    asset_id: str
    source_path: str
    start_timestamp_us: int
    end_timestamp_us: int

    def __post_init__(self) -> None:
        """Reject malformed candidate windows before they become Qdrant filters."""
        if ASSET_ID_PATTERN.fullmatch(self.asset_id) is None:
            raise ValueError("asset_id must be a lowercase SHA-256 hexadecimal string")
        if not self.source_path:
            raise ValueError("source_path must not be empty")
        if self.start_timestamp_us < 0 or self.end_timestamp_us < 0:
            raise ValueError("temporal window timestamps must be non-negative")
        if self.start_timestamp_us > self.end_timestamp_us:
            raise ValueError("temporal window start must be <= end")


def cosine_similarity(query: VisualVector, candidate: VisualVector) -> float:
    """Return exact cosine similarity for two finite non-zero dense visual vectors.

    This deliberately returns the mathematical cosine value unchanged.  A value such
    as ``0.61`` stays ``0.61``; it is not clamped or presented as a confidence
    percentage.  Qdrant ANN finds promising neighborhoods, while this function is the
    exact local score used after every vector in a candidate window has been fetched.
    """
    if query.ndim != 1 or candidate.ndim != 1:
        raise ValueError("cosine vectors must be one-dimensional")
    if query.shape[0] != candidate.shape[0]:
        raise ValueError("cosine vectors must have the same dimension")
    if query.dtype != np.float32 or candidate.dtype != np.float32:
        raise ValueError("cosine vectors must use float32 values")
    if not bool(np.isfinite(query).all()) or not bool(np.isfinite(candidate).all()):
        raise ValueError("cosine vectors must contain only finite values")

    # Accumulate in float64 so the local rerank is numerically stable even though the
    # persisted/query vectors intentionally remain float32.
    query64 = query.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    query_norm = float(np.linalg.norm(query64))
    candidate_norm = float(np.linalg.norm(candidate64))
    if query_norm == 0.0 or candidate_norm == 0.0:
        raise ValueError("cosine vectors must be non-zero")

    score = float(np.dot(query64, candidate64) / (query_norm * candidate_norm))
    if not math.isfinite(score):
        raise ValueError("cosine similarity must be finite")
    return score


def collapse_best_by_timestamp(
    matches: Sequence[ScoredVisual],
) -> tuple[TemporalObservation, ...]:
    """Collapse many regions of one keyframe to its strongest semantic match.

    Example: if the full frame and three tiles at 12s score ``0.27``, ``0.31``,
    ``0.61`` and ``0.29``, the timestamp contributes exactly ``0.61`` and keeps the
    region that produced that score.  This prevents a tiled keyframe from getting more
    temporal votes merely because it generated more indexed regions.
    """
    best: dict[tuple[str, int], ScoredVisual] = {}
    for match in matches:
        visual = match.visual
        timestamp_us = visual.timestamp_us
        if visual.media_type is not MediaType.VIDEO or timestamp_us is None:
            raise ValueError("timestamp collapse accepts only timestamped video matches")
        key = (visual.asset_id, timestamp_us)
        current = best.get(key)
        if current is None or _match_order_key(match) < _match_order_key(current):
            best[key] = match

    return tuple(
        TemporalObservation(timestamp_us=timestamp_us, match=best[(asset_id, timestamp_us)])
        for asset_id, timestamp_us in sorted(best)
    )


def build_temporal_windows(
    seeds: Sequence[TemporalObservation],
    *,
    context_seconds: float,
) -> tuple[TemporalWindow, ...]:
    """Expand coarse seeds by context and merge touching windows within each asset.

    With a 15-second context, seeds at ``100s``, ``108s`` and ``310s`` first become
    ``85..115``, ``93..123`` and ``295..325``.  The first two overlap, so Winston
    retrieves two complete neighborhoods: ``85..123`` and ``295..325``.

    ``context_seconds`` is therefore a semantic parameter: changing it can change
    which sampled moments are grouped into one passage.  In contrast, ANN candidate
    count and Qdrant timeline page size primarily bound discovery/runtime cost.
    """
    if not math.isfinite(context_seconds) or context_seconds <= 0.0:
        raise ValueError("context_seconds must be finite and positive")
    context_us = round(context_seconds * MICROSECONDS_PER_SECOND)
    if context_us <= 0:
        raise ValueError("context_seconds is too small to represent in microseconds")

    expanded: list[TemporalWindow] = []
    for seed in seeds:
        visual = seed.match.visual
        expanded.append(
            TemporalWindow(
                asset_id=visual.asset_id,
                source_path=visual.source_path,
                start_timestamp_us=max(0, seed.timestamp_us - context_us),
                end_timestamp_us=seed.timestamp_us + context_us,
            )
        )

    expanded.sort(
        key=lambda window: (
            window.asset_id,
            window.source_path,
            window.start_timestamp_us,
            window.end_timestamp_us,
        )
    )
    if not expanded:
        return ()

    merged: list[TemporalWindow] = []
    current = expanded[0]
    for candidate in expanded[1:]:
        same_stream = (
            candidate.asset_id == current.asset_id
            and candidate.source_path == current.source_path
        )
        if same_stream and candidate.start_timestamp_us <= current.end_timestamp_us:
            current = TemporalWindow(
                asset_id=current.asset_id,
                source_path=current.source_path,
                start_timestamp_us=current.start_timestamp_us,
                end_timestamp_us=max(current.end_timestamp_us, candidate.end_timestamp_us),
            )
            continue
        merged.append(current)
        current = candidate
    merged.append(current)
    return tuple(merged)


def _match_order_key(match: ScoredVisual) -> tuple[float, str, str, int, int, int, int]:
    """Return a stable key that prefers higher score then deterministic region provenance."""
    visual = match.visual
    region = visual.region
    return (
        -match.score,
        visual.source_path,
        visual.region_kind.value,
        region.x,
        region.y,
        region.width,
        region.height,
    )
