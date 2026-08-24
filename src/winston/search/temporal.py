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


class TemporalObservationAccumulator:
    """Retain only the strongest scored region for each sampled video timestamp.

    Timeline retrieval can stream several full/tile vectors for every keyframe. Keeping
    all of them until the window ends defeats Qdrant pagination because every dense
    vector stays live in Winston. This accumulator applies the same deterministic
    per-timestamp collapse online, so losing region vectors can be released as soon as
    the next stronger region is known.
    """

    def __init__(self) -> None:
        self._best: dict[tuple[str, int], ScoredVisual] = {}

    def add(self, match: ScoredVisual) -> None:
        """Consume one scored video region and retain it only if it wins its timestamp."""
        visual = match.visual
        timestamp_us = visual.timestamp_us
        if visual.media_type is not MediaType.VIDEO or timestamp_us is None:
            raise ValueError("timestamp collapse accepts only timestamped video matches")
        key = (visual.asset_id, timestamp_us)
        current = self._best.get(key)
        if current is None or _match_order_key(match) < _match_order_key(current):
            self._best[key] = match

    def observations(self) -> tuple[TemporalObservation, ...]:
        """Return retained winners in stable asset/timestamp order."""
        return tuple(
            TemporalObservation(
                timestamp_us=timestamp_us,
                match=self._best[(asset_id, timestamp_us)],
            )
            for asset_id, timestamp_us in sorted(self._best)
        )


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


@dataclass(frozen=True, slots=True)
class SelectedPassage:
    """One sampled video interval selected from a locally normalized semantic signal."""

    start_timestamp_us: int
    end_timestamp_us: int
    representative: ScoredVisual

    def __post_init__(self) -> None:
        """Require the representative keyframe to lie inside the sampled passage boundaries."""
        timestamp_us = self.representative.visual.timestamp_us
        if timestamp_us is None:
            raise ValueError("selected passage representative must be a timestamped video visual")
        if self.start_timestamp_us < 0 or self.end_timestamp_us < self.start_timestamp_us:
            raise ValueError("selected passage timestamps must be non-negative and ordered")
        if not self.start_timestamp_us <= timestamp_us <= self.end_timestamp_us:
            raise ValueError("selected passage representative must lie inside the passage")


def cosine_similarity(query: VisualVector, candidate: VisualVector) -> float:
    """Return exact cosine similarity for two finite non-zero dense visual vectors.

    This deliberately returns the mathematical cosine value unchanged. A value such
    as ``0.61`` stays ``0.61``; it is not clamped or presented as a confidence
    percentage. Qdrant ANN finds promising neighborhoods, while this function is the
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
    region that produced that score. This prevents a tiled keyframe from getting more
    temporal votes merely because it generated more indexed regions.
    """
    accumulator = TemporalObservationAccumulator()
    for match in matches:
        accumulator.add(match)
    return accumulator.observations()


def build_temporal_windows(
    seeds: Sequence[TemporalObservation],
    *,
    context_seconds: float,
    max_window_seconds: float,
) -> tuple[TemporalWindow, ...]:
    """Expand seeds, merge semantic neighborhoods, then split them at a hard duration bound.

    With a 15-second context, seeds at ``100s``, ``108s`` and ``310s`` first become
    ``85..115``, ``93..123`` and ``295..325``. The first two overlap, so their union is
    ``85..123``. A long transitive chain is still merged logically, but its union is
    partitioned into contiguous windows no longer than ``max_window_seconds`` before
    Qdrant refinement. Integer-microsecond coverage is preserved without overlap or gaps.

    ``context_seconds`` is a semantic parameter because it changes which neighborhoods
    touch. ``max_window_seconds`` is a runtime/resource hard bound: lowering it can split
    one semantic neighborhood into several independently refined passages, but it never
    silently expands the amount of video scanned by one refinement request.
    """
    if not math.isfinite(context_seconds) or context_seconds <= 0.0:
        raise ValueError("context_seconds must be finite and positive")
    if not math.isfinite(max_window_seconds) or max_window_seconds <= 0.0:
        raise ValueError("max_window_seconds must be finite and positive")

    context_us = round(context_seconds * MICROSECONDS_PER_SECOND)
    max_window_us = round(max_window_seconds * MICROSECONDS_PER_SECOND)
    if context_us <= 0:
        raise ValueError("context_seconds is too small to represent in microseconds")
    if max_window_us <= 0:
        raise ValueError("max_window_seconds is too small to represent in microseconds")
    if max_window_us < 2 * context_us:
        raise ValueError("max_window_seconds must be >= 2 * context_seconds")

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

    bounded: list[TemporalWindow] = []
    for window in merged:
        start_timestamp_us = window.start_timestamp_us
        while window.end_timestamp_us - start_timestamp_us > max_window_us:
            end_timestamp_us = start_timestamp_us + max_window_us
            bounded.append(
                TemporalWindow(
                    asset_id=window.asset_id,
                    source_path=window.source_path,
                    start_timestamp_us=start_timestamp_us,
                    end_timestamp_us=end_timestamp_us,
                )
            )
            start_timestamp_us = end_timestamp_us + 1
        bounded.append(
            TemporalWindow(
                asset_id=window.asset_id,
                source_path=window.source_path,
                start_timestamp_us=start_timestamp_us,
                end_timestamp_us=window.end_timestamp_us,
            )
        )
    return tuple(bounded)


def centered_moving_average(
    values: Sequence[float],
    *,
    width: int,
) -> tuple[float, ...]:
    """Smooth a sequence with a centered odd-width window and partial edge windows.

    For width 3, ``[1, 2, 3, 4, 5]`` becomes ``[1.5, 2, 3, 4, 4.5]``: an interior
    sample uses previous/current/next, while the first and last samples use only
    neighbors that actually exist. Smoothing only shapes temporal interval selection;
    it never replaces the representative raw cosine used for global result ranking.
    """
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0 or width % 2 == 0:
        raise ValueError("moving-average width must be a positive odd integer")
    numeric = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("moving-average values must all be finite")
    if not numeric:
        return ()

    prefix = [0.0]
    for value in numeric:
        prefix.append(prefix[-1] + value)

    radius = width // 2
    smoothed: list[float] = []
    for index in range(len(numeric)):
        start = max(0, index - radius)
        end = min(len(numeric), index + radius + 1)
        smoothed.append((prefix[end] - prefix[start]) / (end - start))
    return tuple(smoothed)


def population_z_scores(values: Sequence[float]) -> tuple[float, ...] | None:
    """Normalize one local sequence using population variance, or return None if flat.

    The z-scores are an internal relative signal, not user-facing confidence. They
    answer only "which sampled moments are high compared with this candidate window?".
    ``ddof=0`` is intentional because the complete retrieved local window is treated as
    the population being normalized rather than as a statistical sample of a dataset.
    """
    numeric = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("z-score values must all be finite")
    if not numeric:
        return ()

    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
    scale = max(1.0, *(abs(value) for value in numeric))
    near_zero_std = np.finfo(np.float64).eps * scale * 32.0
    standard_deviation = math.sqrt(variance)
    if standard_deviation <= near_zero_std:
        return None
    return tuple((value - mean) / standard_deviation for value in numeric)


def maximum_subarray(values: Sequence[float]) -> tuple[int, int] | None:
    """Return inclusive indices of the strongest positive contiguous numerical signal.

    This is Kadane's algorithm with deterministic ties: larger sum wins; for equal
    sums the shorter range wins; for equal sum and length the earlier start wins.
    Example ``[-0.8, -0.3, 1.2, 1.5, 0.9, -0.2, -1.0]`` selects indices ``2..4``.
    Kadane knows nothing about images or events: it only finds a contiguous positive
    run in the already-smoothed, locally-normalized numerical sequence.
    """
    numeric = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("maximum-subarray values must all be finite")

    best_sum = 0.0
    best_start: int | None = None
    best_end: int | None = None
    current_sum = 0.0
    current_start = 0

    for index, value in enumerate(numeric):
        # If the prior prefix is zero or negative, restarting here has the same or a
        # better sum and is strictly shorter when equal.
        if current_sum <= 0.0:
            current_sum = value
            current_start = index
        else:
            current_sum += value

        if current_sum <= 0.0:
            continue
        if best_start is None or best_end is None:
            best_sum = current_sum
            best_start = current_start
            best_end = index
            continue

        current_length = index - current_start + 1
        best_length = best_end - best_start + 1
        if _strictly_greater(current_sum, best_sum):
            best_sum = current_sum
            best_start = current_start
            best_end = index
        elif _sums_equal(current_sum, best_sum) and (
            current_length < best_length
            or (current_length == best_length and current_start < best_start)
        ):
            best_sum = current_sum
            best_start = current_start
            best_end = index

    if best_start is None or best_end is None:
        return None
    return best_start, best_end


def select_passage(
    observations: Sequence[TemporalObservation],
    *,
    moving_average_frames: int,
) -> SelectedPassage:
    """Select one contiguous sampled passage while preserving a representative raw hit.

    The sequence is sorted by exact integer ``timestamp_us``. Fewer than three samples
    cannot establish a sustained signal, and a zero/near-zero z-score variance carries
    no relative evidence; both cases fall back to the strongest raw observation.
    Otherwise Winston smooths raw cosine values, computes local population z-scores,
    and runs Kadane to choose passage extent. The representative is still the strongest
    *raw* semantic match inside that extent, so normalization never becomes confidence.
    """
    if not observations:
        raise ValueError("select_passage requires at least one temporal observation")

    ordered = tuple(sorted(observations, key=lambda observation: observation.timestamp_us))
    first_visual = ordered[0].match.visual
    stream_identity = (first_visual.asset_id, first_visual.source_path)
    seen_timestamps: set[int] = set()
    for observation in ordered:
        visual = observation.match.visual
        if (visual.asset_id, visual.source_path) != stream_identity:
            raise ValueError("select_passage observations must belong to one video stream")
        if observation.timestamp_us in seen_timestamps:
            raise ValueError("select_passage observations must have unique timestamps")
        seen_timestamps.add(observation.timestamp_us)

    # Validate the configured width even when a short sequence immediately falls back.
    centered_moving_average((), width=moving_average_frames)
    if len(ordered) < 3:
        return _raw_fallback(ordered)

    raw_scores = tuple(observation.match.score for observation in ordered)
    smoothed = centered_moving_average(raw_scores, width=moving_average_frames)
    normalized = population_z_scores(smoothed)
    if normalized is None:
        return _raw_fallback(ordered)

    selected = maximum_subarray(normalized)
    if selected is None:
        return _raw_fallback(ordered)
    start_index, end_index = selected
    selected_observations = ordered[start_index : end_index + 1]
    representative = min(selected_observations, key=_observation_order_key).match
    return SelectedPassage(
        start_timestamp_us=ordered[start_index].timestamp_us,
        end_timestamp_us=ordered[end_index].timestamp_us,
        representative=representative,
    )


def _raw_fallback(observations: Sequence[TemporalObservation]) -> SelectedPassage:
    """Return one exact sampled moment using strongest raw score then deterministic provenance."""
    representative_observation = min(observations, key=_observation_order_key)
    timestamp_us = representative_observation.timestamp_us
    return SelectedPassage(
        start_timestamp_us=timestamp_us,
        end_timestamp_us=timestamp_us,
        representative=representative_observation.match,
    )


def _observation_order_key(
    observation: TemporalObservation,
) -> tuple[float, int, str, str, int, int, int, int]:
    """Prefer higher raw score, then earlier timestamp and deterministic visual provenance."""
    match_key = _match_order_key(observation.match)
    return (
        match_key[0],
        observation.timestamp_us,
        match_key[1],
        match_key[2],
        match_key[3],
        match_key[4],
        match_key[5],
        match_key[6],
    )


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


def _sums_equal(left: float, right: float) -> bool:
    """Treat only machine-scale rounding differences as equal Kadane sums."""
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _strictly_greater(left: float, right: float) -> bool:
    """Return whether one Kadane sum is meaningfully larger rather than rounding noise."""
    return left > right and not _sums_equal(left, right)