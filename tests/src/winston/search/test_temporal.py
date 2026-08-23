import math

import numpy as np
import pytest

from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import IndexedVisual, RegionGeometry, SampleKind, ScoredVisual
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind
from winston.search.temporal import (
    build_temporal_windows,
    centered_moving_average,
    collapse_best_by_timestamp,
    cosine_similarity,
    maximum_subarray,
    population_z_scores,
    select_passage,
)

IDENTITY = EmbeddingIdentity(
    model_id="jinaai/jina-clip-v1",
    dimension=3,
    preprocessing_version=1,
)
ASSET_A = "a" * 64
ASSET_B = "b" * 64


def _video_match(
    timestamp_seconds: float,
    score: float,
    *,
    asset_id: str = ASSET_A,
    source_path: str = "cameras/a.mkv",
    region_kind: RegionKind = RegionKind.FULL,
    tile_index: int = 0,
) -> ScoredVisual:
    """Build one strict keyframe match with distinguishable deterministic region geometry."""
    if region_kind is RegionKind.FULL:
        region = RegionGeometry(x=0, y=0, width=1920, height=1080, scale=1.0)
    else:
        region = RegionGeometry(
            x=tile_index * 100,
            y=tile_index * 50,
            width=960,
            height=540,
            scale=0.5,
        )
    visual = IndexedVisual(
        asset_id=asset_id,
        source_path=source_path,
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=timestamp_seconds,
        region_kind=region_kind,
        region=region,
        vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        embedding_identity=IDENTITY,
    )
    return ScoredVisual(visual=visual, score=score)


def test_cosine_similarity_matches_known_vectors() -> None:
    """Exact local cosine reproduces identical, orthogonal, and opposite vector geometry."""
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    same = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    orthogonal = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    opposite = np.array([-4.0, 0.0, 0.0], dtype=np.float32)

    assert cosine_similarity(x, same) == pytest.approx(1.0)
    assert cosine_similarity(x, orthogonal) == pytest.approx(0.0)
    assert cosine_similarity(x, opposite) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("query", "candidate", "message"),
    [
        (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "dimension",
        ),
        (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "non-zero",
        ),
        (
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "non-zero",
        ),
        (
            np.array([1.0, math.nan, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "finite",
        ),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    query: np.ndarray,
    candidate: np.ndarray,
    message: str,
) -> None:
    """Local reranking fails explicitly rather than producing undefined cosine values."""
    with pytest.raises(ValueError, match=message):
        cosine_similarity(query, candidate)  # type: ignore[arg-type]


def test_collapse_best_by_timestamp_keeps_strongest_region() -> None:
    """One keyframe score is the strongest region: 0.27, 0.31, 0.61, 0.29 -> tile2 at 0.61."""
    matches = [
        _video_match(12.0, 0.27, region_kind=RegionKind.FULL),
        _video_match(12.0, 0.31, region_kind=RegionKind.TILE, tile_index=1),
        _video_match(12.0, 0.61, region_kind=RegionKind.TILE, tile_index=2),
        _video_match(12.0, 0.29, region_kind=RegionKind.TILE, tile_index=3),
    ]

    observations = collapse_best_by_timestamp(matches)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.timestamp_us == 12_000_000
    assert observation.match.score == pytest.approx(0.61)
    assert observation.match.visual.region_kind is RegionKind.TILE
    assert observation.match.visual.region.x == 200


def test_collapse_best_by_timestamp_keeps_assets_separate_and_sorted() -> None:
    """Equal timestamps from different videos remain distinct temporal observations."""
    matches = [
        _video_match(
            20.0,
            0.4,
            asset_id=ASSET_B,
            source_path="cameras/b.mkv",
        ),
        _video_match(10.0, 0.5),
        _video_match(
            10.0,
            0.6,
            asset_id=ASSET_B,
            source_path="cameras/b.mkv",
        ),
    ]

    observations = collapse_best_by_timestamp(matches)

    assert [
        (item.match.visual.asset_id, item.timestamp_us)
        for item in observations
    ] == [
        (ASSET_A, 10_000_000),
        (ASSET_B, 10_000_000),
        (ASSET_B, 20_000_000),
    ]


def test_build_temporal_windows_merges_reference_example() -> None:
    """Seeds 100s, 108s, 310s with +/-15s context become 85..123s and 295..325s."""
    seeds = collapse_best_by_timestamp(
        [
            _video_match(100.0, 0.7),
            _video_match(108.0, 0.8),
            _video_match(310.0, 0.6),
        ]
    )

    windows = build_temporal_windows(seeds, context_seconds=15.0)

    assert [
        (window.asset_id, window.source_path, window.start_timestamp_us, window.end_timestamp_us)
        for window in windows
    ] == [
        (ASSET_A, "cameras/a.mkv", 85_000_000, 123_000_000),
        (ASSET_A, "cameras/a.mkv", 295_000_000, 325_000_000),
    ]


def test_build_temporal_windows_merges_touching_windows_and_clips_zero() -> None:
    """Touching [5,15] and [15,25] intervals merge, while an early seed cannot start below zero."""
    touching = collapse_best_by_timestamp(
        [_video_match(10.0, 0.5), _video_match(20.0, 0.6)]
    )
    early = collapse_best_by_timestamp([_video_match(3.0, 0.7)])

    touching_windows = build_temporal_windows(touching, context_seconds=5.0)
    early_windows = build_temporal_windows(early, context_seconds=5.0)

    assert len(touching_windows) == 1
    assert touching_windows[0].start_timestamp_us == 5_000_000
    assert touching_windows[0].end_timestamp_us == 25_000_000
    assert len(early_windows) == 1
    assert early_windows[0].start_timestamp_us == 0
    assert early_windows[0].end_timestamp_us == 8_000_000


def test_build_temporal_windows_never_merges_different_assets() -> None:
    """Overlapping wall-clock positions from independent videos remain independent Qdrant windows."""
    seeds = collapse_best_by_timestamp(
        [
            _video_match(10.0, 0.5),
            _video_match(
                10.0,
                0.6,
                asset_id=ASSET_B,
                source_path="cameras/b.mkv",
            ),
        ]
    )

    windows = build_temporal_windows(seeds, context_seconds=5.0)

    assert [(window.asset_id, window.source_path) for window in windows] == [
        (ASSET_A, "cameras/a.mkv"),
        (ASSET_B, "cameras/b.mkv"),
    ]


@pytest.mark.parametrize("context_seconds", [0.0, -1.0, math.nan, math.inf])
def test_build_temporal_windows_rejects_invalid_context(context_seconds: float) -> None:
    """Semantic context must be finite and positive because it changes temporal grouping."""
    seeds = collapse_best_by_timestamp([_video_match(10.0, 0.5)])

    with pytest.raises(ValueError, match="context_seconds"):
        build_temporal_windows(seeds, context_seconds=context_seconds)


def test_centered_moving_average_width_three_uses_available_edge_neighbors() -> None:
    """Width 3 uses previous/current/next inside the sequence and only available values at edges."""
    smoothed = centered_moving_average([1.0, 2.0, 3.0, 4.0, 5.0], width=3)

    assert smoothed == pytest.approx((1.5, 2.0, 3.0, 4.0, 4.5))


@pytest.mark.parametrize("width", [0, 2, 4])
def test_centered_moving_average_rejects_non_positive_or_even_width(width: int) -> None:
    """A centered semantic smoothing window must have one exact center sample."""
    with pytest.raises(ValueError, match="odd.*positive|positive.*odd"):
        centered_moving_average([1.0, 2.0, 3.0], width=width)


def test_population_z_scores_use_population_standard_deviation() -> None:
    """Population z-score uses ddof=0 and is only an internal local-sequence normalization."""
    scores = population_z_scores([1.0, 2.0, 3.0])

    assert scores is not None
    assert scores == pytest.approx((-1.2247448714, 0.0, 1.2247448714))


def test_population_z_scores_return_none_for_zero_variance() -> None:
    """A flat local sequence carries no relative temporal signal and triggers raw-score fallback."""
    assert population_z_scores([0.5, 0.5, 0.5]) is None


def test_maximum_subarray_selects_documented_positive_cluster() -> None:
    """Kadane selects 1.2, 1.5, 0.9 from the documented local z-score example."""
    selected = maximum_subarray([-0.8, -0.3, 1.2, 1.5, 0.9, -0.2, -1.0])

    assert selected == (2, 4)


def test_maximum_subarray_returns_none_without_positive_signal() -> None:
    """A non-positive sequence has no positive contiguous temporal signal."""
    assert maximum_subarray([-1.0, 0.0, -2.0]) is None


def test_maximum_subarray_prefers_shorter_equal_sum_range() -> None:
    """For equal sum, [1.0] beats the longer [1.0, 0.0] interval."""
    assert maximum_subarray([1.0, 0.0, -2.0]) == (0, 0)


def test_maximum_subarray_prefers_earlier_equal_length_range() -> None:
    """For equal sum and equal length, the earlier interval is deterministic."""
    assert maximum_subarray([1.0, -2.0, 1.0]) == (0, 0)


def test_select_passage_falls_back_to_strongest_raw_match_for_short_sequence() -> None:
    """With fewer than three sampled timestamps Winston cannot infer a sustained temporal signal."""
    observations = collapse_best_by_timestamp(
        [_video_match(4.0, 0.2), _video_match(8.0, 0.8)]
    )

    passage = select_passage(observations, moving_average_frames=3)

    assert passage.start_timestamp_us == 8_000_000
    assert passage.end_timestamp_us == 8_000_000
    assert passage.representative.score == pytest.approx(0.8)


def test_select_passage_falls_back_to_earliest_strongest_raw_match_for_flat_sequence() -> None:
    """Zero local variance cannot produce z-score evidence, so equal raw scores choose the earliest sample."""
    observations = collapse_best_by_timestamp(
        [_video_match(0.0, 0.5), _video_match(4.0, 0.5), _video_match(8.0, 0.5)]
    )

    passage = select_passage(observations, moving_average_frames=3)

    assert passage.start_timestamp_us == 0
    assert passage.end_timestamp_us == 0
    assert passage.representative.visual.timestamp_us == 0


def test_select_passage_prefers_sustained_cluster_over_isolated_raw_spike() -> None:
    """Smoothing + local z-score makes a sustained cluster win over one isolated higher raw frame."""
    raw_scores = [0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.65, 0.8, 0.7, 0.1, 0.1]
    matches = [
        _video_match(
            float(index * 4),
            score,
            region_kind=RegionKind.TILE if index == 8 else RegionKind.FULL,
            tile_index=index,
        )
        for index, score in enumerate(raw_scores)
    ]
    observations = collapse_best_by_timestamp(matches)

    passage = select_passage(observations, moving_average_frames=3)

    assert passage.start_timestamp_us == 28_000_000
    assert passage.end_timestamp_us == 36_000_000
    assert passage.representative.score == pytest.approx(0.8)
    assert passage.representative.visual.timestamp_us == 32_000_000
    assert passage.representative.visual.region_kind is RegionKind.TILE


def test_select_passage_sorts_observations_by_exact_timestamp() -> None:
    """Provider order does not affect passage boundaries because temporal math sorts timestamp_us exactly."""
    observations = list(
        collapse_best_by_timestamp(
            [
                _video_match(0.0, 0.1),
                _video_match(4.0, 0.6),
                _video_match(8.0, 0.7),
                _video_match(12.0, 0.6),
                _video_match(16.0, 0.1),
            ]
        )
    )
    observations.reverse()

    passage = select_passage(observations, moving_average_frames=3)

    assert passage.start_timestamp_us <= passage.representative.visual.timestamp_us <= passage.end_timestamp_us
