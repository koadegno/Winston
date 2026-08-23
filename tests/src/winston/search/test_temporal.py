import math

import numpy as np
import pytest

from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import IndexedVisual, RegionGeometry, SampleKind, ScoredVisual
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind
from winston.search.temporal import (
    build_temporal_windows,
    collapse_best_by_timestamp,
    cosine_similarity,
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
