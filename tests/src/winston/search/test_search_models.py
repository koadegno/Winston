import math

import pytest
from pydantic import BaseModel, ValidationError

from winston import search
from winston.index.models import RegionGeometry
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind


def _search_result_type() -> type[BaseModel]:
    """Return the public result model while making a missing Phase 1F API an explicit contract failure."""
    result_type = getattr(search, "SearchResult", None)
    assert result_type is not None, "winston.search must export SearchResult"
    assert isinstance(result_type, type)
    assert issubclass(result_type, BaseModel)
    return result_type


def _full_region() -> RegionGeometry:
    """Build one valid full-frame geometry reused by result contract tests."""
    return RegionGeometry(x=0, y=0, width=1920, height=1080, scale=1.0)


def _tile_region() -> RegionGeometry:
    """Build one valid tile geometry reused by video result contract tests."""
    return RegionGeometry(x=480, y=270, width=960, height=540, scale=0.5)


def test_search_result_and_error_are_public() -> None:
    """The search package must expose the user-facing result and orchestration error."""
    result_type = _search_result_type()
    error_type = getattr(search, "SearchRunError", None)

    assert result_type.model_config.get("strict") is True
    assert result_type.model_config.get("frozen") is True
    assert isinstance(error_type, type)
    assert issubclass(error_type, RuntimeError)


def test_image_result_has_no_temporal_interval() -> None:
    """A still image result carries a raw semantic score and no fabricated timestamps."""
    result_type = _search_result_type()

    result = result_type(
        source_path="photos/entrance.jpg",
        media_type=MediaType.IMAGE,
        start_timestamp_seconds=None,
        end_timestamp_seconds=None,
        representative_timestamp_seconds=None,
        region_kind=RegionKind.FULL,
        region=_full_region(),
        raw_score=0.713,
    )

    assert result.source_path == "photos/entrance.jpg"
    assert result.raw_score == pytest.approx(0.713)


def test_video_result_requires_ordered_temporal_interval() -> None:
    """A video result identifies one representative keyframe inside its selected passage."""
    result_type = _search_result_type()

    result = result_type(
        source_path="camera/crossing.mkv",
        media_type=MediaType.VIDEO,
        start_timestamp_seconds=96.0,
        end_timestamp_seconds=112.0,
        representative_timestamp_seconds=104.0,
        region_kind=RegionKind.TILE,
        region=_tile_region(),
        raw_score=0.681,
    )

    assert result.start_timestamp_seconds == 96.0
    assert result.representative_timestamp_seconds == 104.0
    assert result.end_timestamp_seconds == 112.0


def test_image_result_rejects_video_timestamps() -> None:
    """Still-image results cannot accidentally expose a partial or complete video interval."""
    result_type = _search_result_type()

    with pytest.raises(ValidationError):
        result_type(
            source_path="photo.jpg",
            media_type=MediaType.IMAGE,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=None,
            representative_timestamp_seconds=None,
            region_kind=RegionKind.FULL,
            region=_full_region(),
            raw_score=0.5,
        )


def test_video_result_rejects_missing_or_misordered_timestamps() -> None:
    """Video intervals must be complete and satisfy start <= representative <= end."""
    result_type = _search_result_type()

    with pytest.raises(ValidationError):
        result_type(
            source_path="camera.mkv",
            media_type=MediaType.VIDEO,
            start_timestamp_seconds=10.0,
            end_timestamp_seconds=None,
            representative_timestamp_seconds=11.0,
            region_kind=RegionKind.FULL,
            region=_full_region(),
            raw_score=0.5,
        )

    with pytest.raises(ValidationError):
        result_type(
            source_path="camera.mkv",
            media_type=MediaType.VIDEO,
            start_timestamp_seconds=12.0,
            end_timestamp_seconds=14.0,
            representative_timestamp_seconds=10.0,
            region_kind=RegionKind.FULL,
            region=_full_region(),
            raw_score=0.5,
        )


@pytest.mark.parametrize("raw_score", [math.nan, math.inf, -math.inf])
def test_search_result_rejects_non_finite_raw_score(raw_score: float) -> None:
    """Raw cosine similarity must remain a real finite number, never an invalid confidence value."""
    result_type = _search_result_type()

    with pytest.raises(ValidationError):
        result_type(
            source_path="photo.jpg",
            media_type=MediaType.IMAGE,
            start_timestamp_seconds=None,
            end_timestamp_seconds=None,
            representative_timestamp_seconds=None,
            region_kind=RegionKind.FULL,
            region=_full_region(),
            raw_score=raw_score,
        )


@pytest.mark.parametrize(
    "source_path",
    ["/absolute.jpg", "../escape.jpg", "folder/../escape.jpg", "folder\\camera.jpg", "folder//camera.jpg"],
)
def test_search_result_rejects_non_normalized_source_path(source_path: str) -> None:
    """Search results preserve the same normalized relative POSIX path contract as indexed visuals."""
    result_type = _search_result_type()

    with pytest.raises(ValidationError):
        result_type(
            source_path=source_path,
            media_type=MediaType.IMAGE,
            start_timestamp_seconds=None,
            end_timestamp_seconds=None,
            representative_timestamp_seconds=None,
            region_kind=RegionKind.FULL,
            region=_full_region(),
            raw_score=0.5,
        )


def test_search_result_is_frozen_and_strict() -> None:
    """Result values cannot be mutated or silently coerce score strings after validation."""
    result_type = _search_result_type()
    result = result_type(
        source_path="photo.jpg",
        media_type=MediaType.IMAGE,
        start_timestamp_seconds=None,
        end_timestamp_seconds=None,
        representative_timestamp_seconds=None,
        region_kind=RegionKind.FULL,
        region=_full_region(),
        raw_score=0.5,
    )

    with pytest.raises(ValidationError):
        result.raw_score = 0.7

    with pytest.raises(ValidationError):
        result_type(
            source_path="photo.jpg",
            media_type=MediaType.IMAGE,
            start_timestamp_seconds=None,
            end_timestamp_seconds=None,
            representative_timestamp_seconds=None,
            region_kind=RegionKind.FULL,
            region=_full_region(),
            raw_score="0.5",
        )
