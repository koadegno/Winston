"""Read-only orchestration for Winston semantic visual search."""

from collections.abc import Sequence
import math

import numpy as np

from winston.config import QdrantSettings, Settings
from winston.embeddings.base import MultimodalEmbedder
from winston.embeddings.factory import create_embedder
from winston.embeddings.models import EmbeddingBatch
from winston.index.base import VisualIndex
from winston.index.models import ScoredVisual, VisualVector
from winston.index.qdrant import QdrantVisualIndex
from winston.ingest.models import MediaType
from winston.search.models import SearchResult, SearchRunError
from winston.search.temporal import (
    MICROSECONDS_PER_SECOND,
    TemporalObservationAccumulator,
    build_temporal_windows,
    collapse_best_by_timestamp,
    cosine_similarity,
    select_passage,
)


class SearchPipeline:
    """Compose one semantic query from text embedding through read-only visual refinement."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: MultimodalEmbedder,
        visual_index: VisualIndex,
    ) -> None:
        """Bind one search pipeline to reusable embedding and visual-index dependencies."""
        self._settings = settings
        self._embedder = embedder
        self._visual_index = visual_index

    async def run(self, query: str, *, limit: int) -> tuple[SearchResult, ...]:
        """Execute one read-only semantic query and return grouped globally ranked results.

        Qdrant ANN is used only to discover promising assets/timestamps. Video windows
        are then read completely from the existing index and scored again with exact
        local cosine, so weak intermediate keyframes are present for smoothing,
        z-score normalization, and Kadane passage selection.
        """
        normalized_query, requested_limit = _validate_search_request(query, limit)

        # Opening is deliberately read-only and occurs before provider inference so an
        # absent/incompatible index fails without spending a remote embedding request.
        await self._visual_index.open_search(self._embedder.identity)
        embedding_batch = await self._embedder.embed_texts([normalized_query])
        query_vector = _query_vector(
            embedding_batch,
            expected_dimension=self._embedder.identity.dimension,
        )

        coarse_matches = await self._discover_coarse_matches(
            query_vector,
            requested_limit=requested_limit,
        )

        results: list[SearchResult] = list(_photo_results(coarse_matches))
        video_matches = tuple(
            match
            for match in coarse_matches
            if match.visual.media_type is MediaType.VIDEO
        )
        if video_matches:
            seed_observations = collapse_best_by_timestamp(video_matches)
            windows = build_temporal_windows(
                seed_observations,
                context_seconds=float(self._settings.search.temporal_context_seconds),
            )
            for window in windows:
                local_observations = TemporalObservationAccumulator()
                async for visual in self._visual_index.iter_visuals(
                    asset_id=window.asset_id,
                    start_timestamp_us=window.start_timestamp_us,
                    end_timestamp_us=window.end_timestamp_us,
                ):
                    # One content hash can theoretically appear at several source paths.
                    # The coarse seed selected this specific path, so do not let a duplicate
                    # copy of the same bytes influence this passage's provenance.
                    if visual.source_path != window.source_path:
                        continue
                    local_observations.add(
                        ScoredVisual(
                            visual=visual,
                            score=cosine_similarity(query_vector, visual.vector),
                        )
                    )

                observations = local_observations.observations()
                if not observations:
                    continue
                passage = select_passage(
                    observations,
                    moving_average_frames=int(
                        self._settings.search.moving_average_frames
                    ),
                )
                representative = passage.representative
                representative_visual = representative.visual
                representative_timestamp = representative_visual.timestamp_seconds
                if representative_timestamp is None:
                    raise SearchRunError(
                        "video passage representative is missing its timestamp"
                    )
                results.append(
                    SearchResult(
                        source_path=representative_visual.source_path,
                        media_type=MediaType.VIDEO,
                        start_timestamp_seconds=(
                            passage.start_timestamp_us / MICROSECONDS_PER_SECOND
                        ),
                        end_timestamp_seconds=(
                            passage.end_timestamp_us / MICROSECONDS_PER_SECOND
                        ),
                        representative_timestamp_seconds=representative_timestamp,
                        region_kind=representative_visual.region_kind,
                        region=representative_visual.region,
                        raw_score=representative.score,
                    )
                )

        results.sort(key=_result_order_key)
        return tuple(results[:requested_limit])

    async def _discover_coarse_matches(
        self,
        query_vector: VisualVector,
        *,
        requested_limit: int,
    ) -> tuple[ScoredVisual, ...]:
        """Overfetch ANN points until grouping exposes enough distinct result opportunities.

        Raw Qdrant points are regions, not user-facing results. A single photo or one
        video event can therefore consume many adjacent ANN ranks. Start with the
        configured candidate budget, then double it only when grouping still exposes
        fewer distinct photo assets/video neighborhoods than the requested result count.
        The configured hard limit bounds the extra ANN work and memory.
        """
        hard_limit = int(self._settings.search.candidate_max_limit)
        current_limit = min(
            max(
                int(self._settings.search.candidate_limit),
                requested_limit,
            ),
            hard_limit,
        )
        context_seconds = float(self._settings.search.temporal_context_seconds)

        while True:
            matches = tuple(
                await self._visual_index.search_visuals(
                    query_vector,
                    limit=current_limit,
                )
            )
            opportunity_count = _coarse_result_opportunity_count(
                matches,
                context_seconds=context_seconds,
            )
            if opportunity_count >= requested_limit:
                return matches
            if len(matches) < current_limit or current_limit >= hard_limit:
                return matches

            next_limit = min(hard_limit, current_limit * 2)
            if next_limit <= current_limit:
                return matches
            current_limit = next_limit


def _coarse_result_opportunity_count(
    matches: Sequence[ScoredVisual],
    *,
    context_seconds: float,
) -> int:
    """Count grouped photo assets plus merged video neighborhoods in raw ANN matches."""
    photo_asset_ids = {
        match.visual.asset_id
        for match in matches
        if match.visual.media_type is MediaType.IMAGE
    }
    video_matches = tuple(
        match
        for match in matches
        if match.visual.media_type is MediaType.VIDEO
    )
    if not video_matches:
        return len(photo_asset_ids)

    video_observations = collapse_best_by_timestamp(video_matches)
    video_windows = build_temporal_windows(
        video_observations,
        context_seconds=context_seconds,
    )
    return len(photo_asset_ids) + len(video_windows)


def _validate_search_request(query: str, limit: int) -> tuple[str, int]:
    """Normalize one query and reject invalid limits before expensive dependencies are used."""
    if not isinstance(query, str):
        raise SearchRunError("search query must be text")
    normalized_query = query.strip()
    if not normalized_query:
        raise SearchRunError("search query must not be blank")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise SearchRunError("search result limit must be a positive integer")
    return normalized_query, limit


def _query_vector(
    batch: EmbeddingBatch,
    *,
    expected_dimension: int,
) -> VisualVector:
    """Validate that embedding one query produced exactly one finite non-zero float32 vector."""
    matrix = batch.vectors
    expected_shape = (1, expected_dimension)
    if matrix.ndim != 2 or matrix.shape != expected_shape:
        raise SearchRunError(
            f"text embedding returned shape {matrix.shape}; expected {expected_shape}"
        )
    if matrix.dtype != np.float32:
        raise SearchRunError("text embedding vector must use float32 values")
    vector = matrix[0]
    if not bool(np.isfinite(vector).all()):
        raise SearchRunError("text embedding vector must contain only finite values")
    if not bool(np.any(vector != 0.0)):
        raise SearchRunError("text embedding vector must be non-zero")
    return vector


def _photo_results(matches: Sequence[ScoredVisual]) -> tuple[SearchResult, ...]:
    """Return one strongest representative region for each still-image asset.

    A photo can produce a full-frame vector plus multiple tile vectors. They represent
    one user asset, so only the strongest raw cosine survives grouping; deterministic
    provenance fields break exact score ties.
    """
    best_by_asset: dict[str, ScoredVisual] = {}
    for match in matches:
        visual = match.visual
        if visual.media_type is not MediaType.IMAGE:
            continue
        current = best_by_asset.get(visual.asset_id)
        if current is None or _visual_match_order_key(match) < _visual_match_order_key(current):
            best_by_asset[visual.asset_id] = match

    mapped: list[SearchResult] = []
    for asset_id in sorted(best_by_asset):
        match = best_by_asset[asset_id]
        visual = match.visual
        mapped.append(
            SearchResult(
                source_path=visual.source_path,
                media_type=MediaType.IMAGE,
                start_timestamp_seconds=None,
                end_timestamp_seconds=None,
                representative_timestamp_seconds=None,
                region_kind=visual.region_kind,
                region=visual.region,
                raw_score=match.score,
            )
        )
    return tuple(mapped)


def _visual_match_order_key(
    match: ScoredVisual,
) -> tuple[float, str, str, int, int, int, int, float]:
    """Prefer higher raw score then stable source and region provenance for exact ties."""
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
        region.scale,
    )


def _result_order_key(
    result: SearchResult,
) -> tuple[float, str, float, float, str, int, int, int, int, float]:
    """Apply Phase 1F's global ranking: raw cosine desc, source path asc, then time asc."""
    start = -1.0 if result.start_timestamp_seconds is None else result.start_timestamp_seconds
    representative = (
        -1.0
        if result.representative_timestamp_seconds is None
        else result.representative_timestamp_seconds
    )
    region = result.region
    return (
        -result.raw_score,
        result.source_path,
        start,
        representative,
        result.region_kind.value,
        region.x,
        region.y,
        region.width,
        region.height,
        region.scale,
    )


async def _close_dependencies(
    visual_index: VisualIndex,
    embedder: MultimodalEmbedder,
) -> tuple[str, ...]:
    """Close both reusable dependencies while retaining every cleanup failure."""
    errors: list[str] = []
    try:
        await visual_index.close()
    except Exception as exc:
        errors.append(f"visual index close failed: {exc}")
    try:
        await embedder.close()
    except Exception as exc:
        errors.append(f"embedder close failed: {exc}")
    return tuple(errors)


async def run_search(
    query: str,
    *,
    limit: int,
    settings: Settings,
) -> tuple[SearchResult, ...]:
    """Build one search stack, execute it read-only, and close resources without masking failures."""
    # Validate user input before model/Qdrant construction. SearchPipeline.run validates
    # again because it is a public independently usable orchestration object.
    normalized_query, requested_limit = _validate_search_request(query, limit)

    embedder = create_embedder(settings)
    try:
        visual_index: VisualIndex = QdrantVisualIndex(
            settings.qdrant,
            timeline_page_size=int(settings.search.timeline_page_size),
        )
    except BaseException as primary:
        try:
            await embedder.close()
        except Exception as cleanup_error:
            primary.add_note(f"embedder close failed: {cleanup_error}")
        raise

    try:
        pipeline = SearchPipeline(
            settings=settings,
            embedder=embedder,
            visual_index=visual_index,
        )
        results = await pipeline.run(normalized_query, limit=requested_limit)
    except BaseException as primary:
        cleanup_errors = await _close_dependencies(visual_index, embedder)
        for cleanup_error in cleanup_errors:
            primary.add_note(cleanup_error)
        raise

    cleanup_errors = await _close_dependencies(visual_index, embedder)
    if cleanup_errors:
        raise SearchRunError("; ".join(cleanup_errors))
    return results