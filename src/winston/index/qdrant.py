"""Qdrant implementation of Winston's class-agnostic visual index."""

from collections.abc import AsyncIterator, Sequence
from itertools import batched
from typing import Protocol, cast
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
from qdrant_client import AsyncQdrantClient, models

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.identity import visual_point_id
from winston.index.models import (
    ASSET_ID_PATTERN,
    IncompatibleVisualIndexError,
    IndexedVisual,
    RegionGeometry,
    SampleKind,
    ScoredVisual,
    VisualIndexConfigurationError,
    VisualIndexError,
    VisualIndexSession,
    VisualSearchSession,
    VisualVector,
)
from winston.ingest.models import MediaType
from winston.sampling.regions import RegionKind

type CollectionMetadataValue = str | int
type CollectionMetadata = dict[str, CollectionMetadataValue]
type DenseVectorOutput = list[float] | dict[str, list[float]] | None
type ScrollOffset = int | str | None

WINSTON_SCHEMA_VERSION = 2
VISUAL_DISTANCE = models.Distance.COSINE
VISUAL_DISTANCE_NAME = "cosine"


class _StoredRegion(BaseModel):
    """Strict JSON region shape persisted in one schema-v2 Qdrant payload."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    x: int
    y: int
    width: int
    height: int
    scale: float


class _StoredVisualPayload(BaseModel):
    """Strict schema-v2 payload required to reconstruct one Winston visual safely."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    asset_id: str
    source_path: str
    media_type: str
    sample_kind: str
    timestamp_seconds: float | None
    region_kind: str
    region: _StoredRegion
    timestamp_us: int | None
    model_id: str
    dimension: int
    preprocessing_version: int


class _StoredPoint(Protocol):
    """Storage-read shape shared by Qdrant scored points and scroll records."""

    payload: dict[str, JsonValue] | None
    vector: DenseVectorOutput


class _CollectionParams(Protocol):
    """Structural subset of Qdrant collection params required by Winston."""

    vectors: models.VectorParams | dict[str, models.VectorParams] | None


class _CollectionConfig(Protocol):
    """Structural subset of Qdrant collection config required by Winston."""

    params: _CollectionParams
    metadata: CollectionMetadata | None


class _CollectionInfo(Protocol):
    """Structural subset of Qdrant collection info required by Winston."""

    config: _CollectionConfig


class _QdrantClient(Protocol):
    """Minimal Qdrant operations used by Winston's visual index."""

    async def collection_exists(self, collection_name: str) -> bool:
        """Return whether the configured collection exists."""
        ...

    async def create_collection(
        self,
        collection_name: str,
        *,
        vectors_config: dict[str, models.VectorParams],
        metadata: CollectionMetadata,
    ) -> bool:
        """Create a collection with named vectors and Winston metadata."""
        ...

    async def get_collection(self, collection_name: str) -> _CollectionInfo:
        """Return the collection configuration needed for compatibility validation."""
        ...

    async def query_points(
        self,
        collection_name: str,
        *,
        query: list[float],
        using: str,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> models.QueryResponse:
        """Return coarse semantic nearest neighbors for one dense query vector."""
        ...

    async def scroll(
        self,
        collection_name: str,
        *,
        scroll_filter: models.Filter,
        limit: int,
        offset: ScrollOffset,
        with_payload: bool,
        with_vectors: list[str],
    ) -> tuple[list[models.Record], ScrollOffset]:
        """Return one bounded page of stored visual records."""
        ...

    async def delete(
        self,
        collection_name: str,
        *,
        points_selector: models.FilterSelector,
        wait: bool,
    ) -> models.UpdateResult:
        """Delete points matching one Winston-owned selector."""
        ...

    async def upsert(
        self,
        collection_name: str,
        *,
        points: list[models.PointStruct],
        wait: bool,
    ) -> models.UpdateResult:
        """Upsert one bounded batch of points."""
        ...

    async def close(self) -> None:
        """Release client resources."""
        ...


class QdrantVisualIndex:
    """Strict, dataset-owned Qdrant backend for Winston visual embeddings."""

    def __init__(
        self,
        settings: QdrantSettings,
        *,
        timeline_page_size: int = 256,
        client: _QdrantClient | None = None,
    ) -> None:
        """Create an index session around one reusable asynchronous Qdrant client."""
        if (
            isinstance(timeline_page_size, bool)
            or not isinstance(timeline_page_size, int)
            or timeline_page_size <= 0
        ):
            raise VisualIndexConfigurationError(
                "timeline_page_size must be a positive integer"
            )
        self._settings = settings
        self._timeline_page_size = timeline_page_size
        self._client: _QdrantClient = client or AsyncQdrantClient(url=settings.url)
        self._identity: EmbeddingIdentity | None = None
        self._search_identity: EmbeddingIdentity | None = None

    async def ensure_compatible(
        self,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> VisualIndexSession:
        """Create or strictly validate one dataset-owned visual collection."""
        try:
            canonical_dataset_id = str(UUID(dataset_instance_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise VisualIndexConfigurationError(
                f"dataset_instance_id must be a UUID, got {dataset_instance_id!r}"
            ) from exc

        try:
            exists = await self._client.collection_exists(self._settings.collection)
            if exists:
                info = await self._client.get_collection(self._settings.collection)
                index_instance_id = self._validate_collection(
                    info,
                    identity,
                    canonical_dataset_id,
                )
            else:
                index_instance_id = str(uuid4())
                created = await self._client.create_collection(
                    self._settings.collection,
                    vectors_config={
                        self._settings.vector_name: models.VectorParams(
                            size=identity.dimension,
                            distance=VISUAL_DISTANCE,
                        )
                    },
                    metadata=self._collection_metadata(
                        identity,
                        dataset_instance_id=canonical_dataset_id,
                        index_instance_id=index_instance_id,
                    ),
                )
                if not created:
                    raise VisualIndexError(
                        f"Qdrant did not create visual index collection '{self._settings.collection}'"
                    )
        except (IncompatibleVisualIndexError, VisualIndexConfigurationError, VisualIndexError):
            raise
        except Exception as exc:
            raise VisualIndexError(
                f"Failed to ensure visual index collection '{self._settings.collection}'"
            ) from exc

        self._identity = identity
        return VisualIndexSession(index_instance_id=index_instance_id)

    async def open_search(self, identity: EmbeddingIdentity) -> VisualSearchSession:
        """Open an existing compatible collection for semantic reads without mutating Qdrant."""
        try:
            exists = await self._client.collection_exists(self._settings.collection)
            if not exists:
                raise VisualIndexConfigurationError(
                    f"Visual index collection '{self._settings.collection}' does not exist; "
                    "run 'winston index' first."
                )

            info = await self._client.get_collection(self._settings.collection)
            dataset_instance_id = self._dataset_instance_id(info)
            index_instance_id = self._validate_collection(
                info,
                identity,
                dataset_instance_id,
            )
        except (IncompatibleVisualIndexError, VisualIndexConfigurationError, VisualIndexError):
            raise
        except Exception as exc:
            raise VisualIndexError(
                f"Failed to open visual index collection '{self._settings.collection}' for search"
            ) from exc

        self._search_identity = identity
        return VisualSearchSession(
            dataset_instance_id=dataset_instance_id,
            index_instance_id=index_instance_id,
        )

    async def search_visuals(
        self,
        query_vector: VisualVector,
        *,
        limit: int,
    ) -> Sequence[ScoredVisual]:
        """Return coarse Qdrant ANN candidates mapped immediately to Winston-owned models."""
        identity = self._search_identity
        if identity is None:
            raise VisualIndexConfigurationError(
                "open_search() must succeed before semantic visual retrieval"
            )
        self._validate_query_vector(query_vector, identity, limit=limit)

        try:
            response = await self._client.query_points(
                self._settings.collection,
                query=query_vector.tolist(),
                using=self._settings.vector_name,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            return tuple(
                self._scored_visual_from_point(point, identity)
                for point in response.points
            )
        except (IncompatibleVisualIndexError, VisualIndexConfigurationError, VisualIndexError):
            raise
        except Exception as exc:
            raise VisualIndexError(
                f"Failed to query visual index collection '{self._settings.collection}'"
            ) from exc

    async def iter_visuals(
        self,
        *,
        asset_id: str,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> AsyncIterator[IndexedVisual]:
        """Stream every stored video visual in one bounded asset/time window page by page."""
        identity = self._search_identity
        if identity is None:
            raise VisualIndexConfigurationError(
                "open_search() must succeed before timeline retrieval"
            )
        self._validate_timeline_window(
            asset_id=asset_id,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
        )

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="asset_id",
                    match=models.MatchValue(value=asset_id),
                ),
                models.FieldCondition(
                    key="media_type",
                    match=models.MatchValue(value=MediaType.VIDEO.value),
                ),
                models.FieldCondition(
                    key="timestamp_us",
                    range=models.Range(
                        gte=start_timestamp_us,
                        lte=end_timestamp_us,
                    ),
                ),
            ]
        )
        offset: ScrollOffset = None

        while True:
            try:
                records, next_offset = await self._client.scroll(
                    self._settings.collection,
                    scroll_filter=scroll_filter,
                    limit=self._timeline_page_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=[self._settings.vector_name],
                )
            except (IncompatibleVisualIndexError, VisualIndexConfigurationError, VisualIndexError):
                raise
            except Exception as exc:
                raise VisualIndexError(
                    f"Failed to scroll visual timeline from collection '{self._settings.collection}'"
                ) from exc

            for record in records:
                visual = self._visual_from_stored_point(
                    cast(_StoredPoint, record),
                    identity,
                )
                timestamp_us = visual.timestamp_us
                if (
                    visual.asset_id != asset_id
                    or visual.media_type is not MediaType.VIDEO
                    or timestamp_us is None
                    or timestamp_us < start_timestamp_us
                    or timestamp_us > end_timestamp_us
                ):
                    raise IncompatibleVisualIndexError(
                        "Qdrant timeline result violates the requested asset/time window"
                    )
                yield visual

            if next_offset is None:
                break
            if next_offset == offset:
                raise VisualIndexError(
                    "Qdrant timeline scroll cursor did not advance"
                )
            offset = next_offset

    async def delete_old_revisions(
        self,
        *,
        source_path: str,
        current_asset_id: str,
    ) -> None:
        """Delete older revisions for one source path before indexing its replacement."""
        if self._identity is None:
            raise VisualIndexConfigurationError(
                "ensure_compatible() must succeed before old revisions can be deleted"
            )

        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_path",
                        match=models.MatchValue(value=source_path),
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="asset_id",
                        match=models.MatchValue(value=current_asset_id),
                    )
                ],
            )
        )
        try:
            await self._client.delete(
                self._settings.collection,
                points_selector=selector,
                wait=True,
            )
        except Exception as exc:
            raise VisualIndexError(
                f"Failed to delete old visual revisions for {source_path} "
                f"from collection '{self._settings.collection}'"
            ) from exc

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        """Validate then idempotently upsert visual candidates in bounded sequential batches."""
        identity = self._identity
        if identity is None:
            raise VisualIndexConfigurationError(
                "ensure_compatible() must succeed before visual points can be upserted"
            )

        # Validate the complete logical batch before the first network write so an identity mismatch
        # cannot leave only the compatible prefix persisted.
        for visual in visuals:
            if visual.embedding_identity != identity:
                raise VisualIndexConfigurationError(
                    "visual embedding identity does not match the identity accepted by ensure_compatible()"
                )

        batch_size = int(self._settings.upsert_batch_size)
        for chunk in batched(visuals, batch_size):
            points = [self._point_from_visual(visual) for visual in chunk]
            try:
                await self._client.upsert(
                    self._settings.collection,
                    points=points,
                    wait=True,
                )
            except Exception as exc:
                raise VisualIndexError(
                    f"Failed to upsert visual points into collection '{self._settings.collection}'"
                ) from exc

    async def close(self) -> None:
        """Close the reusable Qdrant client held by this index session."""
        try:
            await self._client.close()
        except Exception as exc:
            raise VisualIndexError("Failed to close Qdrant visual index client") from exc

    def _validate_collection(
        self,
        info: _CollectionInfo,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> str:
        """Reject vector, embedding, or dataset ownership incompatibility and return its instance ID."""
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or self._settings.vector_name not in vectors:
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}': "
                f"required named vector '{self._settings.vector_name}' is missing. "
                "Explicit reindexing is required."
            )

        vector = vectors[self._settings.vector_name]
        if vector.size != identity.dimension:
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}'. "
                f"Expected vector '{self._settings.vector_name}' size={identity.dimension}; "
                f"actual size={vector.size}. Explicit reindexing is required."
            )
        if vector.distance is not VISUAL_DISTANCE:
            actual_distance = _distance_name(vector.distance)
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}'. "
                f"Expected distance={VISUAL_DISTANCE_NAME}; actual distance={actual_distance}. "
                "Explicit reindexing is required."
            )

        actual_metadata = info.config.metadata
        if actual_metadata is None:
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}': "
                "Winston metadata is missing. Explicit reindexing is required."
            )

        expected_metadata: CollectionMetadata = {
            "winston_schema_version": WINSTON_SCHEMA_VERSION,
            "dataset_instance_id": dataset_instance_id,
            "model_id": identity.model_id,
            "dimension": identity.dimension,
            "preprocessing_version": identity.preprocessing_version,
            "vector_name": self._settings.vector_name,
            "distance": VISUAL_DISTANCE_NAME,
        }
        for field, expected in expected_metadata.items():
            if field not in actual_metadata:
                raise IncompatibleVisualIndexError(
                    f"Incompatible visual index collection '{self._settings.collection}': "
                    f"required metadata field '{field}' is missing. "
                    "Explicit reindexing is required."
                )
            actual = actual_metadata[field]
            if actual != expected:
                raise IncompatibleVisualIndexError(
                    f"Incompatible visual index collection '{self._settings.collection}': "
                    f"metadata field '{field}' expected {expected!r}, actual {actual!r}. "
                    "Explicit reindexing is required."
                )

        raw_instance_id = actual_metadata.get("index_instance_id")
        if not isinstance(raw_instance_id, str):
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}': "
                "required metadata field 'index_instance_id' is missing or invalid. "
                "Explicit reindexing is required."
            )
        try:
            return str(UUID(raw_instance_id))
        except ValueError as exc:
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}': "
                "metadata field 'index_instance_id' is not a UUID. "
                "Explicit reindexing is required."
            ) from exc

    def _dataset_instance_id(self, info: _CollectionInfo) -> str:
        """Read and canonicalize the persisted dataset UUID required by read-only search."""
        metadata = info.config.metadata
        if metadata is None:
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}': "
                "Winston metadata is missing. Explicit reindexing is required."
            )
        raw_dataset_id = metadata.get("dataset_instance_id")
        if not isinstance(raw_dataset_id, str):
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}': "
                "required metadata field 'dataset_instance_id' is missing or invalid. "
                "Explicit reindexing is required."
            )
        try:
            return str(UUID(raw_dataset_id))
        except ValueError as exc:
            raise IncompatibleVisualIndexError(
                f"Incompatible visual index collection '{self._settings.collection}': "
                "metadata field 'dataset_instance_id' is not a UUID. "
                "Explicit reindexing is required."
            ) from exc

    def _validate_query_vector(
        self,
        query_vector: VisualVector,
        identity: EmbeddingIdentity,
        *,
        limit: int,
    ) -> None:
        """Reject malformed query vectors and limits before issuing an ANN request."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise VisualIndexConfigurationError("search limit must be a positive integer")
        if query_vector.ndim != 1:
            raise VisualIndexConfigurationError("query vector must be one-dimensional")
        if query_vector.dtype != np.float32:
            raise VisualIndexConfigurationError("query vector dtype must be float32")
        if query_vector.shape[0] != identity.dimension:
            raise VisualIndexConfigurationError(
                f"query vector dimension {query_vector.shape[0]} does not match "
                f"embedding identity dimension {identity.dimension}"
            )
        if not bool(np.isfinite(query_vector).all()):
            raise VisualIndexConfigurationError("query vector values must all be finite")
        if not bool(np.any(query_vector != 0.0)):
            raise VisualIndexConfigurationError("query vector must be non-zero")

    def _validate_timeline_window(
        self,
        *,
        asset_id: str,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> None:
        """Validate one bounded semantic-refinement window before provider I/O."""
        if ASSET_ID_PATTERN.fullmatch(asset_id) is None:
            raise VisualIndexConfigurationError(
                "asset_id must be a 64-character lowercase SHA-256 hexadecimal string"
            )
        for field_name, value in (
            ("start_timestamp_us", start_timestamp_us),
            ("end_timestamp_us", end_timestamp_us),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise VisualIndexConfigurationError(
                    f"{field_name} must be an integer"
                )
            if value < 0:
                raise VisualIndexConfigurationError(
                    "timeline timestamps must be non-negative"
                )
        if start_timestamp_us > end_timestamp_us:
            raise VisualIndexConfigurationError(
                "timeline start_timestamp_us must be <= end_timestamp_us"
            )

    def _visual_from_stored_point(
        self,
        point: _StoredPoint,
        identity: EmbeddingIdentity,
    ) -> IndexedVisual:
        """Validate one Qdrant payload/vector pair and reconstruct a Winston visual."""
        if point.payload is None:
            raise IncompatibleVisualIndexError(
                "Qdrant visual result is missing its Winston visual payload"
            )
        try:
            stored = _StoredVisualPayload.model_validate(point.payload)
        except ValidationError as exc:
            raise IncompatibleVisualIndexError(
                "Qdrant visual result contains an invalid Winston visual payload"
            ) from exc

        expected_identity: tuple[tuple[str, str | int, str | int], ...] = (
            ("model_id", identity.model_id, stored.model_id),
            ("dimension", identity.dimension, stored.dimension),
            (
                "preprocessing_version",
                identity.preprocessing_version,
                stored.preprocessing_version,
            ),
        )
        for field, expected, actual in expected_identity:
            if actual != expected:
                raise IncompatibleVisualIndexError(
                    f"Qdrant visual payload field '{field}' expected {expected!r}, actual {actual!r}"
                )

        vectors = point.vector
        if not isinstance(vectors, dict) or self._settings.vector_name not in vectors:
            raise IncompatibleVisualIndexError(
                f"Qdrant visual result is missing named vector '{self._settings.vector_name}'"
            )
        raw_vector = vectors[self._settings.vector_name]
        if not isinstance(raw_vector, list):
            raise IncompatibleVisualIndexError(
                f"Qdrant named vector '{self._settings.vector_name}' is not a dense vector"
            )
        try:
            vector = np.asarray(raw_vector, dtype=np.float32)
            media_type = MediaType(stored.media_type)
            sample_kind = SampleKind(stored.sample_kind)
            region_kind = RegionKind(stored.region_kind)
            region = RegionGeometry(
                x=stored.region.x,
                y=stored.region.y,
                width=stored.region.width,
                height=stored.region.height,
                scale=stored.region.scale,
            )
            visual = IndexedVisual(
                asset_id=stored.asset_id,
                source_path=stored.source_path,
                media_type=media_type,
                sample_kind=sample_kind,
                timestamp_seconds=stored.timestamp_seconds,
                region_kind=region_kind,
                region=region,
                vector=vector,
                embedding_identity=identity,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise IncompatibleVisualIndexError(
                "Qdrant visual result cannot be reconstructed as a valid Winston visual"
            ) from exc

        if visual.timestamp_us != stored.timestamp_us:
            raise IncompatibleVisualIndexError(
                "Qdrant visual payload timestamp_us does not match timestamp_seconds"
            )
        return visual

    def _scored_visual_from_point(
        self,
        point: models.ScoredPoint,
        identity: EmbeddingIdentity,
    ) -> ScoredVisual:
        """Pair one validated Qdrant ANN score with its reconstructed Winston visual."""
        visual = self._visual_from_stored_point(cast(_StoredPoint, point), identity)
        try:
            return ScoredVisual(visual=visual, score=point.score)
        except (TypeError, ValueError) as exc:
            raise IncompatibleVisualIndexError(
                "Qdrant semantic result contains an invalid score"
            ) from exc

    def _collection_metadata(
        self,
        identity: EmbeddingIdentity,
        *,
        dataset_instance_id: str,
        index_instance_id: str,
    ) -> CollectionMetadata:
        """Build the required Winston schema-v2 ownership and compatibility metadata."""
        return {
            "winston_schema_version": WINSTON_SCHEMA_VERSION,
            "dataset_instance_id": dataset_instance_id,
            "index_instance_id": index_instance_id,
            "model_id": identity.model_id,
            "dimension": identity.dimension,
            "preprocessing_version": identity.preprocessing_version,
            "vector_name": self._settings.vector_name,
            "distance": VISUAL_DISTANCE_NAME,
        }

    def _point_from_visual(self, visual: IndexedVisual) -> models.PointStruct:
        """Map one validated Winston visual to a Qdrant point without raw media bytes."""
        return models.PointStruct(
            id=str(visual_point_id(visual)),
            vector={self._settings.vector_name: visual.vector.tolist()},
            payload=visual.model_dump(mode="json"),
        )


def _distance_name(distance: models.Distance) -> str:
    """Return a stable lowercase distance name for diagnostics."""
    return distance.value.lower()
