"""Qdrant implementation of Winston's class-agnostic visual index."""

from typing import Protocol

from qdrant_client import AsyncQdrantClient, models

from winston.config import QdrantSettings
from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import (
    IncompatibleVisualIndexError,
    VisualIndexError,
)

WINSTON_SCHEMA_VERSION = 1
VISUAL_DISTANCE = models.Distance.COSINE
VISUAL_DISTANCE_NAME = "cosine"


class _CollectionParams(Protocol):
    """Structural subset of Qdrant collection params required by Winston."""

    vectors: models.VectorParams | dict[str, models.VectorParams] | None


class _CollectionConfig(Protocol):
    """Structural subset of Qdrant collection config required by Winston."""

    params: _CollectionParams
    metadata: dict[str, object] | None


class _CollectionInfo(Protocol):
    """Structural subset of Qdrant collection info required by Winston."""

    config: _CollectionConfig


class _QdrantClient(Protocol):
    """Minimal non-destructive Qdrant operations used by the visual index lifecycle."""

    async def collection_exists(self, collection_name: str) -> bool:
        """Return whether the configured collection exists."""
        ...

    async def create_collection(
        self,
        collection_name: str,
        *,
        vectors_config: dict[str, models.VectorParams],
        metadata: dict[str, object],
    ) -> bool:
        """Create a collection with named vectors and Winston metadata."""
        ...

    async def get_collection(self, collection_name: str) -> _CollectionInfo:
        """Return the collection configuration needed for compatibility validation."""
        ...

    async def close(self) -> None:
        """Release client resources."""
        ...


class QdrantVisualIndex:
    """Strict, non-destructive Qdrant backend for Winston visual embeddings."""

    def __init__(
        self,
        settings: QdrantSettings,
        *,
        client: _QdrantClient | None = None,
    ) -> None:
        """Create an index session around one reusable asynchronous Qdrant client."""
        self._settings = settings
        self._client: _QdrantClient = client or AsyncQdrantClient(url=settings.url)
        self._identity: EmbeddingIdentity | None = None

    async def ensure_compatible(self, identity: EmbeddingIdentity) -> None:
        """Create or strictly validate the configured visual collection."""
        try:
            exists = await self._client.collection_exists(self._settings.collection)
            if exists:
                info = await self._client.get_collection(self._settings.collection)
                self._validate_collection(info, identity)
            else:
                created = await self._client.create_collection(
                    self._settings.collection,
                    vectors_config={
                        self._settings.vector_name: models.VectorParams(
                            size=identity.dimension,
                            distance=VISUAL_DISTANCE,
                        )
                    },
                    metadata=self._collection_metadata(identity),
                )
                if not created:
                    raise VisualIndexError(
                        f"Qdrant did not create visual index collection '{self._settings.collection}'"
                    )
        except IncompatibleVisualIndexError:
            raise
        except VisualIndexError:
            raise
        except Exception as exc:
            raise VisualIndexError(
                f"Failed to ensure visual index collection '{self._settings.collection}'"
            ) from exc

        self._identity = identity

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
    ) -> None:
        """Reject any actual vector or Winston metadata incompatibility."""
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

        expected_metadata = self._collection_metadata(identity)
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

    def _collection_metadata(self, identity: EmbeddingIdentity) -> dict[str, object]:
        """Build the required Winston-owned collection compatibility metadata."""
        return {
            "winston_schema_version": WINSTON_SCHEMA_VERSION,
            "model_id": identity.model_id,
            "dimension": identity.dimension,
            "preprocessing_version": identity.preprocessing_version,
            "vector_name": self._settings.vector_name,
            "distance": VISUAL_DISTANCE_NAME,
        }


def _distance_name(distance: models.Distance) -> str:
    """Return a stable lowercase distance name for diagnostics."""
    return distance.value.lower()
