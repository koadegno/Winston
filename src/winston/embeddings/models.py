"""Engine-independent embedding value types and validation helpers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from winston.config import EmbeddingEngine


type EmbeddingMatrix = NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """Base exception for embedding subsystem failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when an embedding engine is configured inconsistently."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when an embedding provider returns an invalid result."""


class EmbeddingInferenceError(EmbeddingError):
    """Raised when local or remote inference cannot produce embeddings."""


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Semantic identity used to determine whether stored vectors are compatible."""

    model_id: str
    dimension: int
    preprocessing_version: int


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Validated two-dimensional float32 embedding matrix preserving input order."""

    vectors: EmbeddingMatrix

    @property
    def count(self) -> int:
        """Return the number of vectors contained in this batch."""
        return int(self.vectors.shape[0])

    @property
    def dimension(self) -> int:
        """Return the vector dimension of this batch."""
        return int(self.vectors.shape[1])


def coerce_embedding_batch(
    vectors: object,
    *,
    expected_count: int,
    expected_dimension: int,
) -> EmbeddingBatch:
    """Convert provider output to float32 and validate the exact expected matrix shape."""
    # Providers differ in their concrete array types; normalize once at the Winston boundary.
    matrix = np.asarray(vectors, dtype=np.float32)
    expected_shape = (expected_count, expected_dimension)
    if matrix.ndim != 2 or matrix.shape != expected_shape:
        raise ValueError(
            f"expected embedding shape {expected_shape}, got {matrix.shape}"
        )
    return EmbeddingBatch(vectors=matrix)
