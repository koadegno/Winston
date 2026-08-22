"""Public embedding interfaces, value types, and engine factory."""

from winston.embeddings.base import MultimodalEmbedder, RGBImage
from winston.embeddings.factory import create_embedder
from winston.embeddings.models import (
    EmbeddingBatch,
    EmbeddingConfigurationError,
    EmbeddingEngine,
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingInferenceError,
    EmbeddingResponseError,
)

__all__ = [
    "EmbeddingBatch",
    "EmbeddingConfigurationError",
    "EmbeddingEngine",
    "EmbeddingError",
    "EmbeddingIdentity",
    "EmbeddingInferenceError",
    "EmbeddingResponseError",
    "MultimodalEmbedder",
    "RGBImage",
    "create_embedder",
]
