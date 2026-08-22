"""Public structural contracts for Winston embedding engines."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from winston.embeddings.models import EmbeddingBatch, EmbeddingIdentity


@runtime_checkable
class RGBImage(Protocol):
    """Minimal RGB24 image contract consumed by multimodal embedding engines."""

    width: int
    height: int
    rgb24: bytes


class MultimodalEmbedder(Protocol):
    """Engine-agnostic asynchronous contract for image and text embeddings."""

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the semantic model/preprocessing identity for produced vectors."""
        ...

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        """Embed RGB images while preserving their input order."""
        ...

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed text inputs while preserving their input order."""
        ...

    async def close(self) -> None:
        """Release engine resources held for the embedding session."""
        ...
