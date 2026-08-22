"""Public storage-agnostic contract for Winston visual indexes."""

from collections.abc import Sequence
from typing import Protocol

from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import IndexedVisual


class VisualIndex(Protocol):
    """Storage-agnostic async contract for Winston visual embeddings."""

    async def ensure_compatible(self, identity: EmbeddingIdentity) -> None:
        """Create or validate storage for exactly one embedding identity."""
        ...

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        """Idempotently persist already-embedded visual candidates."""
        ...

    async def close(self) -> None:
        """Release storage resources held by the index session."""
        ...
