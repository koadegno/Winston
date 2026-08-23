"""Public storage-agnostic contract for Winston visual indexes."""

from collections.abc import Sequence
from typing import Protocol

from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import IndexedVisual, VisualIndexSession


class VisualIndex(Protocol):
    """Storage-agnostic async contract for Winston visual embeddings."""

    async def ensure_compatible(
        self,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> VisualIndexSession:
        """Create or validate storage owned by one dataset and embedding identity."""
        ...

    async def delete_old_revisions(
        self,
        *,
        source_path: str,
        current_asset_id: str,
    ) -> None:
        """Delete older asset revisions at one normalized source path."""
        ...

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        """Idempotently persist already-embedded visual candidates."""
        ...

    async def close(self) -> None:
        """Release storage resources held by the index session."""
        ...
