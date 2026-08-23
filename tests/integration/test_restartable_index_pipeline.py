from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image
from qdrant_client import AsyncQdrantClient

from winston.config import QdrantSettings, Settings
from winston.embeddings.base import RGBImage
from winston.embeddings.models import EmbeddingBatch, EmbeddingIdentity
from winston.index.qdrant import QdrantVisualIndex
from winston.indexing.pipeline import IndexingPipeline
from winston.sampling.keyframes import KeyframeSampler

QDRANT_URL = "http://localhost:6333"
IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)


class DeterministicFakeEmbedder:
    """Provider-free image embedder for end-to-end orchestration against real Qdrant."""

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the same embedding identity expected by the V0 visual collection."""
        return IDENTITY

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        """Return one finite non-zero float32 vector for each visual candidate."""
        return EmbeddingBatch(
            vectors=np.ones((len(images), IDENTITY.dimension), dtype=np.float32)
        )

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Reject text embedding because Phase 1E only indexes visual inputs."""
        raise AssertionError("text embedding is not used by the index pipeline")

    async def close(self) -> None:
        """Release no resources because this deterministic fake owns none."""


def _make_pipeline(
    *,
    root: Path,
    settings: Settings,
    visual_index: QdrantVisualIndex,
) -> IndexingPipeline:
    """Build the real media-to-Qdrant pipeline with only embedding inference replaced."""
    return IndexingPipeline(
        root=root,
        settings=settings,
        embedder=DeterministicFakeEmbedder(),
        visual_index=visual_index,
        frame_sampler=KeyframeSampler(),
    )


@pytest.mark.asyncio
async def test_real_qdrant_restart_and_collection_recreation(tmp_path: Path) -> None:
    """A second run skips completed media while a recreated collection forces reindexing."""
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 12), (30, 60, 90)).save(image_path, format="JPEG")

    collection = f"winston_restartable_index_{uuid4().hex}"
    qdrant_settings = QdrantSettings(url=QDRANT_URL, collection=collection)
    settings = Settings(qdrant=qdrant_settings)
    inspector = AsyncQdrantClient(url=QDRANT_URL)
    indexes: list[QdrantVisualIndex] = []

    try:
        first_index = QdrantVisualIndex(qdrant_settings)
        indexes.append(first_index)
        first = await _make_pipeline(
            root=tmp_path,
            settings=settings,
            visual_index=first_index,
        ).run()
        assert (first.indexed, first.skipped, first.failed) == (1, 0, 0)

        first_count = (await inspector.count(collection, exact=True)).count
        assert first_count > 0
        first_info = await inspector.get_collection(collection)
        assert first_info.config.metadata is not None
        first_instance_id = first_info.config.metadata["index_instance_id"]
        assert isinstance(first_instance_id, str)

        second_index = QdrantVisualIndex(qdrant_settings)
        indexes.append(second_index)
        second = await _make_pipeline(
            root=tmp_path,
            settings=settings,
            visual_index=second_index,
        ).run()
        assert (second.indexed, second.skipped, second.failed) == (0, 1, 0)
        assert (await inspector.count(collection, exact=True)).count == first_count

        await inspector.delete_collection(collection)
        assert not await inspector.collection_exists(collection)

        third_index = QdrantVisualIndex(qdrant_settings)
        indexes.append(third_index)
        third = await _make_pipeline(
            root=tmp_path,
            settings=settings,
            visual_index=third_index,
        ).run()
        assert (third.indexed, third.skipped, third.failed) == (1, 0, 0)

        third_info = await inspector.get_collection(collection)
        assert third_info.config.metadata is not None
        third_instance_id = third_info.config.metadata["index_instance_id"]
        assert isinstance(third_instance_id, str)
        assert third_instance_id != first_instance_id
        assert (await inspector.count(collection, exact=True)).count == first_count
    finally:
        for index in indexes:
            await index.close()
        if await inspector.collection_exists(collection):
            await inspector.delete_collection(collection)
        await inspector.close()
