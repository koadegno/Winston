"""Explicit real-provider integration tests for Winston embedding engines."""

import asyncio
import os
from dataclasses import dataclass

import numpy as np
import pytest

from winston.embeddings.jina.api import JinaApiEmbedder
from winston.embeddings.jina.local import JinaLocalEmbedder

RUN_INTEGRATION = os.getenv("WINSTON_RUN_EMBEDDING_INTEGRATION") == "1"


@dataclass(frozen=True, slots=True)
class SolidImage:
    """In-memory RGB24 solid-color image used by real semantic regression tests."""

    width: int
    height: int
    rgb24: bytes


def solid_image(red: int, green: int, blue: int, *, size: int = 224) -> SolidImage:
    """Build one square RGB24 image with a constant color."""
    return SolidImage(
        width=size,
        height=size,
        rgb24=bytes((red, green, blue)) * size * size,
    )


@pytest.mark.skipif(not RUN_INTEGRATION, reason="set WINSTON_RUN_EMBEDDING_INTEGRATION=1")
def test_real_local_jina_clip_text_image_semantics_and_batching() -> None:
    """Real Jina CLIP v1 must produce 768D batches and rank a red image above blue for a red query."""
    async def exercise() -> None:
        """Load the real model once and verify image/text semantic compatibility."""
        embedder = JinaLocalEmbedder(backend="cpu", batch_size=2)
        red = solid_image(255, 0, 0)
        blue = solid_image(0, 0, 255)
        images = await embedder.embed_images([red, blue])
        text = await embedder.embed_texts(["a solid red square"])
        assert images.vectors.shape == (2, 768)
        assert text.vectors.shape == (1, 768)
        assert images.vectors.dtype == np.float32
        assert text.vectors.dtype == np.float32
        similarities = images.vectors @ text.vectors[0]
        assert similarities[0] > similarities[1]
        await embedder.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    not RUN_INTEGRATION or not os.getenv("JINA_API_KEY"),
    reason="set WINSTON_RUN_EMBEDDING_INTEGRATION=1 and JINA_API_KEY",
)
def test_real_jina_api_returns_compatible_dimensions() -> None:
    """Optional live Jina API smoke test must return the same 768-dimensional semantic contract."""
    async def exercise() -> None:
        """Call the live API only when the user explicitly supplies a Jina key."""
        api_key = os.environ["JINA_API_KEY"]
        embedder = JinaApiEmbedder(api_key=api_key, batch_size=2)
        images = await embedder.embed_images([solid_image(255, 0, 0)])
        text = await embedder.embed_texts(["a solid red square"])
        assert images.vectors.shape == (1, 768)
        assert text.vectors.shape == (1, 768)
        await embedder.close()

    asyncio.run(exercise())
