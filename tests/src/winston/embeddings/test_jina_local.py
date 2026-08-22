import asyncio
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pytest
from PIL import Image

from winston.embeddings.jina.local import JinaLocalEmbedder
from winston.embeddings.models import EmbeddingInferenceError


@dataclass(frozen=True, slots=True)
class TinyImage:
    """Small RGB24 fixture for local embedding tests."""

    width: int = 2
    height: int = 2
    rgb24: bytes = bytes((255, 0, 0) * 4)


class FakeJinaModel:
    """Small deterministic fake implementing the Jina model methods Winston calls."""

    def __init__(self, *, oom_above: int | None = None) -> None:
        """Initialize call tracking and an optional simulated OOM batch threshold."""
        self.oom_above = oom_above
        self.image_batch_sizes: list[int] = []
        self.text_batch_sizes: list[int] = []
        self.device: str | None = None

    def to(self, device: str) -> "FakeJinaModel":
        """Record the selected device and return the fake model."""
        self.device = device
        return self

    def encode_image(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Return deterministic image vectors or simulate an OOM for oversized batches."""
        del show_progress_bar, convert_to_numpy, normalize_embeddings
        self.image_batch_sizes.append(batch_size)
        if self.oom_above is not None and batch_size > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        return np.stack([np.full(768, index + 1, dtype=np.float32) for index, _ in enumerate(images)])

    def encode_text(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Return deterministic text vectors while recording the chosen batch size."""
        del show_progress_bar, convert_to_numpy, normalize_embeddings
        self.text_batch_sizes.append(batch_size)
        return np.stack([np.full(768, len(text), dtype=np.float32) for text in texts])


def test_local_model_is_lazy_loaded_once_and_reused_across_modalities() -> None:
    """One local model instance must serve the complete engine lifetime."""
    model = FakeJinaModel()
    loads: list[tuple[str, str]] = []

    def loader(model_id: str, device: str) -> FakeJinaModel:
        loads.append((model_id, device))
        return model.to(device)

    async def exercise() -> None:
        embedder = JinaLocalEmbedder(backend="cpu", batch_size=2, model_loader=loader)
        assert loads == []
        images = await embedder.embed_images([TinyImage(), TinyImage()])
        texts = await embedder.embed_texts(["red", "green"])
        assert images.vectors.shape == (2, 768)
        assert texts.vectors.shape == (2, 768)
        assert images.vectors.dtype == np.float32
        assert texts.vectors.dtype == np.float32
        assert loads == [("jinaai/jina-clip-v1", "cpu")]
        await embedder.close()

    asyncio.run(exercise())


def test_local_engine_honors_configured_batch_size() -> None:
    """Local inference must split a larger request using the configured maximum batch size."""
    model = FakeJinaModel()

    async def exercise() -> None:
        embedder = JinaLocalEmbedder(backend="cpu", batch_size=2, model_loader=lambda _id, _device: model)
        batch = await embedder.embed_images([TinyImage(), TinyImage(), TinyImage(), TinyImage(), TinyImage()])
        assert batch.vectors.shape == (5, 768)
        assert model.image_batch_sizes == [2, 2, 1]
        await embedder.close()

    asyncio.run(exercise())


def test_local_engine_reduces_batch_after_oom_and_keeps_reduced_size() -> None:
    """An OOM must reduce 4 -> 2 and keep that safer batch size for later work."""
    model = FakeJinaModel(oom_above=2)

    async def exercise() -> None:
        embedder = JinaLocalEmbedder(backend="cpu", batch_size=4, model_loader=lambda _id, _device: model)
        first = await embedder.embed_images([TinyImage(), TinyImage(), TinyImage(), TinyImage()])
        assert first.vectors.shape == (4, 768)
        assert model.image_batch_sizes == [4, 2, 2]
        assert embedder.effective_batch_size == 2
        await embedder.embed_texts(["one", "two", "three"])
        assert model.text_batch_sizes == [2, 1]
        await embedder.close()

    asyncio.run(exercise())


def test_local_engine_raises_when_batch_one_still_ooms() -> None:
    """Persistent OOM at batch size one must fail explicitly instead of retrying forever."""
    model = FakeJinaModel(oom_above=0)

    async def exercise() -> None:
        embedder = JinaLocalEmbedder(backend="cpu", batch_size=2, model_loader=lambda _id, _device: model)
        with pytest.raises(EmbeddingInferenceError, match="out of memory"):
            await embedder.embed_images([TinyImage()])
        await embedder.close()

    asyncio.run(exercise())


def test_closed_local_engine_rejects_new_work() -> None:
    """A closed engine must not silently reload its model and start a new session."""
    model = FakeJinaModel()

    async def exercise() -> None:
        embedder = JinaLocalEmbedder(backend="cpu", model_loader=lambda _id, _device: model)
        await embedder.close()
        with pytest.raises(EmbeddingInferenceError, match="closed"):
            await embedder.embed_texts(["query"])

    asyncio.run(exercise())


class BombImage:
    """RGB image fixture that fails as soon as its pixel buffer is materialized."""

    width = 2
    height = 2

    @property
    def rgb24(self) -> bytes:
        """Raise to reveal whether later batches were preprocessed eagerly."""
        raise RuntimeError("second image materialized")


def test_local_engine_materializes_only_the_current_image_batch() -> None:
    """Local preprocessing must not duplicate every source image in RAM before batching."""
    model = FakeJinaModel()

    async def exercise() -> None:
        """Prove batch one reaches the model before batch two pixels are accessed."""
        embedder = JinaLocalEmbedder(backend="cpu", batch_size=1, model_loader=lambda _id, _device: model)
        with pytest.raises(RuntimeError, match="second image materialized"):
            await embedder.embed_images([TinyImage(), BombImage()])
        assert model.image_batch_sizes == [1]
        await embedder.close()

    asyncio.run(exercise())
