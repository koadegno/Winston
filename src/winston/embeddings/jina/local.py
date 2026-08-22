"""Local Hugging Face/Transformers engine for Jina CLIP v1."""

import asyncio
import gc
from collections.abc import Callable, Sequence
from typing import Protocol, cast

import numpy as np
from PIL import Image

from winston.embeddings.backends.local import LocalBackend, ResolvedLocalBackend, resolve_local_backend
from winston.embeddings.base import RGBImage
from winston.embeddings.jina.identity import (
    JINA_CLIP_V1_DIMENSION,
    JINA_CLIP_V1_MODEL_ID,
    JINA_CLIP_V1_PREPROCESSING_VERSION,
)
from winston.embeddings.jina.preprocessing import rgb_image_to_pil
from winston.embeddings.models import (
    EmbeddingBatch,
    EmbeddingIdentity,
    EmbeddingInferenceError,
    coerce_embedding_batch,
)


class JinaClipModel(Protocol):
    """Subset of the official Jina CLIP model API used by Winston."""

    def to(self, device: str) -> object:
        """Move the model to a concrete inference device."""
        ...

    def encode_image(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        """Encode a batch of Pillow images using the official Jina image path."""
        ...

    def encode_text(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        """Encode a batch of texts using the official Jina tokenizer/model path."""
        ...


type ModelLoader = Callable[[str, str], JinaClipModel]


def _default_model_loader(model_id: str, device: str) -> JinaClipModel:
    """Load the official Jina CLIP model lazily and move it to the selected device once."""
    # Transformers is intentionally imported only when the local engine first needs a model.
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise EmbeddingInferenceError(
            "transformers is required for the jina-local embedding engine"
        ) from exc

    try:
        model = cast(
            JinaClipModel,
            AutoModel.from_pretrained(model_id, trust_remote_code=True),
        )
        model.to(device)
    except Exception as exc:
        raise EmbeddingInferenceError(f"Failed to load local embedding model {model_id}: {exc}") from exc
    return model


def _is_out_of_memory_error(exc: BaseException) -> bool:
    """Return whether an inference exception represents an accelerator/host OOM condition."""
    # Torch has backend-specific OOM exception classes; matching the stable message also covers
    # CPU/MPS-style RuntimeError variants without importing accelerator modules eagerly.
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class JinaLocalEmbedder:
    """Lazy, reusable local Jina CLIP v1 embedding engine with adaptive batch sizing."""

    engine_id = "jina-local"

    def __init__(
        self,
        *,
        model_id: str = JINA_CLIP_V1_MODEL_ID,
        backend: LocalBackend | str = LocalBackend.AUTO,
        batch_size: int = 4,
        model_loader: ModelLoader | None = None,
    ) -> None:
        """Configure one local embedding session without loading model weights yet."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._model_id = model_id
        self._resolved_backend: ResolvedLocalBackend = resolve_local_backend(backend)
        self._effective_batch_size = batch_size
        self._model_loader = model_loader or _default_model_loader
        self._model: JinaClipModel | None = None
        self._inference_lock = asyncio.Lock()
        self._closed = False

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the semantic identity shared with the equivalent Jina API engine."""
        return EmbeddingIdentity(
            model_id=self._model_id,
            dimension=JINA_CLIP_V1_DIMENSION,
            preprocessing_version=JINA_CLIP_V1_PREPROCESSING_VERSION,
        )

    @property
    def backend(self) -> LocalBackend:
        """Return the concrete local backend selected for this engine instance."""
        return self._resolved_backend.backend

    @property
    def effective_batch_size(self) -> int:
        """Return the current safe batch size after any OOM-driven reductions."""
        return self._effective_batch_size

    def _ensure_open(self) -> None:
        """Reject work submitted after the engine session has been closed."""
        if self._closed:
            raise EmbeddingInferenceError("embedding engine is closed")

    def _ensure_model(self) -> JinaClipModel:
        """Load the model once on first use and reuse it for the engine lifetime."""
        if self._model is None:
            self._model = self._model_loader(self._model_id, self._resolved_backend.device)
        return self._model

    def _reduce_batch_after_oom(self) -> bool:
        """Halve the effective batch size after OOM and report whether a retry is possible."""
        if self._effective_batch_size <= 1:
            return False
        # Persist the lower value for every later call; stability matters more than probing upward again.
        self._effective_batch_size = max(1, self._effective_batch_size // 2)
        return True

    def _embed_images_sync(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        """Preprocess and embed only the current local image batch with OOM backoff."""
        if not images:
            return coerce_embedding_batch(
                np.empty((0, JINA_CLIP_V1_DIMENSION), dtype=np.float32),
                expected_count=0,
                expected_dimension=JINA_CLIP_V1_DIMENSION,
            )

        model = self._ensure_model()
        matrices: list[np.ndarray] = []
        offset = 0
        while offset < len(images):
            current_size = min(self._effective_batch_size, len(images) - offset)
            # Materialize Pillow images only for the active batch so large indexing jobs do not
            # duplicate every RGB24 source buffer in memory before inference starts.
            chunk = [
                rgb_image_to_pil(image)
                for image in images[offset : offset + current_size]
            ]
            try:
                raw = model.encode_image(
                    chunk,
                    batch_size=current_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception as exc:
                if _is_out_of_memory_error(exc) and self._reduce_batch_after_oom():
                    continue
                if _is_out_of_memory_error(exc):
                    raise EmbeddingInferenceError(
                        "Local image embedding failed: out of memory at batch size 1"
                    ) from exc
                raise EmbeddingInferenceError(f"Local image embedding failed: {exc}") from exc

            batch = coerce_embedding_batch(
                raw,
                expected_count=current_size,
                expected_dimension=JINA_CLIP_V1_DIMENSION,
            )
            matrices.append(batch.vectors)
            offset += current_size

        return coerce_embedding_batch(
            np.concatenate(matrices, axis=0),
            expected_count=len(images),
            expected_dimension=JINA_CLIP_V1_DIMENSION,
        )

    def _embed_texts_sync(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Run serialized local text inference in bounded chunks with OOM backoff."""
        if not texts:
            return coerce_embedding_batch(
                np.empty((0, JINA_CLIP_V1_DIMENSION), dtype=np.float32),
                expected_count=0,
                expected_dimension=JINA_CLIP_V1_DIMENSION,
            )

        model = self._ensure_model()
        matrices: list[np.ndarray] = []
        offset = 0
        while offset < len(texts):
            current_size = min(self._effective_batch_size, len(texts) - offset)
            chunk = list(texts[offset : offset + current_size])
            try:
                raw = model.encode_text(
                    chunk,
                    batch_size=current_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception as exc:
                if _is_out_of_memory_error(exc) and self._reduce_batch_after_oom():
                    continue
                if _is_out_of_memory_error(exc):
                    raise EmbeddingInferenceError(
                        "Local text embedding failed: out of memory at batch size 1"
                    ) from exc
                raise EmbeddingInferenceError(f"Local text embedding failed: {exc}") from exc

            batch = coerce_embedding_batch(
                raw,
                expected_count=current_size,
                expected_dimension=JINA_CLIP_V1_DIMENSION,
            )
            matrices.append(batch.vectors)
            offset += current_size

        return coerce_embedding_batch(
            np.concatenate(matrices, axis=0),
            expected_count=len(texts),
            expected_dimension=JINA_CLIP_V1_DIMENSION,
        )

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        """Embed RGB images without eagerly materializing pixels beyond the active batch."""
        self._ensure_open()
        # Freeze only lightweight object references before crossing into the worker thread.
        image_refs = tuple(images)
        async with self._inference_lock:
            self._ensure_open()
            return await asyncio.to_thread(self._embed_images_sync, image_refs)

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed text inputs without blocking the event loop or running concurrent model inference."""
        self._ensure_open()
        async with self._inference_lock:
            self._ensure_open()
            return await asyncio.to_thread(self._embed_texts_sync, list(texts))

    def _release_model(self) -> None:
        """Drop the model reference and release accelerator cache where applicable."""
        self._model = None
        gc.collect()
        if self._resolved_backend.backend is LocalBackend.CUDA:
            try:
                import torch
            except ImportError:
                return
            torch.cuda.empty_cache()

    async def close(self) -> None:
        """Close the local embedding session and release its cached model exactly once."""
        async with self._inference_lock:
            if self._closed:
                return
            self._closed = True
            await asyncio.to_thread(self._release_model)
