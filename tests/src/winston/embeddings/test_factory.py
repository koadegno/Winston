import asyncio

import pytest

from winston.config import (
    JinaApiEmbeddingSettings,
    JinaLocalEmbeddingSettings,
    LocalBackend,
    Settings,
)
from winston.embeddings.factory import create_embedder
from winston.embeddings.jina.api import JinaApiEmbedder
from winston.embeddings.jina.local import JinaLocalEmbedder
from winston.embeddings.models import EmbeddingConfigurationError


def test_factory_creates_local_engine_without_api_key() -> None:
    """The default/local path must never require a Jina API key."""
    settings = Settings(
        embedding=JinaLocalEmbeddingSettings(
            backend=LocalBackend.CPU,
            batch_size=2,
        )
    )
    embedder = create_embedder(settings)
    assert isinstance(embedder, JinaLocalEmbedder)
    assert embedder.identity.model_id == "jinaai/jina-clip-v1"
    assert embedder.backend is LocalBackend.CPU
    assert embedder.effective_batch_size == 2
    asyncio.run(embedder.close())


def test_factory_creates_api_engine_when_selected_with_key() -> None:
    """The factory must construct the remote engine only when explicitly configured."""
    settings = Settings(
        embedding=JinaApiEmbeddingSettings(
            api_key="test-key",
            batch_size=7,
            rpm=80,
            tpm=90_000,
            max_concurrency=3,
        )
    )
    embedder = create_embedder(settings)
    assert isinstance(embedder, JinaApiEmbedder)
    assert embedder.identity.model_id == "jinaai/jina-clip-v1"
    asyncio.run(embedder.close())


def test_factory_rejects_api_engine_without_key() -> None:
    """Selecting jina-api without an API key must fail at engine creation time."""
    settings = Settings(embedding=JinaApiEmbeddingSettings(api_key=None))
    with pytest.raises(EmbeddingConfigurationError, match="JINA_API_KEY"):
        create_embedder(settings)
