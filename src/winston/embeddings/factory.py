"""Factory for constructing the configured embedding engine behind one public contract."""

from winston.config import (
    JinaApiEmbeddingSettings,
    JinaLocalEmbeddingSettings,
    Settings,
    get_config,
)
from winston.embeddings.base import MultimodalEmbedder


def create_embedder(settings: Settings | None = None) -> MultimodalEmbedder:
    """Create exactly one configured embedding engine without leaking provider details to callers."""
    configured = settings or get_config()
    embedding = configured.embedding

    if isinstance(embedding, JinaLocalEmbeddingSettings):
        # Import provider implementations lazily so choosing the API path does not initialize Torch.
        from winston.embeddings.jina.local import JinaLocalEmbedder

        return JinaLocalEmbedder(
            model_id=embedding.model_id,
            backend=embedding.backend,
            batch_size=int(embedding.batch_size),
        )

    from winston.embeddings.jina.api import JinaApiEmbedder

    api_key = embedding.api_key.get_secret_value() if embedding.api_key is not None else ""
    return JinaApiEmbedder(
        api_key=api_key,
        api_url=embedding.api_url,
        batch_size=int(embedding.batch_size),
        rpm=int(embedding.rpm),
        tpm=int(embedding.tpm),
        max_concurrency=int(embedding.max_concurrency),
    )
