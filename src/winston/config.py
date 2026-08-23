"""Application settings for Winston."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingEngine(StrEnum):
    """Embedding engines that Winston can construct."""

    JINA_LOCAL = "jina-local"
    JINA_API = "jina-api"


class LocalBackend(StrEnum):
    """Supported local embedding backend preferences."""

    AUTO = "auto"
    CUDA = "cuda"
    MLX = "mlx"
    CPU = "cpu"


class JinaLocalEmbeddingSettings(BaseSettings):
    """Settings used when embeddings are computed locally with Jina CLIP v1."""

    engine: Literal[EmbeddingEngine.JINA_LOCAL] = EmbeddingEngine.JINA_LOCAL
    backend: LocalBackend = LocalBackend.AUTO
    batch_size: PositiveInt = 4
    model_id: str = "jinaai/jina-clip-v1"


class JinaApiEmbeddingSettings(BaseSettings):
    """Settings used when embeddings are requested from the Jina Embeddings API."""

    engine: Literal[EmbeddingEngine.JINA_API] = EmbeddingEngine.JINA_API
    api_key: SecretStr | None = None
    api_url: str = "https://api.jina.ai/v1/embeddings"
    batch_size: PositiveInt = 16
    rpm: PositiveInt = 100
    tpm: PositiveInt = 100_000
    max_concurrency: PositiveInt = 2


class QdrantSettings(BaseSettings):
    """Settings for Winston's Qdrant visual index backend."""

    url: str = "http://localhost:6333"
    collection: str = "winston_visual"
    vector_name: str = "visual"
    upsert_batch_size: PositiveInt = 256


class IndexingSettings(BaseSettings):
    """Settings that bound Winston's high-level indexing orchestration."""

    visual_batch_size: PositiveInt = 9


# The engine field is the discriminator, so provider-specific settings cannot be mixed together.
type EmbeddingSettings = Annotated[
    JinaLocalEmbeddingSettings | JinaApiEmbeddingSettings,
    Field(discriminator="engine"),
]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    embedding: EmbeddingSettings = Field(default_factory=JinaLocalEmbeddingSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)


def get_config() -> Settings:
    """Build and return a fresh Winston settings instance."""
    return Settings()
