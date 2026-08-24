"""Application settings for Winston."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, PositiveInt, SecretStr, field_validator, model_validator
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

    visual_batch_size: PositiveInt = 20


class SearchSettings(BaseSettings):
    """Settings that bound semantic retrieval and local temporal refinement."""

    result_limit: PositiveInt = Field(
        default=10,
        description=(
            "Default number of user-facing results returned when no explicit limit is supplied; "
            "this is output policy and does not change semantic scoring."
        ),
    )
    candidate_limit: PositiveInt = Field(
        default=200,
        description=(
            "Initial number of raw ANN visual points requested from Qdrant; this is a runtime/recall "
            "tradeoff before grouping into user-facing results."
        ),
    )
    candidate_max_limit: PositiveInt = Field(
        default=2_000,
        description=(
            "Hard maximum number of raw ANN visual points considered after iterative overfetch; "
            "this is an explicit runtime/resource bound and can cap recall when exhausted."
        ),
    )
    timeline_page_size: PositiveInt = Field(
        default=256,
        description=(
            "Number of indexed visual points requested per Qdrant timeline scroll page; this is a "
            "runtime/resource transport bound and does not change covered semantic time ranges."
        ),
    )
    temporal_context_seconds: Annotated[
        FiniteFloat,
        Field(
            gt=0,
            description=(
                "Seconds of context added before and after each video ANN seed before grouping; "
                "this is a semantic parameter that can change candidate passage extent."
            ),
        ),
    ] = 15.0
    temporal_max_window_seconds: Annotated[
        FiniteFloat,
        Field(
            gt=0,
            description=(
                "Hard maximum duration in seconds of each merged video candidate window; this is "
                "an explicit runtime/resource bound that can split long semantic neighborhoods."
            ),
        ),
    ] = 60.0
    moving_average_frames: PositiveInt = Field(
        default=3,
        description=(
            "Odd number of sampled frames used by centered moving-average smoothing; this is a "
            "semantic passage-shaping parameter."
        ),
    )

    @field_validator("moving_average_frames")
    @classmethod
    def validate_odd_moving_average_frames(cls, value: int) -> int:
        """Require an odd width so every centered moving window has one exact center frame."""
        if value % 2 == 0:
            raise ValueError("moving_average_frames must be odd")
        return value

    @model_validator(mode="after")
    def validate_search_bounds(self) -> Self:
        """Keep ANN and temporal safety limits internally consistent."""
        if self.candidate_max_limit < self.candidate_limit:
            raise ValueError("candidate_max_limit must be >= candidate_limit")
        if self.temporal_max_window_seconds < 2 * self.temporal_context_seconds:
            raise ValueError(
                "temporal_max_window_seconds must be >= 2 * temporal_context_seconds"
            )
        return self


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
    search: SearchSettings = Field(default_factory=SearchSettings)


def get_config() -> Settings:
    """Build and return a fresh Winston settings instance."""
    return Settings()