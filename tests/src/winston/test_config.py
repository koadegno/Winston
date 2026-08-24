from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from winston.config import (
    EmbeddingEngine,
    JinaApiEmbeddingSettings,
    JinaLocalEmbeddingSettings,
    LocalBackend,
    QdrantSettings,
    Settings,
    get_config,
)


def test_get_config_reads_unprefixed_environment_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FFMPEG_BINARY", "custom-ffmpeg")
    monkeypatch.setenv("FFPROBE_BINARY", "custom-ffprobe")
    configured = get_config()
    assert configured.data_dir == tmp_path
    assert configured.ffmpeg_binary == "custom-ffmpeg"
    assert configured.ffprobe_binary == "custom-ffprobe"


def test_get_config_returns_a_new_settings_instance() -> None:
    assert get_config() is not get_config()


def test_embedding_settings_have_safe_typed_local_defaults() -> None:
    """Winston must default to typed local embeddings without requiring an API key."""
    configured = get_config()
    assert isinstance(configured.embedding, JinaLocalEmbeddingSettings)
    assert isinstance(configured.embedding, BaseSettings)
    assert configured.embedding.engine is EmbeddingEngine.JINA_LOCAL
    assert configured.embedding.backend is LocalBackend.AUTO
    assert configured.embedding.batch_size == 4
    assert configured.embedding.model_id == "jinaai/jina-clip-v1"


def test_embedding_settings_read_nested_api_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested environment settings must select and configure the API engine as one typed object."""
    monkeypatch.setenv("EMBEDDING__ENGINE", "jina-api")
    monkeypatch.setenv("EMBEDDING__API_KEY", "secret-value")
    monkeypatch.setenv("EMBEDDING__BATCH_SIZE", "7")
    monkeypatch.setenv("EMBEDDING__RPM", "80")
    monkeypatch.setenv("EMBEDDING__TPM", "90000")
    monkeypatch.setenv("EMBEDDING__MAX_CONCURRENCY", "3")
    configured = get_config()
    assert isinstance(configured.embedding, JinaApiEmbeddingSettings)
    assert isinstance(configured.embedding, BaseSettings)
    assert configured.embedding.engine is EmbeddingEngine.JINA_API
    assert configured.embedding.api_key is not None
    assert configured.embedding.api_key.get_secret_value() == "secret-value"
    assert configured.embedding.batch_size == 7
    assert configured.embedding.rpm == 80
    assert configured.embedding.tpm == 90_000
    assert configured.embedding.max_concurrency == 3


def test_embedding_settings_read_nested_local_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested environment settings must validate local backend and batch options with enums."""
    monkeypatch.setenv("EMBEDDING__ENGINE", "jina-local")
    monkeypatch.setenv("EMBEDDING__BACKEND", "cpu")
    monkeypatch.setenv("EMBEDDING__BATCH_SIZE", "2")
    configured = get_config()
    assert isinstance(configured.embedding, JinaLocalEmbeddingSettings)
    assert configured.embedding.backend is LocalBackend.CPU
    assert configured.embedding.batch_size == 2


def test_embedding_settings_reject_unknown_engine() -> None:
    """Unknown engine identifiers must be rejected while settings are parsed."""
    with pytest.raises(ValidationError):
        Settings(embedding={"engine": "unknown-engine"})


def test_embedding_settings_reject_unknown_local_backend() -> None:
    """Unknown local backend identifiers must be rejected while settings are parsed."""
    with pytest.raises(ValidationError):
        Settings(embedding={"engine": "jina-local", "backend": "metal-magic"})


def test_qdrant_settings_have_safe_defaults() -> None:
    """Qdrant configuration must expose the Phase 1D defaults as one typed object."""
    configured = Settings()
    assert isinstance(configured.qdrant, QdrantSettings)
    assert configured.qdrant.url == "http://localhost:6333"
    assert configured.qdrant.collection == "winston_visual"
    assert configured.qdrant.vector_name == "visual"
    assert configured.qdrant.upsert_batch_size == 256


def test_qdrant_settings_read_nested_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested environment variables must configure Qdrant without free-form parsing elsewhere."""
    monkeypatch.setenv("QDRANT__URL", "http://qdrant.internal:6333")
    monkeypatch.setenv("QDRANT__COLLECTION", "custom_visual")
    monkeypatch.setenv("QDRANT__VECTOR_NAME", "image")
    monkeypatch.setenv("QDRANT__UPSERT_BATCH_SIZE", "32")
    configured = get_config()
    assert configured.qdrant.url == "http://qdrant.internal:6333"
    assert configured.qdrant.collection == "custom_visual"
    assert configured.qdrant.vector_name == "image"
    assert configured.qdrant.upsert_batch_size == 32


def test_qdrant_settings_reject_non_positive_batch_size() -> None:
    """Qdrant writes must always have a positive bounded batch size."""
    with pytest.raises(ValidationError):
        QdrantSettings(upsert_batch_size=0)


def test_indexing_visual_batch_size_defaults_to_twenty() -> None:
    """Phase 1E bounds one orchestration embedding batch to twenty visuals by default."""
    assert Settings().indexing.visual_batch_size == 20


def test_indexing_visual_batch_size_reads_nested_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented INDEXING__ override must use the existing nested settings convention."""
    monkeypatch.setenv("INDEXING__VISUAL_BATCH_SIZE", "3")

    assert Settings().indexing.visual_batch_size == 3


def test_search_settings_have_v0_defaults() -> None:
    """Phase 1F exposes explicit semantic-search defaults as one nested settings object."""
    assert Settings().model_dump().get("search") == {
        "result_limit": 10,
        "candidate_limit": 200,
        "candidate_max_limit": 2_000,
        "temporal_context_seconds": 15.0,
        "moving_average_frames": 3,
        "timeline_page_size": 256,
    }


def test_search_settings_read_nested_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEARCH__ overrides must use the same nested environment convention as other settings."""
    monkeypatch.setenv("SEARCH__RESULT_LIMIT", "7")
    monkeypatch.setenv("SEARCH__CANDIDATE_LIMIT", "80")
    monkeypatch.setenv("SEARCH__CANDIDATE_MAX_LIMIT", "640")
    monkeypatch.setenv("SEARCH__TEMPORAL_CONTEXT_SECONDS", "9.5")
    monkeypatch.setenv("SEARCH__MOVING_AVERAGE_FRAMES", "5")
    monkeypatch.setenv("SEARCH__TIMELINE_PAGE_SIZE", "64")

    assert Settings().model_dump().get("search") == {
        "result_limit": 7,
        "candidate_limit": 80,
        "candidate_max_limit": 640,
        "temporal_context_seconds": 9.5,
        "moving_average_frames": 5,
        "timeline_page_size": 64,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result_limit", 0),
        ("candidate_limit", 0),
        ("candidate_max_limit", 0),
        ("temporal_context_seconds", 0.0),
        ("timeline_page_size", 0),
    ],
)
def test_search_settings_reject_non_positive_values(field: str, value: int | float) -> None:
    """Search limits and temporal context must remain strictly positive."""
    with pytest.raises(ValidationError):
        Settings(search={field: value})


def test_search_settings_reject_candidate_max_below_candidate_limit() -> None:
    """The overfetch maximum must be a true upper bound above the initial candidate budget."""
    with pytest.raises(ValidationError):
        Settings(search={"candidate_limit": 20, "candidate_max_limit": 19})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_search_settings_reject_non_finite_temporal_context(value: float) -> None:
    """Temporal context cannot accept NaN or infinite values."""
    with pytest.raises(ValidationError):
        Settings(search={"temporal_context_seconds": value})


def test_search_settings_reject_even_moving_average_width() -> None:
    """A centered moving average requires an odd positive frame width."""
    with pytest.raises(ValidationError):
        Settings(search={"moving_average_frames": 4})
