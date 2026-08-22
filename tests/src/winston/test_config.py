from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from winston.config import (
    EmbeddingEngine,
    JinaApiEmbeddingSettings,
    JinaLocalEmbeddingSettings,
    LocalBackend,
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
