import pytest

from winston.embeddings.backends.local import (
    LocalBackend,
    LocalBackendUnavailableError,
    resolve_local_backend,
)


def test_auto_backend_prefers_cuda_before_mlx_and_cpu() -> None:
    """Automatic local selection must prefer CUDA when it is usable."""
    resolved = resolve_local_backend(
        LocalBackend.AUTO,
        cuda_available=lambda: True,
        mlx_available=lambda: True,
    )
    assert resolved.backend is LocalBackend.CUDA
    assert resolved.device == "cuda"


def test_auto_backend_uses_compatible_mlx_when_cuda_is_unavailable() -> None:
    """MLX must be the second automatic choice when a compatible adapter exists."""
    resolved = resolve_local_backend(
        LocalBackend.AUTO,
        cuda_available=lambda: False,
        mlx_available=lambda: True,
    )
    assert resolved.backend is LocalBackend.MLX
    assert resolved.device == "mlx"


def test_auto_backend_falls_back_to_cpu_when_accelerators_are_unavailable() -> None:
    """CPU must remain the reliable baseline when CUDA and compatible MLX are absent."""
    resolved = resolve_local_backend(
        LocalBackend.AUTO,
        cuda_available=lambda: False,
        mlx_available=lambda: False,
    )
    assert resolved.backend is LocalBackend.CPU
    assert resolved.device == "cpu"


def test_explicit_unavailable_mlx_fails_instead_of_substituting_a_model() -> None:
    """Explicit MLX requests must fail when Jina CLIP v1 has no compatible adapter."""
    with pytest.raises(LocalBackendUnavailableError, match="MLX"):
        resolve_local_backend(
            LocalBackend.MLX,
            cuda_available=lambda: False,
            mlx_available=lambda: False,
        )
