"""Local accelerator selection for embedding engines."""

from collections.abc import Callable
from dataclasses import dataclass

from winston.config import LocalBackend
from winston.embeddings.models import EmbeddingConfigurationError


class LocalBackendUnavailableError(EmbeddingConfigurationError):
    """Raised when an explicitly requested local backend cannot run the configured model."""


@dataclass(frozen=True, slots=True)
class ResolvedLocalBackend:
    """Concrete local backend selected for one embedding engine instance."""

    backend: LocalBackend
    device: str


def cuda_is_available() -> bool:
    """Return whether PyTorch reports a usable CUDA accelerator."""
    # Import lazily so reading configuration or using the API engine does not eagerly initialize Torch.
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def jina_clip_v1_mlx_is_available() -> bool:
    """Return whether a compatible Jina CLIP v1 MLX adapter is installed."""
    # No official/compatible Jina CLIP v1 MLX adapter is currently shipped by Winston.
    # Keep this probe explicit so a future adapter can be added without changing selection order.
    return False


def resolve_local_backend(
    requested: LocalBackend | str,
    *,
    cuda_available: Callable[[], bool] = cuda_is_available,
    mlx_available: Callable[[], bool] = jina_clip_v1_mlx_is_available,
) -> ResolvedLocalBackend:
    """Resolve a backend preference using CUDA, compatible MLX, then CPU in automatic mode."""
    try:
        backend = requested if isinstance(requested, LocalBackend) else LocalBackend(requested)
    except ValueError as exc:
        raise LocalBackendUnavailableError(f"Unknown local embedding backend: {requested!r}") from exc

    if backend is LocalBackend.AUTO:
        if cuda_available():
            return ResolvedLocalBackend(LocalBackend.CUDA, "cuda")
        if mlx_available():
            return ResolvedLocalBackend(LocalBackend.MLX, "mlx")
        return ResolvedLocalBackend(LocalBackend.CPU, "cpu")

    if backend is LocalBackend.CUDA:
        if not cuda_available():
            raise LocalBackendUnavailableError("CUDA was requested but is not available")
        return ResolvedLocalBackend(LocalBackend.CUDA, "cuda")

    if backend is LocalBackend.MLX:
        if not mlx_available():
            raise LocalBackendUnavailableError(
                "MLX was requested but no compatible Jina CLIP v1 MLX adapter is available"
            )
        return ResolvedLocalBackend(LocalBackend.MLX, "mlx")

    return ResolvedLocalBackend(LocalBackend.CPU, "cpu")
