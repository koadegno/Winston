import numpy as np
import pytest

from winston.embeddings.base import RGBImage
from winston.embeddings.models import EmbeddingBatch, EmbeddingIdentity, coerce_embedding_batch
from winston.sampling.regions import VisualRegion, RegionKind


def test_visual_region_satisfies_rgb_image_protocol() -> None:
    """Phase 1B regions must satisfy the engine-agnostic RGB image contract structurally."""
    region = VisualRegion(RegionKind.FULL, 0, 0, 2, 1, 1.0, b"\x00\x01\x02\x03\x04\x05")
    assert isinstance(region, RGBImage)


def test_embedding_identity_is_value_comparable() -> None:
    """Embedding identity must be stable enough for later index compatibility checks."""
    assert EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1) == EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)


def test_coerce_embedding_batch_returns_float32_matrix() -> None:
    """Provider outputs must be normalized to a precise float32 matrix contract."""
    batch = coerce_embedding_batch([[1.0, 2.0], [3.0, 4.0]], expected_count=2, expected_dimension=2)
    assert isinstance(batch, EmbeddingBatch)
    assert batch.vectors.dtype == np.float32
    assert batch.vectors.shape == (2, 2)


def test_coerce_embedding_batch_rejects_wrong_shape() -> None:
    """A provider returning the wrong count or vector dimension must fail explicitly."""
    with pytest.raises(ValueError, match="expected embedding shape"):
        coerce_embedding_batch([[1.0, 2.0]], expected_count=2, expected_dimension=2)
