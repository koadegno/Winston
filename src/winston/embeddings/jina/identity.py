"""Canonical semantic identity for Jina CLIP v1 embeddings."""

from winston.embeddings.models import EmbeddingIdentity

JINA_CLIP_V1_MODEL_ID = "jinaai/jina-clip-v1"
JINA_CLIP_V1_DIMENSION = 768
JINA_CLIP_V1_PREPROCESSING_VERSION = 1
JINA_CLIP_V1_IMAGE_SIZE = 224

JINA_CLIP_V1_IDENTITY = EmbeddingIdentity(
    model_id=JINA_CLIP_V1_MODEL_ID,
    dimension=JINA_CLIP_V1_DIMENSION,
    preprocessing_version=JINA_CLIP_V1_PREPROCESSING_VERSION,
)
