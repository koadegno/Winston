# Jina CLIP Embeddings Design

## Goal

Implement Winston Phase 1C as an engine-agnostic multimodal embedding subsystem with two Jina CLIP v1 engines:

- a local engine backed by the official Hugging Face/Transformers implementation;
- an optional Jina Embeddings API engine.

The rest of Winston must depend only on a shared embedding contract so another provider or local implementation can be added later without changing sampling, indexing, or search logic.

## Constraints

- Baseline semantic model: `jinaai/jina-clip-v1`.
- Vector dimension: `768` for both image and text embeddings.
- Similarity metric used downstream: cosine.
- Model/preprocessing identity must be explicit and stable enough for Qdrant compatibility checks.
- No detector classes, labels, or tracking are involved.
- Every function and method added by this phase must have a docstring. Non-obvious rate-limit, batching, device-selection, and lifecycle logic must have explanatory comments.
- Use precise types; do not introduce `Any` in the embedding implementation.
- The Jina API engine is optional. Normal local tests and Winston startup must not require an API key.

## Public abstraction

Winston exposes one structural interface:

```python
class MultimodalEmbedder(Protocol):
    @property
    def identity(self) -> EmbeddingIdentity: ...

    async def embed_images(
        self,
        images: Sequence[RGBImage],
    ) -> EmbeddingBatch: ...

    async def embed_texts(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch: ...

    async def close(self) -> None: ...
```

`RGBImage` is also structural and contains only the information the embedding layer needs:

```python
class RGBImage(Protocol):
    width: int
    height: int
    rgb24: bytes
```

`VisualRegion` from Phase 1B satisfies this contract without the embedding package importing `VisualRegion`.

The shared async interface exists because remote engines perform real asynchronous I/O. Local engines keep their synchronous model internals private so callers never branch on engine type.

## Embedding identity

Semantic compatibility is independent from how an embedding was produced.

```python
@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    model_id: str
    dimension: int
    preprocessing_version: int
```

For both Jina CLIP v1 engines:

```text
model_id = jinaai/jina-clip-v1
dimension = 768
preprocessing_version = 1
```

Engine/runtime metadata is separate:

```text
engine_id = jina-local | jina-api
backend = cuda | mlx | cpu | remote
```

A future engine using another model must have another `model_id` even if it also returns 768-dimensional vectors.

## Package structure

```text
src/winston/embeddings/
├── __init__.py
├── base.py
├── models.py
├── factory.py
├── jina/
│   ├── __init__.py
│   ├── identity.py
│   ├── preprocessing.py
│   ├── api.py
│   ├── rate_limit.py
│   └── local.py
└── backends/
    ├── __init__.py
    └── local.py
```

Each module has one responsibility:

- `base.py`: public `Protocol` contracts.
- `models.py`: engine-independent identity/result/value types.
- `factory.py`: instantiate the configured engine.
- `jina/identity.py`: canonical Jina CLIP v1 identity constants.
- `jina/preprocessing.py`: conversion of Winston RGB24 buffers to image inputs shared by Jina engines where appropriate.
- `jina/api.py`: HTTP client, batching, retries, response validation.
- `jina/rate_limit.py`: configurable RPM/TPM admission control.
- `jina/local.py`: lifecycle and batching for the local Jina model.
- `backends/local.py`: local backend/device resolution.

## Jina local engine

### Model loading

Load `jinaai/jina-clip-v1` through the official Transformers integration with `trust_remote_code=True` and use the model's official `encode_image` and `encode_text` methods.

The model is lazy-loaded on the first embedding request and cached on the engine instance. It is not reloaded per batch or per frame.

```text
create engine
    ↓
no model loaded yet
    ↓
first embed call
    ↓
load model once
    ↓
reuse for every batch
    ↓
close engine
    ↓
release references / accelerator cache where applicable
```

A single engine instance is intended to live for the complete indexing/search operation that needs it.

### Batch size

Default local batch size: `4` because the target Mac may have only 16 GB of unified memory.

The configured value is a maximum, not a guarantee. If inference raises an out-of-memory condition, retry the failed batch with progressively smaller sizes:

```text
4 -> 2 -> 1
```

Once reduced, retain the smaller effective batch size for the remainder of that engine lifetime. Do not automatically increase it again.

Do not run concurrent inference calls against the same local model by default. Batching is the primary form of parallelism for local inference.

### Backend selection

Configuration supports:

```text
auto | cuda | mlx | cpu
```

Automatic priority:

```text
CUDA -> compatible MLX adapter -> CPU
```

A backend is valid only if it can produce embeddings in the exact `jinaai/jina-clip-v1` semantic space.

As of this design, the official Jina CLIP v1 distribution supports PyTorch/Transformers and ONNX, while no official Jina CLIP v1 MLX implementation is published. Therefore `auto` may skip MLX and fall back to CPU until a compatible MLX adapter is implemented. Winston must never silently substitute another model merely because it has an MLX implementation.

Explicitly requesting `mlx` when no compatible adapter exists fails with a clear backend-unavailable error rather than changing models.

## Jina API engine

Endpoint:

```text
POST https://api.jina.ai/v1/embeddings
```

The API engine is opt-in. `JINA_API_KEY` may be absent from normal environments. The key is validated only when the Jina API engine is instantiated/used.

### Configuration

```text
EMBEDDING_ENGINE=jina-api
JINA_API_KEY=<secret>
JINA_API_BATCH_SIZE=16
JINA_API_RPM=100
JINA_API_TPM=100000
JINA_API_MAX_CONCURRENCY=2
```

RPM/TPM are configuration values because different accounts may have different limits. The free-key values above are defaults matching the limits currently documented by Jina, not hard-coded assumptions in scheduling logic.

### Async HTTP and concurrency

Use one reusable async HTTP client/session for the engine lifetime. Requests may run concurrently up to `JINA_API_MAX_CONCURRENCY`, but every request must first pass both the RPM and TPM limiters.

Do not create a new HTTP client for every batch.

### RPM/TPM limiter

Admission requires capacity in both rolling budgets:

```text
batch
  ├── RPM capacity
  └── TPM capacity
        ↓
     request
```

The configured API batch size is a maximum. If the current TPM budget cannot admit the full batch, the scheduler may reduce the batch before waiting. If even one item cannot be admitted, wait until budget becomes available.

For text, estimate token cost conservatively before sending. For image requests, preprocessing/request representation must allow a deterministic conservative cost estimate. After a successful response, reconcile the reservation against Jina's returned usage information when available.

On HTTP 429, honor `Retry-After` when supplied; otherwise use bounded exponential backoff. Retries remain subject to RPM/TPM admission and never bypass the limiter.

### Image API input

Do not upload original 2688x1512 source frames blindly. The API engine prepares the image using the Jina CLIP v1 canonical 224x224 preprocessing semantics before encoding it for the request, avoiding token usage that scales with the full surveillance resolution.

The local engine delegates preprocessing/tokenization to the official Jina model methods so local and remote implementations remain aligned with Jina CLIP v1 semantics.

## Engine factory and settings

Default engine:

```text
EMBEDDING_ENGINE=jina-local
```

Local defaults:

```text
EMBEDDING_LOCAL_BACKEND=auto
EMBEDDING_LOCAL_BATCH_SIZE=4
JINA_CLIP_MODEL_ID=jinaai/jina-clip-v1
```

API defaults:

```text
JINA_API_BATCH_SIZE=16
JINA_API_RPM=100
JINA_API_TPM=100000
JINA_API_MAX_CONCURRENCY=2
```

`factory.py` maps configuration to an implementation exactly once. Sampling/index/search code receives a `MultimodalEmbedder` and does not inspect engine names.

## Error handling

Expose embedding-specific exceptions with clear causes for:

- unsupported/unavailable local backend;
- invalid RGB24 image geometry/buffer;
- local model load failure;
- local inference failure after batch size reaches 1;
- missing API key when the API engine is selected;
- malformed API response or wrong vector dimension;
- HTTP/API failures after bounded retries.

A returned embedding batch must preserve input order and contain exactly one 768-dimensional vector per input.

## Testing strategy

### Fast unit tests

Run without network, model download, GPU, or API key:

- public `MultimodalEmbedder` contract and identity types;
- `VisualRegion` compatibility through structural RGB image typing;
- engine factory selection;
- API key is not required for the local engine;
- API engine fails clearly only when selected without a key;
- local model loader is invoked once and reused across batches;
- local configured batch size is honored;
- OOM reduces batch size `4 -> 2 -> 1` and keeps the reduced size;
- output order, dtype, and `(N, 768)` shape validation;
- backend resolution priority and explicit unavailable-backend errors;
- RPM admission control;
- TPM admission control;
- adaptive API batch reduction;
- `429` / `Retry-After` behavior;
- API response usage reconciliation;
- one reusable HTTP client per API engine lifetime.

Use small fakes at the engine boundary; tests assert Winston behavior rather than re-testing Transformers or HTTP libraries.

### Local real-model integration test

An explicit integration test downloads/loads the real `jinaai/jina-clip-v1` model and verifies:

- image vectors have 768 dimensions;
- text vectors have 768 dimensions;
- batched image inference works;
- a simple related text/image pair has higher cosine similarity than an unrelated pair;
- multiple embedding calls reuse the same loaded model instance.

This test is separate from the fast default suite because model download and inference are expensive.

### API integration test

An explicit API integration test runs only when `JINA_API_KEY` is set. It verifies real request/response shape and semantic behavior but is skipped otherwise. No CI/default test should fail because the project has no Jina API key.

## Dependencies

Expected runtime dependencies for this phase include:

- `numpy`;
- `pillow`;
- `torch`;
- `transformers`;
- `timm`;
- `einops`;
- an async HTTP client such as `httpx`.

Avoid optional accelerator packages that are not required for the baseline to run on CPU/CUDA.

## Current upstream references

- Jina Embeddings API and rate-limit documentation: https://jina.ai/embeddings/
- Jina CLIP v1 model: https://huggingface.co/jinaai/jina-clip-v1
- Jina CLIP implementation requirements: https://huggingface.co/jinaai/jina-clip-implementation
- Jina CLIP v1 preprocessing configuration: https://huggingface.co/jinaai/jina-clip-v1/blob/main/preprocessor_config.json

## PR boundary

Phase 1C remains one focused PR and closes issue #7. It may add the engine abstraction, both Jina engines, settings, dependencies, and tests required to prove the shared contract. It must not add Qdrant indexing or search behavior from later phases.
