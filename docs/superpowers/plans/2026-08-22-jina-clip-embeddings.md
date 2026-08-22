# Jina CLIP Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Winston Phase 1C as an engine-agnostic multimodal embedding layer with both local Jina CLIP v1 inference and an optional Jina Embeddings API engine.

**Architecture:** Winston depends on a single `MultimodalEmbedder` protocol and model identity. `JinaLocalEmbedder` lazy-loads one Transformers model per engine lifetime and batches inference; `JinaApiEmbedder` reuses one async HTTP client and gates requests through configurable RPM/TPM/concurrency limits. Both produce normalized float32 vectors in the same 768-dimensional Jina CLIP v1 space.

**Tech Stack:** Python 3.13+, NumPy, Pillow, PyTorch, Transformers, timm, einops, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-jina-clip-embeddings-design.md`

## Global Constraints

- Model: `jinaai/jina-clip-v1`; dimension: `768`; downstream metric: cosine.
- Default engine: `jina-local`; Jina API is optional and must not require a key unless selected.
- Local backend policy: `CUDA -> compatible MLX adapter -> CPU`; never substitute a different model.
- Default local batch size: `4`; configured batch sizes are maxima and may be reduced after OOM.
- API defaults: batch size `16`, RPM `100`, TPM `100000`, max concurrency `2`.
- Every new function/method must have a docstring; non-obvious logic must have comments.
- Precise typing only; no new `Any` in the embedding implementation.
- No Qdrant/index/search implementation in this phase.

---

### Task 1: Public contracts, value types, settings, and factory

**Files:**
- Create: `src/winston/embeddings/base.py`
- Create: `src/winston/embeddings/models.py`
- Create: `src/winston/embeddings/factory.py`
- Modify: `src/winston/embeddings/__init__.py`
- Modify: `src/winston/config.py`
- Test: `tests/src/winston/embeddings/test_contracts.py`
- Test: `tests/src/winston/embeddings/test_factory.py`
- Modify: `tests/src/winston/test_config.py`

**Interfaces:**
- `RGBImage` protocol: `width: int`, `height: int`, `rgb24: bytes`.
- `MultimodalEmbedder` protocol: `identity`, async `embed_images`, async `embed_texts`, async `close`.
- `EmbeddingIdentity(model_id: str, dimension: int, preprocessing_version: int)`.
- `EmbeddingBatch`: contiguous `np.ndarray` float32 matrix with validated `(N, dimension)` shape.
- `EmbeddingEngine`: `jina-local | jina-api`.

- [ ] **Step 1: Write failing tests** for structural RGB image compatibility, identity equality, batch validation, config defaults/environment overrides, local/API factory selection, and missing API key only when the API engine is selected.
- [ ] **Step 2: Run the focused tests and verify RED** because the public embedding types/factory do not exist yet.
- [ ] **Step 3: Implement the minimal contracts/settings/factory** with lazy imports inside the factory so importing Winston does not eagerly import Torch/Transformers.
- [ ] **Step 4: Run focused tests and verify GREEN.**

---

### Task 2: Jina identity, RGB preprocessing, and local backend resolution

**Files:**
- Create: `src/winston/embeddings/jina/__init__.py`
- Create: `src/winston/embeddings/jina/identity.py`
- Create: `src/winston/embeddings/jina/preprocessing.py`
- Create: `src/winston/embeddings/backends/__init__.py`
- Create: `src/winston/embeddings/backends/local.py`
- Test: `tests/src/winston/embeddings/test_preprocessing.py`
- Test: `tests/src/winston/embeddings/test_local_backend.py`

**Interfaces:**
- `JINA_CLIP_V1_IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)`.
- `rgb_image_to_pil(image: RGBImage) -> PIL.Image.Image` validates geometry/buffer and preserves exact RGB bytes.
- `LocalBackend`: `auto | cuda | mlx | cpu`.
- `ResolvedLocalBackend`: backend identifier plus device string for the local engine.
- `resolve_local_backend(...)` selects CUDA first, then a registered compatible MLX capability, then CPU; explicit unavailable backends raise `LocalBackendUnavailableError`.

- [ ] **Step 1: Write failing tests** for exact RGB24 conversion, invalid buffers, auto order `cuda -> mlx -> cpu`, explicit backend selection, and explicit unavailable MLX.
- [ ] **Step 2: Run focused tests and verify RED.**
- [ ] **Step 3: Implement preprocessing and backend resolver.** MLX capability is represented explicitly but returns unavailable for Jina CLIP v1 until a compatible adapter exists.
- [ ] **Step 4: Run focused tests and verify GREEN.**

---

### Task 3: Local Jina CLIP v1 engine

**Files:**
- Create: `src/winston/embeddings/jina/local.py`
- Modify: `src/winston/embeddings/factory.py`
- Test: `tests/src/winston/embeddings/test_jina_local.py`

**Interfaces:**
- `JinaLocalEmbedder(model_id, backend, batch_size, model_loader=...)`.
- Lazy `_ensure_model()` loads with `AutoModel.from_pretrained(model_id, trust_remote_code=True)` once per engine instance.
- `embed_images` converts Winston RGB24 images to PIL and calls official `model.encode_image(..., batch_size=effective_batch_size, normalize_embeddings=True, convert_to_numpy=True, device=...)`.
- `embed_texts` calls official `model.encode_text(..., batch_size=effective_batch_size, normalize_embeddings=True, convert_to_numpy=True, device=...)`.
- Local synchronous inference is serialized per engine and invoked from the shared async API without running multiple model inferences concurrently.

- [ ] **Step 1: Write failing tests** using a small fake model/loader for load-once lifecycle, reuse across image/text calls, configured batch size, order preservation, float32 `(N, 768)` validation, and close behavior.
- [ ] **Step 2: Run focused tests and verify RED.**
- [ ] **Step 3: Implement lazy local engine.** Keep the model referenced until `close()`; do not reload per batch.
- [ ] **Step 4: Add failing OOM tests** proving `4 -> 2 -> 1` reduction and persistence of the reduced effective batch size.
- [ ] **Step 5: Implement bounded OOM retry** for Torch/CUDA-style OOM failures; at batch size 1 raise a clear `EmbeddingInferenceError`.
- [ ] **Step 6: Run focused tests and verify GREEN.**

---

### Task 4: API request encoding and configurable RPM/TPM limiter

**Files:**
- Create: `src/winston/embeddings/jina/rate_limit.py`
- Create: `src/winston/embeddings/jina/api.py`
- Modify: `src/winston/embeddings/factory.py`
- Test: `tests/src/winston/embeddings/test_rate_limit.py`
- Test: `tests/src/winston/embeddings/test_jina_api.py`

**Interfaces:**
- API endpoint: `https://api.jina.ai/v1/embeddings`.
- Images are converted to canonical 224x224 Jina request images before base64 transport so each image has a deterministic one-tile cost of 1000 tokens.
- `JinaRateLimiter(rpm, tpm, clock, sleep)` uses rolling 60-second reservations and admits only when both request and token budgets allow.
- `JinaApiEmbedder` owns one reusable `httpx.AsyncClient`, one limiter, and one concurrency semaphore for its lifetime.
- Request payload uses `model="jina-clip-v1"`, normalized float embeddings, and `{"text": ...}` / `{"image": <base64>}` inputs.

- [ ] **Step 1: Write failing limiter tests** with a fake monotonic clock for RPM blocking, TPM blocking, expiry after 60 seconds, and reservation reconciliation.
- [ ] **Step 2: Run limiter tests and verify RED.**
- [ ] **Step 3: Implement the limiter and verify GREEN.**
- [ ] **Step 4: Write failing API-engine tests** using `httpx.MockTransport` for missing key, reusable client, adaptive batch splitting under TPM, response index ordering, `usage.total_tokens` reconciliation, wrong dimensions, HTTP 429 with `Retry-After`, and bounded retry failure.
- [ ] **Step 5: Run API tests and verify RED.**
- [ ] **Step 6: Implement the API engine** with async I/O, concurrency bounds, limiter admission, retry/backoff, and exact response validation.
- [ ] **Step 7: Run API tests and verify GREEN.**

---

### Task 5: Dependencies and full fast-suite regression

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: all `tests/src/winston`

- [ ] **Step 1: Add runtime dependencies** `numpy`, `pillow`, `torch`, `transformers`, `timm`, `einops`, and `httpx` with Python-3.13-compatible version floors.
- [ ] **Step 2: Regenerate the lockfile with `uv lock`.**
- [ ] **Step 3: Run `python -m compileall -q src/winston`.**
- [ ] **Step 4: Run the complete fast test suite and require zero failures.**
- [ ] **Step 5: Inspect `src/winston/embeddings` for `Any`, undocumented functions/methods, and accidental Qdrant/search coupling.**

---

### Task 6: Real-model and optional real-API integration tests

**Files:**
- Create: `tests/src/winston/embeddings/test_jina_integration.py`
- Temporarily create for verification: `.github/workflows/jina-clip-integration.yml` (remove before final PR if used only as an ephemeral runner)

**Interfaces:**
- Real local test is opt-in via `WINSTON_RUN_EMBEDDING_INTEGRATION=1` and uses the actual `jinaai/jina-clip-v1` model.
- Real API test additionally requires `JINA_API_KEY`; otherwise it is skipped.

- [ ] **Step 1: Write integration tests** that generate simple red/blue image fixtures in memory, verify both modalities are 768D float32, verify image batching with at least two images, and assert `cos(text="a red image", red) > cos(text="a red image", blue)`.
- [ ] **Step 2: Run the default suite and verify the expensive tests are skipped by default.**
- [ ] **Step 3: Run the local real-model integration in an environment with network/model download access.** If the local container cannot reach Hugging Face, use an ephemeral GitHub Actions runner on the feature branch.
- [ ] **Step 4: Do not require or fabricate a Jina API key.** Confirm the real API test reports skipped when `JINA_API_KEY` is absent.
- [ ] **Step 5: Remove any ephemeral workflow before the final PR unless retaining it provides lasting project value.**

---

### Task 7: Final verification and PR

**Files:**
- Review all Phase 1C diffs only.

- [ ] **Step 1: Compare `main...feat/jina-clip-embeddings` and confirm no Phase 1D/Qdrant changes.**
- [ ] **Step 2: Re-run the full fast suite and compilation from the final branch state.**
- [ ] **Step 3: Verify the real local-model integration evidence from Task 6.**
- [ ] **Step 4: Open a draft PR titled `feat: add Jina CLIP embedding engines` with `Closes #7`.**
- [ ] **Step 5: Leave the branch unmerged for user review.**
