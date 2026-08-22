# Phase 1D — Qdrant Visual Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #8 by adding deterministic asset/point identities, a strict Winston-owned visual-index contract, and a Qdrant 1.18.x implementation with idempotent batched upserts and real integration validation.

**Architecture:** Winston owns the public index protocol and domain models. `QdrantVisualIndex` is the V0 backend implementation behind that contract, using `AsyncQdrantClient`, native collection metadata, named vector `visual`, Cosine distance, strict compatibility checks, and deterministic UUIDv5 point IDs. Asset identity remains a synchronous ingestion concern and is derived from canonical relative path + size + nanosecond mtime.

**Tech Stack:** Python 3.13+, NumPy, Pydantic Settings, qdrant-client 1.18.x, Qdrant server 1.18.2, pytest, pytest-asyncio, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-22-qdrant-visual-index-design.md`

## Global Constraints

- Implement only GitHub issue #8 / Phase 1D; do not add the end-to-end indexing pipeline, CLI indexing command, Qdrant search, grouping/reranking, progress jobs, or automatic reindexing.
- Keep Qdrant server pinned to `qdrant/qdrant:v1.18.2` in `compose.yml`.
- Add `qdrant-client>=1.18,<1.19` to project dependencies.
- Default collection: `winston_visual`.
- Default named vector: `visual`.
- Jina CLIP v1 vector dimension: 768.
- Distance metric: `Cosine`.
- Collection metadata must include `winston_schema_version=1`, `model_id`, `dimension`, `preprocessing_version`, `vector_name`, and `distance`.
- Existing collections without complete matching Winston metadata are incompatible.
- Existing collections with incompatible actual vector configuration are incompatible even if metadata claims compatibility.
- Never drop, clear, recreate, rename, or silently adopt an incompatible collection.
- `asset_id` is SHA-256 over versioned canonical JSON containing relative source path, file size, and `mtime_ns`.
- Source paths are relative to the explicit index root and serialized with POSIX separators.
- Sources outside the resolved index root are rejected.
- Point IDs are UUIDv5 using `uuid.NAMESPACE_URL` and versioned canonical JSON.
- Point identity includes asset, sample, integer-microsecond timestamp, exact region geometry, and embedding identity; `scale` is intentionally excluded.
- Convert timestamp seconds to integer microseconds with decimal `ROUND_HALF_UP` semantics.
- Raw image/video bytes never enter Qdrant payloads.
- `upsert()` requires a prior successful `ensure_compatible()` call.
- Reject a mixed/incompatible embedding-identity batch before the first network write.
- Upserts are sequential and bounded; default batch size is 256.
- Use `wait=True` for Qdrant writes.
- Reuse one async Qdrant client per `QdrantVisualIndex` lifetime and close it explicitly.
- Do not use `Any` in Winston code added by this phase.
- Use `Protocol`, not ABCs, for contracts.
- Every new function/method gets a docstring; non-obvious invariants get concise comments.
- Keep synchronous filesystem/identity work synchronous; use async only for Qdrant I/O.
- The implementation PR stays draft until user review and its body contains `Closes #8`.

---

## File Structure

### Create

- `src/winston/ingest/identity.py` — canonical source path + deterministic asset identity.
- `src/winston/index/base.py` — public `VisualIndex` protocol.
- `src/winston/index/models.py` — index enums/value types/errors and validation.
- `src/winston/index/identity.py` — timestamp canonicalization + deterministic UUIDv5 point identity.
- `src/winston/index/qdrant.py` — Qdrant collection lifecycle, compatibility validation, mapping, and batched upsert.
- `tests/src/winston/ingest/test_identity.py` — asset identity unit tests.
- `tests/src/winston/index/test_models.py` — domain validation tests.
- `tests/src/winston/index/test_identity.py` — UUID/timestamp identity tests.
- `tests/src/winston/index/test_qdrant.py` — Qdrant adapter unit tests using a typed fake client.
- `tests/integration/test_qdrant_visual_index.py` — real Qdrant 1.18.2 acceptance tests.
- `.github/workflows/qdrant-index.yml` — focused Qdrant integration CI without installing the Torch/CUDA stack.

### Modify

- `src/winston/config.py` — add nested `QdrantSettings` and `Settings.qdrant`.
- `src/winston/index/__init__.py` — export Winston-owned public index types only.
- `tests/src/winston/test_config.py` — Qdrant defaults/env validation.
- `pyproject.toml` — add qdrant-client 1.18.x dependency.
- `uv.lock` — regenerate after dependency change.

### Delete

- `tests/src/winston/index/.gitkeep` — directory is no longer empty.

---

### Task 1: Add typed Qdrant configuration and dependency

**Files:**
- Modify: `src/winston/config.py`
- Modify: `tests/src/winston/test_config.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: existing `SettingsConfigDict(env_nested_delimiter="__")` behavior.
- Produces: `QdrantSettings` and `Settings.qdrant` used by `QdrantVisualIndex`.

- [ ] **Step 1: Add failing configuration tests**

Append tests equivalent to:

```python
from pydantic import ValidationError

from winston.config import QdrantSettings


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
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
uv run python -m pytest -q tests/src/winston/test_config.py
```

Expected: collection fails because `QdrantSettings` / `Settings.qdrant` do not exist yet.

- [ ] **Step 3: Implement `QdrantSettings`**

Add to `src/winston/config.py`:

```python
class QdrantSettings(BaseSettings):
    """Settings for Winston's Qdrant visual index backend."""

    url: str = "http://localhost:6333"
    collection: str = "winston_visual"
    vector_name: str = "visual"
    upsert_batch_size: PositiveInt = 256
```

Then add to `Settings`:

```python
qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
```

Do not add direct `os.environ` reads.

- [ ] **Step 4: Add the Qdrant client dependency and lock it**

Add to `[project].dependencies`:

```toml
"qdrant-client>=1.18,<1.19",
```

Then run:

```bash
uv lock
```

Expected: `uv.lock` resolves a qdrant-client 1.18.x release and its runtime dependencies.

- [ ] **Step 5: Run config tests**

```bash
uv run python -m pytest -q tests/src/winston/test_config.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/winston/config.py tests/src/winston/test_config.py pyproject.toml uv.lock
git commit -m "feat: add qdrant index configuration"
```

---

### Task 2: Add deterministic asset identity

**Files:**
- Create: `src/winston/ingest/identity.py`
- Create: `tests/src/winston/ingest/test_identity.py`

**Interfaces:**
- Consumes: `Path` filesystem metadata.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class AssetIdentity:
    asset_id: str
    source_path: str


def identify_asset(*, index_root: Path, source_path: Path) -> AssetIdentity: ...
```

- [ ] **Step 1: Write failing deterministic/path tests**

Cover at least:

```python
def test_identify_asset_is_stable_for_unchanged_file(tmp_path: Path) -> None:
    """The same relative path, size, and mtime must produce exactly the same asset identity."""
    root = tmp_path / "data"
    source = root / "cameras" / "cam01.mkv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"camera-data")
    os.utime(source, ns=(1_787_412_345_678_900_000, 1_787_412_345_678_900_000))

    first = identify_asset(index_root=root, source_path=source)
    second = identify_asset(index_root=root, source_path=source)

    assert first == second
    assert first.source_path == "cameras/cam01.mkv"
    assert len(first.asset_id) == 64
    assert first.asset_id == first.asset_id.lower()


def test_identify_asset_changes_when_file_metadata_changes(tmp_path: Path) -> None:
    """A changed concrete media revision must not retain the previous asset ID."""
    root = tmp_path / "data"
    source = root / "cam01.mkv"
    root.mkdir()
    source.write_bytes(b"a")
    before = identify_asset(index_root=root, source_path=source)
    source.write_bytes(b"changed-size")
    after = identify_asset(index_root=root, source_path=source)
    assert before.asset_id != after.asset_id


def test_identify_asset_rejects_source_outside_index_root(tmp_path: Path) -> None:
    """Asset identity must never silently encode an absolute path outside the dataset root."""
    root = tmp_path / "data"
    root.mkdir()
    source = tmp_path / "outside.mkv"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="inside the indexing root"):
        identify_asset(index_root=root, source_path=source)
```

Also test that a path change within the root changes the ID and that a directory source is rejected.

- [ ] **Step 2: Run the tests and verify failure**

```bash
uv run python -m pytest -q tests/src/winston/ingest/test_identity.py
```

Expected: import failure because `winston.ingest.identity` does not exist.

- [ ] **Step 3: Implement canonical source identity**

Implement `src/winston/ingest/identity.py` with this exact material shape:

```python
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

ASSET_IDENTITY_PREFIX = "winston:asset:v1:"


@dataclass(frozen=True, slots=True)
class AssetIdentity:
    """Stable identity and portable relative path for one concrete source media revision."""

    asset_id: str
    source_path: str


def identify_asset(*, index_root: Path, source_path: Path) -> AssetIdentity:
    """Return a deterministic asset ID from canonical path and filesystem revision metadata."""
    root = Path(index_root).resolve(strict=True)
    source = Path(source_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("source path must reference a file")

    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError("source path must be inside the indexing root") from exc

    relative_path = relative.as_posix()
    stat = source.stat()
    material = {
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "relative_path": relative_path,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(f"{ASSET_IDENTITY_PREFIX}{canonical}".encode("utf-8")).hexdigest()
    return AssetIdentity(asset_id=digest, source_path=relative_path)
```

The resolved-path check intentionally makes symlinks escaping the dataset root fail.

- [ ] **Step 4: Run asset identity tests**

```bash
uv run python -m pytest -q tests/src/winston/ingest/test_identity.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/winston/ingest/identity.py tests/src/winston/ingest/test_identity.py
git commit -m "feat: add deterministic asset identity"
```

---

### Task 3: Add Winston visual-index models, validation, protocol, and UUID identity

**Files:**
- Create: `src/winston/index/base.py`
- Create: `src/winston/index/models.py`
- Create: `src/winston/index/identity.py`
- Modify: `src/winston/index/__init__.py`
- Create: `tests/src/winston/index/test_models.py`
- Create: `tests/src/winston/index/test_identity.py`
- Delete: `tests/src/winston/index/.gitkeep`

**Interfaces:**
- Consumes: `MediaType`, `RegionKind`, `EmbeddingIdentity`, `numpy.float32` vectors.
- Produces:

```python
class SampleKind(StrEnum):
    IMAGE = "image"
    KEYFRAME = "keyframe"


@dataclass(frozen=True, slots=True)
class RegionGeometry:
    x: int
    y: int
    width: int
    height: int
    scale: float


@dataclass(frozen=True, slots=True)
class IndexedVisual:
    asset_id: str
    source_path: str
    media_type: MediaType
    sample_kind: SampleKind
    timestamp_seconds: float | None
    region_kind: RegionKind
    region: RegionGeometry
    vector: NDArray[np.float32]
    embedding_identity: EmbeddingIdentity


class VisualIndex(Protocol):
    async def ensure_compatible(self, identity: EmbeddingIdentity) -> None: ...
    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None: ...
    async def close(self) -> None: ...


def timestamp_to_microseconds(timestamp_seconds: float | None) -> int | None: ...
def visual_point_id(visual: IndexedVisual) -> UUID: ...
```

Errors:

```python
class VisualIndexError(RuntimeError): ...
class VisualIndexConfigurationError(VisualIndexError): ...
class IncompatibleVisualIndexError(VisualIndexError): ...
```

- [ ] **Step 1: Write failing model-validation tests**

Use a helper that creates a valid visual with `np.ones(768, dtype=np.float32)` and `EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)`.

Required checks:

```python
def test_indexed_visual_accepts_valid_video_tile() -> None:
    """A valid keyframe tile must retain exact provenance and a 1-D float32 vector."""
    visual = make_visual()
    assert visual.sample_kind is SampleKind.KEYFRAME
    assert visual.region_kind is RegionKind.TILE
    assert visual.vector.shape == (768,)


def test_indexed_visual_rejects_wrong_vector_dimension() -> None:
    """Vector size must match the embedding identity before any Qdrant I/O."""
    with pytest.raises(VisualIndexConfigurationError, match="vector dimension"):
        make_visual(vector=np.ones(767, dtype=np.float32))


def test_indexed_visual_rejects_image_with_timestamp() -> None:
    """Photos must use sample_kind=image and no timestamp."""
    with pytest.raises(VisualIndexConfigurationError, match="image samples must not have a timestamp"):
        make_image_visual(timestamp_seconds=1.0)


def test_indexed_visual_rejects_video_keyframe_without_timestamp() -> None:
    """Video keyframes must preserve a concrete source timestamp."""
    with pytest.raises(VisualIndexConfigurationError, match="keyframe samples require a timestamp"):
        make_visual(timestamp_seconds=None)
```

Also cover:
- asset ID is exactly 64 lowercase hexadecimal characters;
- `source_path` is relative POSIX form with no `..` component;
- vector is 1-D `float32` and finite;
- region x/y are non-negative and width/height positive;
- `RegionKind.FULL` requires x=0, y=0, scale=1.0;
- tile scale is positive and <=1.0;
- timestamp is finite and non-negative.

- [ ] **Step 2: Write failing timestamp/UUID tests**

Required examples:

```python
def test_timestamp_to_microseconds_uses_half_up_rounding() -> None:
    """Point identity must not depend on binary float string formatting or banker's rounding."""
    assert timestamp_to_microseconds(0.0000005) == 1
    assert timestamp_to_microseconds(123.456) == 123_456_000
    assert timestamp_to_microseconds(None) is None


def test_visual_point_id_is_deterministic() -> None:
    """The same logical visual candidate must always map to the same UUIDv5."""
    visual = make_visual()
    assert visual_point_id(visual) == visual_point_id(visual)
    assert visual_point_id(visual).version == 5


def test_visual_point_id_changes_with_timestamp_region_or_embedding_identity() -> None:
    """Every semantic identity component must affect the deterministic point ID."""
    base = make_visual()
    assert visual_point_id(base) != visual_point_id(make_visual(timestamp_seconds=124.0))
    assert visual_point_id(base) != visual_point_id(make_visual(region=RegionGeometry(1, 0, 100, 100, 0.5)))
    assert visual_point_id(base) != visual_point_id(
        make_visual(embedding_identity=EmbeddingIdentity("other-model", 768, 1))
    )


def test_visual_point_id_ignores_scale_when_pixel_rectangle_is_identical() -> None:
    """Scale is provenance, while exact pixel geometry defines spatial identity."""
    first = make_visual(region=RegionGeometry(10, 20, 100, 80, 0.5))
    second = make_visual(region=RegionGeometry(10, 20, 100, 80, 0.6))
    assert visual_point_id(first) == visual_point_id(second)
```

- [ ] **Step 3: Run focused tests and verify failure**

```bash
uv run python -m pytest -q tests/src/winston/index/test_models.py tests/src/winston/index/test_identity.py
```

Expected: imports fail because the index models/identity are not implemented.

- [ ] **Step 4: Implement errors, enums, geometry, and `IndexedVisual` validation**

Use dataclasses and `StrEnum`, matching the existing codebase. Validation belongs in `IndexedVisual.__post_init__()` / `RegionGeometry.__post_init__()` and raises `VisualIndexConfigurationError` with actionable messages.

Do not copy the NumPy vector just to enforce frozen dataclass semantics; validate shape/dtype/finiteness without doubling memory.

- [ ] **Step 5: Implement decimal timestamp canonicalization**

Use:

```python
from decimal import Decimal, ROUND_HALF_UP

MICROSECONDS_PER_SECOND = Decimal(1_000_000)


def timestamp_to_microseconds(timestamp_seconds: float | None) -> int | None:
    """Convert a non-negative finite timestamp to deterministic integer microseconds."""
    if timestamp_seconds is None:
        return None
    value = Decimal(str(timestamp_seconds)) * MICROSECONDS_PER_SECOND
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
```

The model validation guarantees the input is finite/non-negative before point-ID generation.

- [ ] **Step 6: Implement canonical UUIDv5 identity**

Use exactly these fields in the canonical JSON:

```python
material = {
    "asset_id": visual.asset_id,
    "dimension": visual.embedding_identity.dimension,
    "height": visual.region.height,
    "model_id": visual.embedding_identity.model_id,
    "preprocessing_version": visual.embedding_identity.preprocessing_version,
    "region_kind": visual.region_kind.value,
    "sample_kind": visual.sample_kind.value,
    "timestamp_us": timestamp_to_microseconds(visual.timestamp_seconds),
    "width": visual.region.width,
    "x": visual.region.x,
    "y": visual.region.y,
}
canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
return uuid.uuid5(uuid.NAMESPACE_URL, f"winston:visual:v1:{canonical}")
```

Do not include `scale`.

- [ ] **Step 7: Implement the public `VisualIndex` protocol and exports**

`src/winston/index/base.py`:

```python
from collections.abc import Sequence
from typing import Protocol

from winston.embeddings.models import EmbeddingIdentity
from winston.index.models import IndexedVisual


class VisualIndex(Protocol):
    """Storage-agnostic async contract for Winston visual embeddings."""

    async def ensure_compatible(self, identity: EmbeddingIdentity) -> None:
        """Create or validate storage for exactly one embedding identity."""
        ...

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        """Idempotently persist already-embedded visual candidates."""
        ...

    async def close(self) -> None:
        """Release storage resources held by the index session."""
        ...
```

Export only Winston-owned types from `src/winston/index/__init__.py`; do not re-export qdrant-client models.

- [ ] **Step 8: Delete the now-unneeded index test `.gitkeep` and run tests**

```bash
rm tests/src/winston/index/.gitkeep
uv run python -m pytest -q tests/src/winston/index/test_models.py tests/src/winston/index/test_identity.py
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/winston/index tests/src/winston/index
git commit -m "feat: add visual index domain contracts"
```

---

### Task 4: Implement strict Qdrant collection lifecycle and compatibility checks

**Files:**
- Create: `src/winston/index/qdrant.py`
- Create: `tests/src/winston/index/test_qdrant.py`

**Interfaces:**
- Consumes: `QdrantSettings`, `EmbeddingIdentity`, Qdrant `VectorParams` / `Distance` / collection info.
- Produces:

```python
class QdrantVisualIndex:
    def __init__(self, settings: QdrantSettings, client: _QdrantClient | None = None) -> None: ...
    async def ensure_compatible(self, identity: EmbeddingIdentity) -> None: ...
    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None: ...
    async def close(self) -> None: ...
```

The private `_QdrantClient` protocol exposes only methods Phase 1D uses: `collection_exists`, `create_collection`, `get_collection`, `upsert`, and `close`. It deliberately has no delete/recreate method.

- [ ] **Step 1: Build a typed fake client in the unit test file**

The fake must record:

```text
created_collection_name
created_vectors_config
created_metadata
upsert_calls
closed
```

and expose a configurable fake collection-info view containing:

```text
config.params.vectors
config.metadata
```

Use real qdrant `VectorParams` / `Distance` instances for vector config so enum/size behavior matches the SDK.

- [ ] **Step 2: Write failing collection-creation tests**

Required assertion shape:

```python
@pytest.mark.asyncio
async def test_ensure_compatible_creates_missing_collection_with_metadata() -> None:
    """A missing collection must be created with the exact Winston vector and ownership contract."""
    fake = FakeQdrantClient(collection_exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=fake)
    identity = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)

    await index.ensure_compatible(identity)

    assert fake.created_collection_name == "winston_visual"
    params = fake.created_vectors_config["visual"]
    assert params.size == 768
    assert params.distance == Distance.COSINE
    assert fake.created_metadata == {
        "winston_schema_version": 1,
        "model_id": "jinaai/jina-clip-v1",
        "dimension": 768,
        "preprocessing_version": 1,
        "vector_name": "visual",
        "distance": "cosine",
    }
```

- [ ] **Step 3: Write failing compatibility tests**

Cover:
- matching existing collection succeeds;
- missing metadata fails;
- missing one Winston metadata key fails;
- different model fails;
- different preprocessing version fails;
- actual vector size mismatch fails even if metadata says 768;
- actual vector distance mismatch fails;
- configured vector name missing from actual collection fails;
- `IncompatibleVisualIndexError` message contains expected vs actual details and says explicit reindexing is required.

Also assert the fake exposes no deletion call and the implementation never obtains/uses a destructive client method.

- [ ] **Step 4: Run focused tests and verify failure**

```bash
uv run python -m pytest -q tests/src/winston/index/test_qdrant.py -k "compatible or collection"
```

Expected: failure because `QdrantVisualIndex` does not exist.

- [ ] **Step 5: Implement collection constants and metadata builder**

Use:

```python
WINSTON_SCHEMA_VERSION = 1
VISUAL_DISTANCE = Distance.COSINE


def _collection_metadata(
    *,
    identity: EmbeddingIdentity,
    vector_name: str,
) -> dict[str, str | int]:
    """Build the complete Winston ownership/compatibility metadata contract."""
    return {
        "winston_schema_version": WINSTON_SCHEMA_VERSION,
        "model_id": identity.model_id,
        "dimension": identity.dimension,
        "preprocessing_version": identity.preprocessing_version,
        "vector_name": vector_name,
        "distance": "cosine",
    }
```

- [ ] **Step 6: Implement `ensure_compatible()`**

Exact behavior:

```text
await collection_exists(name)
  false -> create_collection(name, {vector_name: VectorParams(size=identity.dimension, distance=COSINE)}, metadata=...)
  true  -> await get_collection(name) -> strict vector + metadata validation
```

After successful creation/validation, store the accepted `EmbeddingIdentity` on the `QdrantVisualIndex` instance for later upsert validation.

Catch Qdrant/network exceptions, wrap them in `VisualIndexError`, and preserve the original exception via `raise ... from exc`. Do not wrap `IncompatibleVisualIndexError` into a less specific error.

- [ ] **Step 7: Implement reusable client lifecycle**

If no client is injected:

```python
self._client = AsyncQdrantClient(url=settings.url)
```

`close()` calls exactly:

```python
await self._client.close()
```

and is safe to call after normal use.

- [ ] **Step 8: Run collection lifecycle tests**

```bash
uv run python -m pytest -q tests/src/winston/index/test_qdrant.py -k "compatible or collection or close"
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/winston/index/qdrant.py tests/src/winston/index/test_qdrant.py
git commit -m "feat: add qdrant collection compatibility"
```

---

### Task 5: Implement payload mapping and bounded idempotent upserts

**Files:**
- Modify: `src/winston/index/qdrant.py`
- Modify: `tests/src/winston/index/test_qdrant.py`

**Interfaces:**
- Consumes: accepted session `EmbeddingIdentity`, `IndexedVisual`, `visual_point_id()`.
- Produces: named-vector `PointStruct` writes with complete provenance payload and deterministic IDs.

- [ ] **Step 1: Write failing state/identity tests**

Required cases:

```python
@pytest.mark.asyncio
async def test_upsert_requires_ensure_compatible_first() -> None:
    """Writes must not occur before Winston has established the collection contract."""
    fake = FakeQdrantClient(collection_exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=fake)
    with pytest.raises(VisualIndexConfigurationError, match="ensure_compatible"):
        await index.upsert([make_visual()])
    assert fake.upsert_calls == []


@pytest.mark.asyncio
async def test_upsert_rejects_mixed_identity_before_first_network_write() -> None:
    """One incompatible item must reject the complete supplied batch atomically at the Winston boundary."""
    fake = FakeQdrantClient(collection_exists=False)
    index = QdrantVisualIndex(QdrantSettings(), client=fake)
    accepted = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)
    await index.ensure_compatible(accepted)

    mixed = [
        make_visual(embedding_identity=accepted),
        make_visual(embedding_identity=EmbeddingIdentity("other-model", 768, 1)),
    ]
    with pytest.raises(VisualIndexConfigurationError, match="embedding identity"):
        await index.upsert(mixed)
    assert fake.upsert_calls == []
```

- [ ] **Step 2: Write failing payload/ID tests**

For one video tile, assert the generated `PointStruct` has:

```python
assert point.id == str(visual_point_id(visual))
assert point.vector == {"visual": visual.vector.tolist()}
assert point.payload == {
    "asset_id": visual.asset_id,
    "source_path": "cameras/brussels/cam01.mkv",
    "media_type": "video",
    "sample_kind": "keyframe",
    "timestamp_seconds": 123.456,
    "timestamp_us": 123_456_000,
    "region_kind": "tile",
    "region": {
        "x": 1008,
        "y": 567,
        "width": 1344,
        "height": 756,
        "scale": 0.5,
    },
    "model_id": "jinaai/jina-clip-v1",
    "dimension": 768,
    "preprocessing_version": 1,
}
```

Also test photo payload has `timestamp_seconds=None` and `timestamp_us=None` and full-frame payload preserves x=0/y=0/scale=1.0.

- [ ] **Step 3: Write failing batching tests**

With `upsert_batch_size=2` and five visuals, assert three calls with sizes `[2, 2, 1]`, in original order, each call has `wait=True`, and each call targets the configured collection.

An empty sequence after successful `ensure_compatible()` should perform zero Qdrant writes.

- [ ] **Step 4: Run focused upsert tests and verify failure**

```bash
uv run python -m pytest -q tests/src/winston/index/test_qdrant.py -k "upsert or payload or batch"
```

Expected: FAIL because the write path is not implemented.

- [ ] **Step 5: Implement complete-batch identity validation before mapping**

Perform the session-state and identity loop before the chunk loop:

```python
if self._identity is None:
    raise VisualIndexConfigurationError(
        "ensure_compatible() must succeed before upsert()"
    )

for visual in visuals:
    if visual.embedding_identity != self._identity:
        raise VisualIndexConfigurationError(
            "all visuals must match the embedding identity accepted by ensure_compatible()"
        )
```

Only after that loop may any `PointStruct` be created or any network request begin.

- [ ] **Step 6: Implement payload/point mapping**

Build one point at a time inside the current write chunk. The payload contains no `rgb24` or raw media field. Convert the NumPy vector with `.tolist()` only for the active chunk.

- [ ] **Step 7: Implement sequential bounded writes**

Use index slicing over the caller-supplied `Sequence`:

```python
batch_size = int(self._settings.upsert_batch_size)
for start in range(0, len(visuals), batch_size):
    chunk = visuals[start : start + batch_size]
    points = [self._to_point(visual) for visual in chunk]
    await self._client.upsert(
        collection_name=self._settings.collection,
        points=points,
        wait=True,
    )
```

Do not pre-build a list of Qdrant points for the full input and do not dispatch multiple batches concurrently.

- [ ] **Step 8: Run all Qdrant adapter unit tests**

```bash
uv run python -m pytest -q tests/src/winston/index/test_qdrant.py
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/winston/index/qdrant.py tests/src/winston/index/test_qdrant.py
git commit -m "feat: add idempotent qdrant visual upserts"
```

---

### Task 6: Prove behavior against real Qdrant 1.18.2

**Files:**
- Create: `tests/integration/test_qdrant_visual_index.py`
- Create: `.github/workflows/qdrant-index.yml`

**Interfaces:**
- Consumes: production `QdrantVisualIndex` and a real `qdrant/qdrant:v1.18.2` service.
- Produces: acceptance evidence for native collection metadata, named-vector configuration, idempotent upsert, payload reconstruction, and non-destructive incompatibility handling.

- [ ] **Step 1: Write the real idempotent-upsert integration test**

Use default collection `winston_visual`, production settings/model types, and a direct inspector client only for assertions:

```python
@pytest.mark.asyncio
async def test_real_qdrant_visual_index_is_idempotent() -> None:
    """Real Qdrant must keep one point when Winston upserts the same logical candidate twice."""
    settings = QdrantSettings(url="http://127.0.0.1:6333")
    identity = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)
    index = QdrantVisualIndex(settings)
    inspector = AsyncQdrantClient(url=settings.url)
    visual = make_integration_visual()

    try:
        await index.ensure_compatible(identity)
        await index.upsert([visual])
        await index.upsert([visual])

        info = await inspector.get_collection(settings.collection)
        params = info.config.params.vectors[settings.vector_name]
        assert params.size == 768
        assert params.distance == Distance.COSINE
        assert info.config.metadata["model_id"] == "jinaai/jina-clip-v1"
        assert info.config.metadata["preprocessing_version"] == 1

        count = await inspector.count(settings.collection, exact=True)
        assert count.count == 1

        records = await inspector.retrieve(
            settings.collection,
            ids=[str(visual_point_id(visual))],
            with_payload=True,
            with_vectors=False,
        )
        assert len(records) == 1
        assert records[0].payload["source_path"] == visual.source_path
        assert records[0].payload["timestamp_us"] == 123_456_000
        assert records[0].payload["region"]["x"] == visual.region.x
    finally:
        await index.close()
        await inspector.close()
```

The Qdrant service used by CI is ephemeral, so this test may use the exact default collection without adding test-only cleanup behavior to Winston.

- [ ] **Step 2: Write the real incompatible-collection integration test**

Create a uniquely named collection directly through the inspector with vector `visual`, size 512, Cosine, but Winston metadata that claims dimension 768. Then instantiate `QdrantVisualIndex` pointing at that collection and assert:

```text
ensure_compatible() -> IncompatibleVisualIndexError
collection still exists
actual vector size is still 512
```

This proves actual Qdrant configuration wins over metadata claims and Winston did not destructively migrate anything.

- [ ] **Step 3: Verify integration tests fail cleanly without a Qdrant service**

Run with no local Qdrant:

```bash
uv run python -m pytest -q tests/integration/test_qdrant_visual_index.py
```

Expected: connection failure, demonstrating these are true external integration tests rather than local mocks. They are intentionally kept outside `tests/src/winston` so the standard unit suite does not require Qdrant.

- [ ] **Step 4: Add focused GitHub Actions workflow with Qdrant service**

Create `.github/workflows/qdrant-index.yml`:

```yaml
name: Qdrant visual index

on:
  pull_request:
    paths:
      - 'src/winston/index/**'
      - 'src/winston/ingest/identity.py'
      - 'src/winston/config.py'
      - 'tests/src/winston/index/**'
      - 'tests/src/winston/ingest/test_identity.py'
      - 'tests/integration/test_qdrant_visual_index.py'
      - 'pyproject.toml'
      - 'uv.lock'
      - '.github/workflows/qdrant-index.yml'
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    services:
      qdrant:
        image: qdrant/qdrant:v1.18.2
        ports:
          - 6333:6333

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - name: Install uv
        uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0
      - name: Install focused index test dependencies
        run: |
          uv venv --python 3.13
          uv pip install --no-deps -e .
          uv pip install \
            'numpy>=2.3.0' \
            'pydantic>=2.13.4' \
            'pydantic-settings>=2.15.0' \
            'pytest>=9.1.1' \
            'pytest-asyncio>=1.4.0' \
            'qdrant-client>=1.18,<1.19'
      - name: Wait for Qdrant
        run: |
          for attempt in $(seq 1 30); do
            if curl --fail --silent http://127.0.0.1:6333/collections >/dev/null; then
              exit 0
            fi
            sleep 1
          done
          exit 1
      - name: Run Phase 1D unit tests
        run: .venv/bin/python -m pytest -q tests/src/winston/ingest/test_identity.py tests/src/winston/index tests/src/winston/test_config.py
      - name: Run real Qdrant integration tests
        run: .venv/bin/python -m pytest -q tests/integration/test_qdrant_visual_index.py
      - name: Compile Winston index code
        run: .venv/bin/python -m compileall -q src/winston
```

The deliberate `uv pip install --no-deps -e .` + focused dependency install prevents this Qdrant-only job from downloading Torch/CUDA packages that it never imports.

- [ ] **Step 5: Run local unit verification**

```bash
uv run python -m pytest -q tests/src/winston/ingest/test_identity.py tests/src/winston/index tests/src/winston/test_config.py
uv run python -m compileall -q src/winston
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_qdrant_visual_index.py .github/workflows/qdrant-index.yml
git commit -m "test: verify qdrant visual index integration"
```

- [ ] **Step 7: Push and verify the real GitHub Actions job**

Push the branch and wait for `Qdrant visual index / test` to finish. Inspect job logs if it fails. Acceptance requires both unit and real-Qdrant steps green; do not treat mocked unit tests alone as completion.

---

### Task 7: Final regression verification and draft PR

**Files:**
- No new production files expected.
- Review all Phase 1D files plus spec/plan.

**Interfaces:**
- Produces: one focused draft PR for issue #8.

- [ ] **Step 1: Run the Winston regression suite**

```bash
uv run python -m pytest -q tests/src/winston
uv run python -m compileall -q src/winston
```

Expected: all existing Winston unit tests plus new Phase 1D tests pass; compileall succeeds.

- [ ] **Step 2: Confirm the diff is in scope**

Review:

```bash
git diff main...HEAD --stat
git diff main...HEAD
```

The implementation diff may contain only:
- approved spec and implementation plan;
- Qdrant settings/dependency/lock change;
- asset identity;
- Winston index protocol/models/identity/Qdrant backend;
- Phase 1D unit/integration tests;
- focused Qdrant workflow;
- deletion of the obsolete `.gitkeep`.

Reject accidental search/CLI/pipeline/recorder refactors from this PR.

- [ ] **Step 3: Check the issue acceptance criteria one by one**

Verify evidence exists for:

```text
same visual twice -> one real Qdrant point
payload reconstructs asset/path/media/sample/timestamp/region
model/dimension/preprocessing incompatibility rejected
actual collection vector config validated independently from metadata
named vector=visual, dimension=768, metric=Cosine
no automatic destructive migration
```

- [ ] **Step 4: Create the draft PR**

Title:

```text
feat: implement Qdrant visual index
```

Body:

```markdown
## Summary

- add portable deterministic asset identities and UUIDv5 visual point IDs
- add Winston-owned visual index contracts and validated provenance models
- add strict Qdrant 1.18.x collection creation/compatibility checks using native collection metadata
- add bounded sequential idempotent upserts with complete media/region payloads
- validate the contract against a real Qdrant 1.18.2 service in focused CI

## Validation

- `python -m pytest -q tests/src/winston`
- `python -m compileall -q src/winston`
- `Qdrant visual index / test` GitHub Actions job against `qdrant/qdrant:v1.18.2`

Closes #8
```

Create it as **draft** and leave it unmerged for user review.

---

## Self-Review Checklist for the Implementer

Before calling the implementation complete:

- [ ] No `Any` was introduced in Winston-owned Phase 1D code.
- [ ] Public callers depend on `VisualIndex` / Winston models, not qdrant-client request types.
- [ ] `source_path` is relative to index root and POSIX-normalized.
- [ ] `asset_id` changes when path, size, or mtime changes.
- [ ] point ID is UUIDv5 with `uuid.NAMESPACE_URL`.
- [ ] point ID excludes `scale` but includes exact x/y/width/height.
- [ ] timestamps use decimal half-up integer microseconds for identity.
- [ ] collection metadata and actual vector config are both validated.
- [ ] existing metadata-less/incompatible collection is never adopted silently.
- [ ] no production call path uses delete/recreate collection.
- [ ] mixed embedding identity is rejected before first upsert request.
- [ ] only one Qdrant write chunk is materialized at a time.
- [ ] `wait=True` is asserted by tests.
- [ ] raw media/RGB bytes are absent from point payloads.
- [ ] real Qdrant integration proves duplicate upsert count remains 1.
- [ ] real incompatible-collection test proves no destructive rewrite.
- [ ] workflow avoids unnecessary Torch/CUDA installation.
- [ ] PR is draft and contains `Closes #8`.
