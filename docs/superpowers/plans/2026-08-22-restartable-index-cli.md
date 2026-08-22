# Phase 1E — Restartable Index CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `winston index <path>` as a restartable, idempotent, bounded-memory V0 indexing pipeline for videos and JPG/JPEG photos.

**Architecture:** Keep `cli.py` thin and add a dedicated `winston.indexing` orchestration layer. Persist dataset identity and asset-level completion state under `<root>/.winston`, evolve Qdrant ownership metadata to schema version 2, stream one asset at a time, batch at most 9 visual regions per embedding call, and rely on deterministic Qdrant point IDs for safe retries.

**Tech Stack:** Python 3.13+, argparse, asyncio, Pillow, Pydantic/Pydantic Settings, NumPy, FFmpeg/ffprobe, Jina CLIP embedding contracts, qdrant-client 1.18.x, pytest, pytest-asyncio, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-22-restartable-index-cli-design.md`

## Global Constraints

- `winston index [path]` defaults to `Settings.data_dir` when `path` is omitted.
- Source media are immutable for the duration of one indexing run. Do not add mutation detection.
- Process exactly one media asset at a time. Do not add inter-asset concurrency.
- Preserve streaming video behavior; never materialize all keyframes from a video.
- Add `IndexingSettings.visual_batch_size: PositiveInt = 9`.
- `visual_batch_size` is an orchestration/memory bound and is independent from provider-specific embedding batch sizes.
- One Qdrant visual collection belongs to exactly one dataset identity.
- Winston Qdrant metadata schema becomes version `2` and requires `dataset_instance_id` and `index_instance_id`.
- A schema-1 or otherwise incompatible collection is rejected. Winston must never auto-drop, clear, rename, migrate, or recreate it.
- For a changed `source_path`, delete points whose `asset_id` differs from the current `asset_id` before indexing the replacement.
- If stale-revision deletion fails, do not write the new revision during that run.
- Durable restart state is asset-level only. Do not add per-keyframe/per-region checkpoints or a generic job system.
- A failed asset is not marked complete; later assets are still attempted.
- Use `Protocol`, not `ABC`, for structural contracts. Do not introduce `typing.Any`.
- Every new function/method has a docstring; comment crash-safety, ordering, filtering, and cleanup logic where it is not obvious.
- Use async only for actual asynchronous I/O/concurrency boundaries.
- Preserve the existing deterministic `visual_point_id()` algorithm unchanged.
- Keep `qdrant-client>=1.18,<1.19` and Qdrant server `v1.18.2`.
- The implementation stays in one draft PR for issue #9 and the PR body contains `Closes #9`.

---

## File Map

```text
src/winston/
├── cli.py
├── config.py
├── sampling/
│   ├── __init__.py
│   ├── images.py                  # new: JPG/JPEG -> RGB24
│   └── models.py                  # add SampledImage
├── index/
│   ├── __init__.py
│   ├── base.py                    # evolve VisualIndex protocol
│   ├── models.py                  # add VisualIndexSession
│   └── qdrant.py                  # schema v2 + delete old revisions
└── indexing/
    ├── __init__.py                # new package
    ├── manifest.py                # dataset identity + completion journal
    ├── models.py                  # run/failure models
    └── pipeline.py                # sequential orchestration + lifecycle

tests/src/winston/
├── test_cli.py
├── test_config.py
├── sampling/test_images.py
├── index/test_qdrant.py
├── index/test_qdrant_upsert.py
└── indexing/
    ├── test_manifest.py
    └── test_pipeline.py

tests/integration/
├── test_qdrant_visual_index.py
└── test_restartable_index_pipeline.py

.github/workflows/qdrant-index.yml
```

Pillow is already in `pyproject.toml`; do not add a dependency.

---

### Task 1: Add indexing configuration and photo sampling

**Files:**
- Modify: `src/winston/config.py`
- Modify: `src/winston/sampling/models.py`
- Create: `src/winston/sampling/images.py`
- Modify: `src/winston/sampling/__init__.py`
- Modify: `tests/src/winston/test_config.py`
- Create: `tests/src/winston/sampling/test_images.py`

**Interfaces:**
- Produces `IndexingSettings.visual_batch_size: PositiveInt = 9`
- Produces `SampledImage(source_path: Path, width: int, height: int, rgb24: bytes)`
- Produces `ImageSamplingError`
- Produces `load_image(metadata: ImageMetadata) -> SampledImage`

- [ ] **Step 1: Write the failing config tests**

Add to `tests/src/winston/test_config.py`:

```python
def test_indexing_visual_batch_size_defaults_to_nine() -> None:
    """Phase 1E bounds one orchestration embedding batch to nine visuals by default."""
    assert Settings().indexing.visual_batch_size == 9


def test_indexing_visual_batch_size_reads_nested_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented INDEXING__ override must use the existing nested settings convention."""
    monkeypatch.setenv("INDEXING__VISUAL_BATCH_SIZE", "3")

    assert Settings().indexing.visual_batch_size == 3
```

Run:

```bash
uv run pytest tests/src/winston/test_config.py -q
```

Expected: FAIL because `Settings.indexing` does not exist.

- [ ] **Step 2: Implement the setting**

Add to `src/winston/config.py`:

```python
class IndexingSettings(BaseSettings):
    """Settings that bound Winston's high-level indexing orchestration."""

    visual_batch_size: PositiveInt = 9
```

Add to `Settings`:

```python
indexing: IndexingSettings = Field(default_factory=IndexingSettings)
```

Run the config test again; expected PASS.

- [ ] **Step 3: Write failing photo loader tests**

Create `tests/src/winston/sampling/test_images.py` with these concrete cases:

```python
from pathlib import Path

import pytest
from PIL import Image

from winston.ingest.models import ImageMetadata
from winston.sampling.images import ImageSamplingError, load_image


def test_load_image_decodes_rgb24(tmp_path: Path) -> None:
    """A JPEG becomes contiguous RGB24 with the probed geometry."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 3), (10, 20, 30)).save(path, format="JPEG")

    sampled = load_image(ImageMetadata(path=path, width=4, height=3))

    assert sampled.source_path == path
    assert (sampled.width, sampled.height) == (4, 3)
    assert len(sampled.rgb24) == 4 * 3 * 3


def test_load_image_converts_non_rgb_source(tmp_path: Path) -> None:
    """Pillow modes other than RGB are converted explicitly before bytes leave sampling."""
    path = tmp_path / "gray.jpg"
    Image.new("L", (2, 2), 127).save(path, format="JPEG")

    sampled = load_image(ImageMetadata(path=path, width=2, height=2))

    assert len(sampled.rgb24) == 12


def test_load_image_rejects_probe_dimension_mismatch(tmp_path: Path) -> None:
    """Decoded and probed dimensions must describe the same image."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 3)).save(path, format="JPEG")

    with pytest.raises(ImageSamplingError, match=r"photo\.jpg.*4x3.*5x3"):
        load_image(ImageMetadata(path=path, width=5, height=3))


def test_load_image_wraps_corrupt_media_with_path(tmp_path: Path) -> None:
    """A corrupt image error names the asset and preserves the Pillow cause."""
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a jpeg")

    with pytest.raises(ImageSamplingError, match="broken.jpg") as caught:
        load_image(ImageMetadata(path=path, width=1, height=1))

    assert caught.value.__cause__ is not None
```

Run:

```bash
uv run pytest tests/src/winston/sampling/test_images.py -q
```

Expected: FAIL because the module and `SampledImage` do not exist.

- [ ] **Step 4: Implement `SampledImage` and `load_image()`**

Add to `src/winston/sampling/models.py`:

```python
@dataclass(frozen=True, slots=True)
class SampledImage:
    """One decoded still image with source provenance and contiguous RGB24 bytes."""

    source_path: Path
    width: int
    height: int
    rgb24: bytes
```

Create `src/winston/sampling/images.py`:

```python
"""Decode still images into Winston's RGB24 sampling contract."""

from PIL import Image

from winston.ingest.models import ImageMetadata
from winston.sampling.models import SampledImage


class ImageSamplingError(RuntimeError):
    """Raised when a still image cannot satisfy Winston's sampled-image contract."""


def load_image(metadata: ImageMetadata) -> SampledImage:
    """Decode one image, convert to RGB, and validate it against probe metadata."""
    try:
        with Image.open(metadata.path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            if (width, height) != (metadata.width, metadata.height):
                raise ImageSamplingError(
                    f"Decoded dimensions for {metadata.path} are {width}x{height}; "
                    f"probe reported {metadata.width}x{metadata.height}"
                )
            rgb24 = rgb.tobytes()
    except ImageSamplingError:
        raise
    except OSError as exc:
        raise ImageSamplingError(f"Failed to decode image {metadata.path}: {exc}") from exc

    expected_bytes = width * height * 3
    if len(rgb24) != expected_bytes:
        raise ImageSamplingError(
            f"Decoded RGB24 buffer for {metadata.path} has {len(rgb24)} bytes; "
            f"expected {expected_bytes}"
        )

    return SampledImage(metadata.path, width, height, rgb24)
```

Export `ImageSamplingError`, `SampledImage`, and `load_image` from `src/winston/sampling/__init__.py`.

- [ ] **Step 5: Run focused tests and commit**

```bash
uv run pytest tests/src/winston/test_config.py tests/src/winston/sampling -q
git add src/winston/config.py src/winston/sampling tests/src/winston/test_config.py tests/src/winston/sampling/test_images.py
git commit -m "feat: add bounded photo sampling for indexing"
```

Expected: PASS.

---

### Task 2: Upgrade VisualIndex/Qdrant to dataset-owned schema version 2

**Files:**
- Modify: `src/winston/index/base.py`
- Modify: `src/winston/index/models.py`
- Modify: `src/winston/index/qdrant.py`
- Modify: `src/winston/index/__init__.py`
- Modify: `tests/src/winston/index/test_qdrant.py`
- Modify: `tests/src/winston/index/test_qdrant_upsert.py`

**Interfaces:**
- Add `VisualIndexSession(index_instance_id: str)`
- Change `VisualIndex.ensure_compatible(identity: EmbeddingIdentity, dataset_instance_id: str) -> VisualIndexSession`
- Add `VisualIndex.delete_old_revisions(*, source_path: str, current_asset_id: str) -> None`
- Keep `upsert()` and `close()` unchanged

- [ ] **Step 1: Write failing schema-v2 ownership tests**

In `tests/src/winston/index/test_qdrant.py`, define:

```python
DATASET_ID = "8f7ad0c0-7ab7-4ec0-9025-22f40dd70c4a"
INDEX_INSTANCE_ID = "0f27f24e-7436-4c46-9260-4a37f419b3a6"


def expected_metadata(
    identity: EmbeddingIdentity = IDENTITY,
    *,
    dataset_instance_id: str = DATASET_ID,
    index_instance_id: str = INDEX_INSTANCE_ID,
) -> dict[str, object]:
    """Return the exact Phase 1E collection metadata contract."""
    return {
        "winston_schema_version": 2,
        "dataset_instance_id": dataset_instance_id,
        "index_instance_id": index_instance_id,
        "model_id": identity.model_id,
        "dimension": identity.dimension,
        "preprocessing_version": identity.preprocessing_version,
        "vector_name": "visual",
        "distance": "cosine",
    }
```

Update successful existing-collection tests to assert:

```python
session = await index.ensure_compatible(IDENTITY, DATASET_ID)
assert session.index_instance_id == INDEX_INSTANCE_ID
```

Add tests for:

```text
- missing collection: a UUIDv4 is generated, persisted in create_collection metadata, and returned
- existing schema-2 collection: stored index_instance_id is returned
- dataset_instance_id mismatch: IncompatibleVisualIndexError
- schema version 1: IncompatibleVisualIndexError with reindex guidance
- missing index_instance_id: IncompatibleVisualIndexError
- malformed index_instance_id: IncompatibleVisualIndexError
- malformed requested dataset_instance_id: VisualIndexConfigurationError
```

Run:

```bash
uv run pytest tests/src/winston/index/test_qdrant.py -q
```

Expected: FAIL on the old Phase 1D signature/schema.

- [ ] **Step 2: Add the public session model and protocol signatures**

Add to `src/winston/index/models.py`:

```python
@dataclass(frozen=True, slots=True)
class VisualIndexSession:
    """Identity of the exact compatible visual collection used by one indexing run."""

    index_instance_id: str
```

Import/export it from `src/winston/index/__init__.py`.

Update `src/winston/index/base.py` so `VisualIndex` exposes exactly these four methods:

```text
ensure_compatible(identity, dataset_instance_id) -> VisualIndexSession
delete_old_revisions(*, source_path, current_asset_id) -> None
upsert(visuals) -> None
close() -> None
```

The protocol remains free of Qdrant SDK types.

- [ ] **Step 3: Implement schema-v2 creation and validation**

In `src/winston/index/qdrant.py`:

```python
from uuid import UUID, uuid4

WINSTON_SCHEMA_VERSION = 2
```

Change `ensure_compatible()` to this behavior:

```python
async def ensure_compatible(
    self,
    identity: EmbeddingIdentity,
    dataset_instance_id: str,
) -> VisualIndexSession:
    """Create or strictly validate one dataset-owned visual collection."""
    try:
        canonical_dataset_id = str(UUID(dataset_instance_id))
    except ValueError as exc:
        raise VisualIndexConfigurationError(
            f"dataset_instance_id must be a UUID, got {dataset_instance_id!r}"
        ) from exc

    try:
        exists = await self._client.collection_exists(self._settings.collection)
        if exists:
            info = await self._client.get_collection(self._settings.collection)
            index_instance_id = self._validate_collection(
                info,
                identity,
                canonical_dataset_id,
            )
        else:
            index_instance_id = str(uuid4())
            created = await self._client.create_collection(
                self._settings.collection,
                vectors_config={
                    self._settings.vector_name: models.VectorParams(
                        size=identity.dimension,
                        distance=VISUAL_DISTANCE,
                    )
                },
                metadata=self._collection_metadata(
                    identity,
                    dataset_instance_id=canonical_dataset_id,
                    index_instance_id=index_instance_id,
                ),
            )
            if not created:
                raise VisualIndexError(
                    f"Qdrant did not create visual index collection '{self._settings.collection}'"
                )
    except (IncompatibleVisualIndexError, VisualIndexConfigurationError, VisualIndexError):
        raise
    except Exception as exc:
        raise VisualIndexError(
            f"Failed to ensure visual index collection '{self._settings.collection}'"
        ) from exc

    self._identity = identity
    return VisualIndexSession(index_instance_id=index_instance_id)
```

Change `_validate_collection()` to return `str`. Keep all Phase 1D vector/model/dimension/preprocessing/vector-name/distance checks, require `winston_schema_version == 2`, require `dataset_instance_id == canonical_dataset_id`, and finish with:

```python
raw_instance_id = actual_metadata.get("index_instance_id")
if not isinstance(raw_instance_id, str):
    raise IncompatibleVisualIndexError(
        f"Incompatible visual index collection '{self._settings.collection}': "
        "required metadata field 'index_instance_id' is missing or invalid. "
        "Explicit reindexing is required."
    )
try:
    return str(UUID(raw_instance_id))
except ValueError as exc:
    raise IncompatibleVisualIndexError(
        f"Incompatible visual index collection '{self._settings.collection}': "
        "metadata field 'index_instance_id' is not a UUID. "
        "Explicit reindexing is required."
    ) from exc
```

Change `_collection_metadata()` to require keyword-only `dataset_instance_id` and `index_instance_id` and include both values plus schema version 2.

- [ ] **Step 4: Write failing stale-revision deletion tests**

Extend the fake Qdrant client with a captured `delete()` call and add a test that inspects the resulting `models.FilterSelector`.

The required selector is exactly:

```python
models.FilterSelector(
    filter=models.Filter(
        must=[
            models.FieldCondition(
                key="source_path",
                match=models.MatchValue(value="cameras/cam01.mkv"),
            )
        ],
        must_not=[
            models.FieldCondition(
                key="asset_id",
                match=models.MatchValue(value="a" * 64),
            )
        ],
    )
)
```

Assert `wait=True`. Add a provider-failure test that checks `VisualIndexError` and preserved `__cause__`.

Run:

```bash
uv run pytest tests/src/winston/index -q
```

Expected: FAIL until `delete_old_revisions()` exists.

- [ ] **Step 5: Implement stale-revision deletion**

Add `delete()` to the internal `_QdrantClient` protocol and implement:

```python
async def delete_old_revisions(
    self,
    *,
    source_path: str,
    current_asset_id: str,
) -> None:
    """Delete older revisions for one source path before indexing its replacement."""
    if self._identity is None:
        raise VisualIndexConfigurationError(
            "ensure_compatible() must succeed before old revisions can be deleted"
        )

    selector = models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_path",
                    match=models.MatchValue(value=source_path),
                )
            ],
            must_not=[
                models.FieldCondition(
                    key="asset_id",
                    match=models.MatchValue(value=current_asset_id),
                )
            ],
        )
    )
    try:
        await self._client.delete(
            self._settings.collection,
            points_selector=selector,
            wait=True,
        )
    except Exception as exc:
        raise VisualIndexError(
            f"Failed to delete old visual revisions for {source_path} "
            f"from collection '{self._settings.collection}'"
        ) from exc
```

Do not delete partial points belonging to the current asset ID; retries overwrite those deterministic point IDs.

- [ ] **Step 6: Run and commit Task 2**

```bash
uv run pytest tests/src/winston/index -q
git add src/winston/index tests/src/winston/index
git commit -m "feat: add dataset-owned Qdrant index sessions"
```

Expected: PASS.

---

### Task 3: Add durable dataset identity and restart manifest

**Files:**
- Create: `src/winston/indexing/__init__.py`
- Create: `src/winston/indexing/manifest.py`
- Create: `tests/src/winston/indexing/test_manifest.py`

**Interfaces:**
- Produces `DatasetIdentity(dataset_instance_id: str)`
- Produces `ManifestError`
- Produces `load_or_create_dataset_identity(root: Path) -> DatasetIdentity`
- Produces `IndexManifest.completed_asset_ids(index_instance_id: str) -> set[str]`
- Produces `IndexManifest.mark_completed(*, index_instance_id: str, asset_id: str, source_path: str) -> None`

- [ ] **Step 1: Write failing dataset identity tests**

Create `tests/src/winston/indexing/test_manifest.py` with:

```python
def test_dataset_identity_is_created_once_and_reused(tmp_path: Path) -> None:
    """One indexing root keeps one stable logical dataset UUID."""
    first = load_or_create_dataset_identity(tmp_path)
    second = load_or_create_dataset_identity(tmp_path)

    assert first == second
    assert str(UUID(first.dataset_instance_id)) == first.dataset_instance_id
    assert (tmp_path / ".winston" / "dataset.json").is_file()


def test_invalid_dataset_document_is_fatal(tmp_path: Path) -> None:
    """Corrupt ownership state is never silently replaced with a new identity."""
    state_dir = tmp_path / ".winston"
    state_dir.mkdir()
    (state_dir / "dataset.json").write_text(
        '{"schema_version":1,"dataset_instance_id":"broken"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="dataset.json"):
        load_or_create_dataset_identity(tmp_path)
```

Run the file; expected FAIL because the module does not exist.

- [ ] **Step 2: Implement atomic `dataset.json` state**

Use strict Pydantic serialization models:

```python
@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Stable logical identity stored with one indexing root."""

    dataset_instance_id: str


class _DatasetDocument(BaseModel):
    """Strict serialized representation of `.winston/dataset.json`."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    dataset_instance_id: UUID
```

`load_or_create_dataset_identity()` must perform these operations in order:

```text
resolve root with strict=True
reject non-directory root
mkdir <root>/.winston
if dataset.json exists:
    read UTF-8
    _DatasetDocument.model_validate_json(...)
    return canonical UUID string
else:
    generate uuid4
    create a uniquely named temp file inside .winston
    write compact JSON + newline
    flush temp file
    fsync temp file
    os.replace(temp, dataset.json)
    return generated UUID
```

On read/validation/write failure, raise `ManifestError` naming `dataset.json`. Remove a leftover temp file in the write-error path. Concurrent index processes are out of scope, so do not add locking.

- [ ] **Step 3: Write failing journal tests**

Cover exact semantics:

```python
def test_manifest_filters_completion_by_index_instance(tmp_path: Path) -> None:
    manifest = IndexManifest(tmp_path)
    old_id = "d44b56ad-b945-4dc7-afc3-5e9af7e19e26"
    current_id = "ddbbd067-036e-4f45-a36c-69f702377c97"

    manifest.mark_completed(index_instance_id=old_id, asset_id="a" * 64, source_path="old.jpg")
    manifest.mark_completed(
        index_instance_id=current_id,
        asset_id="b" * 64,
        source_path="current.jpg",
    )

    assert manifest.completed_asset_ids(current_id) == {"b" * 64}
```

Also add:

```text
- missing index-state.jsonl -> empty set
- malformed/truncated final non-empty line -> ignored
- invalid UTF-8 in final non-empty line -> ignored
- malformed JSON before the final non-empty line -> ManifestError with line number
- structurally invalid record before the final line -> ManifestError with line number
- records from old index_instance_id -> ignored
- mark_completed writes only status="completed"
- mark_completed flushes and calls os.fsync before returning
- invalid asset_id or non-normalized source_path -> ManifestError
```

- [ ] **Step 4: Implement strict append-only JSONL handling**

Use:

```python
class _CompletionRecord(BaseModel):
    """One completed asset tied to one exact Qdrant collection instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    index_instance_id: UUID
    asset_id: str
    source_path: str
    status: Literal["completed"]
```

Add field validation for 64 lowercase hexadecimal `asset_id` and normalized relative POSIX `source_path` using the same rules as `IndexedVisual.source_path`.

`completed_asset_ids()` must read bytes, split into physical lines, ignore blank lines, and decode/validate each non-empty line separately. This is required so a crash that truncates the final UTF-8 code point can still be treated as an ignorable final partial record. Any decode/JSON/schema failure before the final non-empty line raises `ManifestError`.

`mark_completed()` must create `.winston` if necessary, serialize one compact record, append one newline, call `flush()`, then `os.fsync(file.fileno())`, and only then return success.

- [ ] **Step 5: Run and commit Task 3**

```bash
uv run pytest tests/src/winston/indexing/test_manifest.py -q
git add src/winston/indexing tests/src/winston/indexing/test_manifest.py
git commit -m "feat: add durable indexing restart state"
```

Expected: PASS.

---

### Task 4: Implement the sequential indexing orchestrator

**Files:**
- Create: `src/winston/indexing/models.py`
- Create: `src/winston/indexing/pipeline.py`
- Modify: `src/winston/indexing/__init__.py`
- Create: `tests/src/winston/indexing/test_pipeline.py`

**Interfaces:**
- Consumes `FrameSampler`, `MultimodalEmbedder`, `VisualIndex`, `IndexManifest`, ingest/probe/sampling/region helpers
- Produces `PipelineStage`, `AssetFailure`, `IndexRunResult`, `IndexingRunError`
- Produces `IndexingPipeline.run() -> IndexRunResult`
- Produces `run_indexing(root: Path, settings: Settings) -> IndexRunResult`

- [ ] **Step 1: Add failing tests and exact fakes**

Create `tests/src/winston/indexing/test_pipeline.py`. The embedder fake must match the real protocol type, including `RGBImage`:

```python
class FakeEmbedder:
    """Deterministic embedder that records orchestration batch sizes."""

    def __init__(self) -> None:
        self.image_batch_sizes: list[int] = []
        self.closed = False

    @property
    def identity(self) -> EmbeddingIdentity:
        return IDENTITY

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        self.image_batch_sizes.append(len(images))
        return EmbeddingBatch(
            vectors=np.zeros((len(images), 768), dtype=np.float32)
        )

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise AssertionError("text embedding is not used by the visual indexing pipeline")

    async def close(self) -> None:
        self.closed = True
```

Create a `FakeVisualIndex` implementing the Phase 1E `VisualIndex` signatures and recording `ensure`, `delete`, and `upsert` events. Create a fake `FrameSampler` whose async iterator yields configured `SampledFrame` values.

Run:

```bash
uv run pytest tests/src/winston/indexing/test_pipeline.py -q
```

Expected: FAIL because the orchestration package is incomplete.

- [ ] **Step 2: Implement structured run models**

Create `src/winston/indexing/models.py`:

```python
class IndexingRunError(RuntimeError):
    """Raised for fatal initialization or cleanup failures that invalidate a whole run."""


class PipelineStage(StrEnum):
    """Asset stage retained in actionable failure diagnostics."""

    IDENTITY = "identity"
    DELETE = "delete"
    PROBE = "probe"
    DECODE = "decode"
    SAMPLING = "sampling"
    REGIONS = "regions"
    EMBEDDING = "embedding"
    UPSERT = "upsert"
    MANIFEST = "manifest"


@dataclass(frozen=True, slots=True)
class AssetFailure:
    """One failed media asset with its original exception preserved."""

    source_path: str
    stage: PipelineStage
    message: str
    cause: Exception = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class IndexRunResult:
    """Structured outcome of one complete sequential indexing run."""

    indexed: int
    skipped: int
    failures: tuple[AssetFailure, ...]

    @property
    def failed(self) -> int:
        """Return the number of failed assets."""
        return len(self.failures)

    @property
    def exit_code(self) -> int:
        """Return zero only when every discovered asset succeeded or was skipped."""
        return 1 if self.failures else 0
```

- [ ] **Step 3: Implement `IndexingPipeline` initialization and fatal setup**

Use this constructor:

```python
def __init__(
    self,
    *,
    root: Path,
    settings: Settings,
    embedder: MultimodalEmbedder,
    visual_index: VisualIndex,
    frame_sampler: FrameSampler,
) -> None:
    """Bind one sequential indexing run to reusable embedding/storage dependencies."""
    self._root = root.resolve(strict=True)
    if not self._root.is_dir():
        raise IndexingRunError(f"indexing root is not a directory: {self._root}")
    self._settings = settings
    self._embedder = embedder
    self._visual_index = visual_index
    self._frame_sampler = frame_sampler
    self._stage = PipelineStage.IDENTITY
```

`run()` performs fatal setup before the asset loop:

```text
load_or_create_dataset_identity(root)
await visual_index.ensure_compatible(embedder.identity, dataset_instance_id)
IndexManifest(root).completed_asset_ids(session.index_instance_id)
scan_media(root)
```

If any of those operations fail, let the exception escape as a run-level failure. Do not turn dataset corruption, collection incompatibility, Qdrant initialization failure, or scanner-root failure into an asset failure.

- [ ] **Step 4: Implement the common per-sample region pipeline**

Use `itertools.batched()` directly over `generate_regions()`:

```python
async def _index_sample(
    self,
    *,
    identity: AssetIdentity,
    media_type: MediaType,
    sample_kind: SampleKind,
    timestamp_seconds: float | None,
    width: int,
    height: int,
    rgb24: bytes,
) -> None:
    """Generate, embed, and persist one photo or keyframe with bounded buffering."""
    batches = batched(
        generate_regions(width=width, height=height, rgb24=rgb24),
        int(self._settings.indexing.visual_batch_size),
    )

    while True:
        self._stage = PipelineStage.REGIONS
        try:
            regions = next(batches)
        except StopIteration:
            return

        self._stage = PipelineStage.EMBEDDING
        embedded = await self._embedder.embed_images(regions)
        if embedded.count != len(regions):
            raise ValueError(
                f"embedder returned {embedded.count} vectors for {len(regions)} regions"
            )

        visuals = tuple(
            IndexedVisual(
                asset_id=identity.asset_id,
                source_path=identity.source_path,
                media_type=media_type,
                sample_kind=sample_kind,
                timestamp_seconds=timestamp_seconds,
                region_kind=region.region_kind,
                region=RegionGeometry(
                    x=region.x,
                    y=region.y,
                    width=region.width,
                    height=region.height,
                    scale=region.scale,
                ),
                vector=embedded.vectors[position],
                embedding_identity=self._embedder.identity,
            )
            for position, region in enumerate(regions)
        )

        self._stage = PipelineStage.UPSERT
        await self._visual_index.upsert(visuals)
```

At most 9 region objects are retained by this orchestration batch. Do not collect all regions or keyframes.

- [ ] **Step 5: Implement exact asset ordering and accurate stage tracking**

Inside the asset loop:

```text
stage IDENTITY -> identify_asset(root, media.path)
if asset_id in completed -> increment skipped and continue
stage DELETE -> await delete_old_revisions(source_path, current_asset_id)
stage PROBE -> probe_media(media)
```

For an image:

```text
stage DECODE -> load_image(metadata)
await _index_sample(media_type=image, sample_kind=image, timestamp=None)
```

For a video, do not use an `async for` that leaves the stage set to the previous upsert while fetching the next frame. Fetch explicitly:

```python
frames = self._frame_sampler.sample(metadata)
while True:
    self._stage = PipelineStage.SAMPLING
    try:
        frame = await anext(frames)
    except StopAsyncIteration:
        break
    await self._index_sample(
        identity=identity,
        media_type=MediaType.VIDEO,
        sample_kind=SampleKind.KEYFRAME,
        timestamp_seconds=frame.timestamp_seconds,
        width=frame.width,
        height=frame.height,
        rgb24=frame.rgb24,
    )
```

After every sample/upsert succeeds:

```text
stage MANIFEST -> mark_completed(index_instance_id, asset_id, source_path)
add asset_id to in-memory completed set
increment indexed
```

The asset loop catches only expected media/pipeline failures (`OSError`, `ValueError`, probe/sampling/image/embedding/index/manifest domain errors), records `AssetFailure(source_path, current_stage, str(exc), exc)`, and continues. Unexpected programmer errors are allowed to escape as run-level failures rather than being mislabeled as bad media.

- [ ] **Step 6: Add the orchestration behavior tests**

Add these named tests with the stated assertions:

```text
test_photo_reaches_regions_embeddings_and_qdrant
  -> indexed=1, image sample kind, timestamp=None, completion recorded last

test_video_streams_keyframes_in_source_order
  -> keyframe timestamps preserved; sampler is consumed one frame at a time

test_visual_batch_size_never_exceeds_nine
  -> >9 generated regions; max(fake_embedder.image_batch_sizes) <= 9; multiple calls

test_completed_asset_skips_before_probe
  -> probe fake raises if called; skipped=1; no delete/upsert

test_interrupted_asset_is_retried
  -> first run fails before manifest append; second run indexes same asset

test_partial_prior_upserts_are_idempotently_rewritten
  -> fake index receives same deterministic visuals again on retry

test_changed_revision_deletes_before_first_upsert
  -> delete event precedes every upsert for that asset

test_delete_failure_prevents_new_writes
  -> stage=DELETE; zero upserts for that asset

test_one_asset_failure_does_not_stop_next_asset
  -> first probe fails; second succeeds; indexed=1, failed=1

test_manifest_append_failure_is_asset_failure
  -> Qdrant write happened; no completion; next run retries

test_assets_are_processed_sequentially
  -> all start/end events for asset A precede asset B
test_sampling_failure_on_second_frame_reports_sampling_stage
  -> explicit anext stage reset is verified
```

Run:

```bash
uv run pytest tests/src/winston/indexing/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 7: Implement reusable dependency lifecycle**

`run_indexing()` constructs one embedder, one Qdrant client wrapper, and one keyframe sampler for the whole command. Close Qdrant and the embedder on success and failure.

Use a helper with this contract:

```python
async def _close_dependencies(
    visual_index: VisualIndex,
    embedder: MultimodalEmbedder,
) -> tuple[str, ...]:
    """Close both reusable dependencies and return cleanup diagnostics without masking peers."""
    errors: list[str] = []
    try:
        await visual_index.close()
    except Exception as exc:
        errors.append(f"visual index close failed: {exc}")
    try:
        await embedder.close()
    except Exception as exc:
        errors.append(f"embedder close failed: {exc}")
    return tuple(errors)
```

The `run_indexing()` control flow is:

```text
create_embedder(settings)
construct QdrantVisualIndex(settings.qdrant)
construct IndexingPipeline(..., KeyframeSampler())
try await pipeline.run()
if a primary BaseException occurs:
    await _close_dependencies
    add each cleanup diagnostic with primary.add_note(...)
    re-raise the primary exception
otherwise:
    await _close_dependencies
    if cleanup errors exist -> raise IndexingRunError
    return IndexRunResult
```

If constructing `QdrantVisualIndex` itself raises after the embedder exists, close the embedder before re-raising and attach a cleanup note if that close also fails.

Add tests for both closes on success, both closes after a fatal pipeline error, a cleanup-only failure producing `IndexingRunError`, and a primary failure remaining primary when cleanup also fails.

- [ ] **Step 8: Run and commit Task 4**

```bash
uv run pytest tests/src/winston/indexing -q
git add src/winston/indexing tests/src/winston/indexing
git commit -m "feat: orchestrate restartable media indexing"
```

Expected: PASS.

---

### Task 5: Wire `winston index` into the CLI

**Files:**
- Modify: `src/winston/cli.py`
- Modify: `tests/src/winston/test_cli.py`

**Interfaces:**
- Consumes `run_indexing(root, settings) -> IndexRunResult`
- Produces `winston index [path]`
- Exit `0`: all discovered assets indexed or skipped
- Exit `1`: asset failure or fatal run-level failure

- [ ] **Step 1: Write failing CLI tests**

Add tests for:

```text
explicit path -> resolved path is passed to run_indexing
omitted path -> Settings.data_dir is used
successful run -> exit 0
all-skipped/no-op run -> exit 0
asset failure -> exit 1 and relative path/stage/message are printed
fatal IndexingRunError -> exit 1 and diagnostic goes to stderr
```

Use this failure object in the output test:

```python
failure = AssetFailure(
    source_path="cameras/cam12.mkv",
    stage=PipelineStage.SAMPLING,
    message="ffmpeg failed to decode keyframes",
    cause=RuntimeError("decoder error"),
)
```

Assert the summary contains:

```text
Indexed: 3
Skipped: 2
Failed: 1

FAILED cameras/cam12.mkv [sampling]
  ffmpeg failed to decode keyframes
```

- [ ] **Step 2: Use one settings snapshot for parsing and execution**

Change `build_parser()` to:

```python
def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """Build Winston's CLI parser from one runtime settings snapshot."""
    config = settings or get_config()
    parser = argparse.ArgumentParser(prog="winston")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Discover and inspect local media files",
    )
    scan_parser.add_argument("path", nargs="?", type=Path, default=config.data_dir)

    index_parser = subparsers.add_parser(
        "index",
        help="Index local videos and photos into the configured visual index",
    )
    index_parser.add_argument("path", nargs="?", type=Path, default=config.data_dir)
    return parser
```

Change `main()` to create one `Settings` object, pass it into `build_parser()`, and pass the same object to `index_command()`.

- [ ] **Step 3: Implement summary and fatal-error mapping**

Add:

```python
def _print_index_result(result: IndexRunResult) -> None:
    """Print a concise Phase 1E summary and actionable asset failures."""
    print(f"Indexed: {result.indexed}")
    print(f"Skipped: {result.skipped}")
    print(f"Failed: {result.failed}")
    for failure in result.failures:
        print()
        print(f"FAILED {failure.source_path} [{failure.stage.value}]")
        print(f"  {failure.message}")


def index_command(root: Path, settings: Settings) -> int:
    """Run asynchronous indexing and map the structured outcome to a shell exit code."""
    try:
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_dir():
            raise ValueError(f"indexing root is not a directory: {resolved_root}")
        result = asyncio.run(run_indexing(resolved_root, settings))
    except Exception as exc:
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1

    _print_index_result(result)
    return result.exit_code
```

Do not catch `KeyboardInterrupt`; Ctrl+C keeps normal shell interruption semantics and restart safety comes from the manifest/idempotent upserts.

- [ ] **Step 4: Run and commit Task 5**

```bash
uv run pytest tests/src/winston/test_cli.py tests/src/winston/test_config.py tests/src/winston/indexing -q
git add src/winston/cli.py tests/src/winston/test_cli.py
git commit -m "feat: expose restartable index CLI"
```

Expected: PASS.

---

### Task 6: Real-Qdrant integration, GitHub Action coverage, and draft PR

**Files:**
- Modify: `tests/integration/test_qdrant_visual_index.py`
- Create: `tests/integration/test_restartable_index_pipeline.py`
- Modify: `.github/workflows/qdrant-index.yml`

**Interfaces verified:**
- schema-v2 ownership metadata
- stale-revision deletion filter against real Qdrant
- first-run indexing and second-run skip/idempotency
- recreated collection obtains a new `index_instance_id`, invalidating old completion records

- [ ] **Step 1: Update the existing real-Qdrant test to schema version 2**

Use a stable dataset UUID and assert:

```python
session = await index.ensure_compatible(IDENTITY, DATASET_ID)
info = await client.get_collection(collection)

assert info.config.metadata is not None
assert info.config.metadata["winston_schema_version"] == 2
assert info.config.metadata["dataset_instance_id"] == DATASET_ID
assert info.config.metadata["index_instance_id"] == session.index_instance_id
assert str(UUID(session.index_instance_id)) == session.index_instance_id
```

Keep the existing vector size, cosine metric, payload, and duplicate-upsert count assertions.

- [ ] **Step 2: Add real stale-revision deletion coverage**

Upsert two visuals with identical `source_path` and different asset IDs. Call:

```python
await index.delete_old_revisions(
    source_path="cameras/integration/cam01.mkv",
    current_asset_id="f" * 64,
)
```

Query the path and assert only the `f * 64` revision remains.

- [ ] **Step 3: Add a real-Qdrant restart smoke test**

Create `tests/integration/test_restartable_index_pipeline.py` using:

```text
tmp_path dataset
one Pillow-generated JPG
real scanner
real ffprobe-based probe
real Pillow sampling
real region generation
real asset identity + manifest
fake deterministic 768D embedder
real QdrantVisualIndex
collection = f"winston_restartable_index_{uuid4().hex}"
```

First run assertions:

```python
assert first.indexed == 1
assert first.skipped == 0
assert first.failed == 0
first_count = (await client.count(collection, exact=True)).count
assert first_count > 0
```

Second run against the same dataset/collection:

```python
assert second.indexed == 0
assert second.skipped == 1
assert second.failed == 0
assert (await client.count(collection, exact=True)).count == first_count
```

Then delete the collection, build a fresh `QdrantVisualIndex` against the same local `.winston` state, run again, and assert the asset is indexed rather than skipped and the new session has a different `index_instance_id`.

Delete the unique test collection in `finally`.

- [ ] **Step 4: Expand the existing Qdrant GitHub Action for Phase 1E**

Modify `.github/workflows/qdrant-index.yml` path filters to add:

```yaml
      - 'src/winston/indexing/**'
      - 'src/winston/sampling/**'
      - 'src/winston/cli.py'
      - 'tests/src/winston/indexing/**'
      - 'tests/src/winston/sampling/test_images.py'
      - 'tests/src/winston/test_cli.py'
      - 'tests/integration/test_restartable_index_pipeline.py'
```

Add Pillow to the focused pip install:

```bash
'pillow>=12.0.0'
```

Add an FFmpeg install step before tests because the restart integration test intentionally exercises real `ffprobe`:

```yaml
      - name: Install FFmpeg
        run: |
          sudo apt-get update
          sudo apt-get install -y ffmpeg
```

Replace the Phase 1D focused unit command with:

```yaml
      - name: Run Phase 1E focused unit tests
        run: >-
          python -m pytest -q
          tests/src/winston/test_config.py
          tests/src/winston/test_cli.py
          tests/src/winston/ingest/test_identity.py
          tests/src/winston/sampling/test_images.py
          tests/src/winston/index
          tests/src/winston/indexing
```

Run both real-Qdrant files:

```yaml
      - name: Run real Qdrant integration tests
        run: >-
          python -m pytest -q
          tests/integration/test_qdrant_visual_index.py
          tests/integration/test_restartable_index_pipeline.py
```

Keep the existing Qdrant `v1.18.2` service and 15-minute timeout.

- [ ] **Step 5: Run local verification**

```bash
uv lock --check
uv run pytest tests/src -q
docker compose up -d qdrant
uv run pytest tests/integration/test_qdrant_visual_index.py tests/integration/test_restartable_index_pipeline.py -q
uv run pytest -q
uv run python -m compileall -q src tests
uv lock --check
```

Expected: every command PASS. Do not stop a Qdrant service that the execution environment did not start/own.

- [ ] **Step 6: Self-review the implementation against the spec before PR creation**

Verify each item explicitly:

```text
[ ] visual_batch_size default is 9
[ ] photos decode in sampling, not ingest
[ ] one asset at a time
[ ] keyframes remain streamed
[ ] dataset.json is atomic and stable
[ ] index-state.jsonl records completed assets only
[ ] completion append flushes + fsyncs
[ ] only final malformed/truncated journal line is ignored
[ ] Qdrant Winston schema is 2
[ ] wrong dataset_instance_id is rejected
[ ] index_instance_id scopes manifest completion
[ ] schema-1 collections require explicit reindex
[ ] old revisions are deleted before replacement indexing
[ ] delete failure causes zero new writes for that asset
[ ] deterministic point ID implementation is unchanged
[ ] failed asset does not stop later assets
[ ] cleanup failure cannot hide an earlier primary failure
[ ] CLI prints Indexed / Skipped / Failed plus asset path/stage/message
[ ] file mutation during indexing is intentionally unsupported
```

- [ ] **Step 7: Commit integration/workflow coverage**

```bash
git add tests/integration .github/workflows/qdrant-index.yml
git commit -m "test: cover restartable indexing with Qdrant"
```

- [ ] **Step 8: Push and create one draft PR for issue #9**

```bash
git push -u origin feat/restartable-index-cli
```

Draft PR title:

```text
feat: add restartable index CLI
```

PR body:

```markdown
## Summary
- compose ingest, sampling, deterministic regions, embeddings, and Qdrant behind `winston index`
- add asset-level durable restart state and dataset/collection ownership identities
- support videos and JPG/JPEG photos with bounded visual batching
- delete stale source revisions before indexing replacements

## Validation
- `uv lock --check`
- `uv run pytest -q`
- real Qdrant 1.18 integration tests

Closes #9
```

Keep the PR draft until implementation review and all relevant GitHub Actions are green.
