# Phase 1E — Restartable Index CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `winston index <path>` as a restartable, idempotent, bounded-memory V0 indexing pipeline for videos and JPG/JPEG photos.

**Architecture:** Keep `cli.py` thin and add a dedicated `winston.indexing` orchestration layer. Persist dataset identity and asset-level completion state under `<root>/.winston`, evolve the Qdrant collection contract to schema version 2 with dataset/collection-instance ownership, stream one asset at a time, batch at most 9 visual regions per embedding call, and rely on deterministic Qdrant point IDs for retry idempotency.

**Tech Stack:** Python 3.13+, argparse, asyncio, Pillow, Pydantic/Pydantic Settings, NumPy, FFmpeg/ffprobe, Jina CLIP embedding contracts, qdrant-client 1.18.x, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-08-22-restartable-index-cli-design.md`

## Global Constraints

- The command is `winston index [path]`; when `path` is omitted it defaults to `Settings.data_dir`.
- Source media are immutable for the duration of one indexing run. Do not add end-of-run restat/mutation detection.
- Process exactly one media asset at a time. Do not add inter-asset concurrency.
- Preserve streaming video behavior: never materialize all keyframes for a video.
- Add `IndexingSettings.visual_batch_size: PositiveInt = 9`; this bounds orchestration memory and is independent of embedding-provider batch sizes.
- One Qdrant visual collection belongs to exactly one dataset identity.
- Qdrant Winston collection metadata schema becomes version `2` and requires `dataset_instance_id` plus `index_instance_id`.
- Do not automatically drop, clear, rename, migrate, or recreate an incompatible Qdrant collection.
- When a source path has a new `asset_id`, delete older revisions for that `source_path` before indexing the current revision.
- If stale-revision deletion fails, do not write the new revision during that run.
- A failed asset is not marked complete; continue with later assets and return exit code `1` after the run.
- Durable completion is asset-level only. Do not add per-keyframe/per-region checkpoints or a generic job framework.
- Use `Protocol`, not `ABC`, for structural contracts. Do not introduce `typing.Any`.
- Every new function and method must have a docstring; add comments around non-obvious crash-safety, ordering, filtering, and cleanup logic.
- Keep async for actual asynchronous I/O/concurrency boundaries; do not convert pure validation/mapping helpers to async.
- Preserve deterministic `visual_point_id()` behavior from Phase 1D.
- Keep `qdrant-client>=1.18,<1.19` and the pinned Qdrant 1.18 service line.
- The implementation PR remains one focused draft PR for issue #9 and its body must contain `Closes #9`.

---

## File Structure

Create or modify these files only as needed for Phase 1E:

```text
src/winston/
├── cli.py                              # add index command, summary, exit status
├── config.py                           # add IndexingSettings.visual_batch_size=9
├── sampling/
│   ├── __init__.py                     # export photo sampling public names
│   ├── images.py                       # Pillow photo decode + metadata validation
│   └── models.py                       # add SampledImage
├── index/
│   ├── __init__.py                     # export VisualIndexSession
│   ├── base.py                         # evolve VisualIndex protocol
│   ├── models.py                       # add VisualIndexSession
│   └── qdrant.py                       # schema v2, ownership, delete old revisions
└── indexing/
    ├── __init__.py                     # public Phase 1E orchestration exports
    ├── manifest.py                     # dataset identity + durable JSONL completion state
    ├── models.py                       # PipelineStage, AssetFailure, IndexRunResult/errors
    └── pipeline.py                     # full sequential indexing orchestration + lifecycle

tests/src/winston/
├── test_config.py
├── test_cli.py
├── sampling/test_images.py
├── index/test_qdrant.py
├── index/test_qdrant_upsert.py
└── indexing/
    ├── test_manifest.py
    └── test_pipeline.py

tests/integration/
├── test_qdrant_visual_index.py          # update schema v2 expectations + revision deletion
└── test_restartable_index_pipeline.py   # real-Qdrant restart/idempotency smoke test
```

Do not add a new dependency: Pillow is already present in `pyproject.toml`.

---

### Task 1: Add indexing configuration and deterministic photo sampling

**Files:**
- Modify: `src/winston/config.py`
- Modify: `src/winston/sampling/models.py`
- Create: `src/winston/sampling/images.py`
- Modify: `src/winston/sampling/__init__.py`
- Modify: `tests/src/winston/test_config.py`
- Create: `tests/src/winston/sampling/test_images.py`

**Interfaces:**
- Produces: `IndexingSettings.visual_batch_size: PositiveInt = 9`
- Produces: `SampledImage(source_path: Path, width: int, height: int, rgb24: bytes)`
- Produces: `ImageSamplingError`
- Produces: `load_image(metadata: ImageMetadata) -> SampledImage`
- Consumed later by: `IndexingPipeline`

- [ ] **Step 1: Write failing config tests for the Phase 1E batch-size contract**

Add explicit assertions to `tests/src/winston/test_config.py`:

```python
def test_indexing_visual_batch_size_defaults_to_nine() -> None:
    """Phase 1E must bound orchestration batches to nine visuals by default."""
    settings = Settings()

    assert settings.indexing.visual_batch_size == 9


def test_indexing_visual_batch_size_can_be_overridden_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested settings must expose the documented INDEXING__ override."""
    monkeypatch.setenv("INDEXING__VISUAL_BATCH_SIZE", "3")

    settings = Settings()

    assert settings.indexing.visual_batch_size == 3
```

Run:

```bash
uv run pytest tests/src/winston/test_config.py -q
```

Expected: FAIL because `Settings.indexing` does not exist yet.

- [ ] **Step 2: Implement `IndexingSettings` and wire it into `Settings`**

Add to `src/winston/config.py`:

```python
class IndexingSettings(BaseSettings):
    """Settings that bound Winston's high-level indexing orchestration."""

    visual_batch_size: PositiveInt = 9
```

Add the field to `Settings`:

```python
indexing: IndexingSettings = Field(default_factory=IndexingSettings)
```

Run:

```bash
uv run pytest tests/src/winston/test_config.py -q
```

Expected: PASS.

- [ ] **Step 3: Write failing photo-decoding tests**

Create `tests/src/winston/sampling/test_images.py` with concrete coverage for RGB conversion, byte shape, dimension validation, and path-aware errors. Use Pillow to generate fixtures rather than checking in binary test assets:

```python
from pathlib import Path

import pytest
from PIL import Image

from winston.ingest.models import ImageMetadata
from winston.sampling.images import ImageSamplingError, load_image


def test_load_image_decodes_rgb24(tmp_path: Path) -> None:
    """A normal JPEG must become contiguous RGB24 with probed dimensions preserved."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 3), (10, 20, 30)).save(path, format="JPEG")

    sampled = load_image(ImageMetadata(path=path, width=4, height=3))

    assert sampled.source_path == path
    assert sampled.width == 4
    assert sampled.height == 3
    assert len(sampled.rgb24) == 4 * 3 * 3


def test_load_image_converts_non_rgb_input(tmp_path: Path) -> None:
    """Non-RGB source modes must be converted explicitly before bytes are exposed."""
    path = tmp_path / "gray.jpg"
    Image.new("L", (2, 2), 127).save(path, format="JPEG")

    sampled = load_image(ImageMetadata(path=path, width=2, height=2))

    assert len(sampled.rgb24) == 12


def test_load_image_rejects_probe_dimension_mismatch(tmp_path: Path) -> None:
    """Decoded geometry must match the metadata already accepted by ingest/probe."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 3)).save(path, format="JPEG")

    with pytest.raises(ImageSamplingError, match=r"photo\.jpg.*4x3.*5x3"):
        load_image(ImageMetadata(path=path, width=5, height=3))


def test_load_image_wraps_corrupt_image_with_path(tmp_path: Path) -> None:
    """Decode failures must identify the media path and retain the original cause."""
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

Expected: FAIL because `winston.sampling.images` and `SampledImage` do not exist yet.

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

Create `src/winston/sampling/images.py` around Pillow:

```python
"""Decode still images into Winston's RGB24 sampling contract."""

from PIL import Image, UnidentifiedImageError

from winston.ingest.models import ImageMetadata
from winston.sampling.models import SampledImage


class ImageSamplingError(RuntimeError):
    """Raised when a still image cannot be decoded into the expected RGB24 sample."""


def load_image(metadata: ImageMetadata) -> SampledImage:
    """Decode one image, convert it to RGB, and validate dimensions against probe metadata."""
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
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageSamplingError(f"Failed to decode image {metadata.path}: {exc}") from exc

    expected = width * height * 3
    if len(rgb24) != expected:
        raise ImageSamplingError(
            f"Decoded RGB24 buffer for {metadata.path} has {len(rgb24)} bytes; expected {expected}"
        )

    return SampledImage(
        source_path=metadata.path,
        width=width,
        height=height,
        rgb24=rgb24,
    )
```

Export `ImageSamplingError`, `SampledImage`, and `load_image` from `src/winston/sampling/__init__.py` consistently with the existing public sampling exports.

- [ ] **Step 5: Run the focused sampling/config suite**

Run:

```bash
uv run pytest tests/src/winston/test_config.py tests/src/winston/sampling -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/winston/config.py src/winston/sampling tests/src/winston/test_config.py tests/src/winston/sampling/test_images.py
git commit -m "feat: add bounded photo sampling for indexing"
```

---

### Task 2: Evolve the VisualIndex/Qdrant contract to collection schema version 2

**Files:**
- Modify: `src/winston/index/base.py`
- Modify: `src/winston/index/models.py`
- Modify: `src/winston/index/qdrant.py`
- Modify: `src/winston/index/__init__.py`
- Modify: `tests/src/winston/index/test_qdrant.py`
- Modify: `tests/src/winston/index/test_qdrant_upsert.py`

**Interfaces:**
- Produces: `VisualIndexSession(index_instance_id: str)`
- Produces: `VisualIndex.ensure_compatible(identity, dataset_instance_id) -> VisualIndexSession`
- Produces: `VisualIndex.delete_old_revisions(*, source_path, current_asset_id) -> None`
- Preserves: `VisualIndex.upsert(...)` and deterministic point IDs
- Consumed later by: `IndexingPipeline`

- [ ] **Step 1: Update tests to define the schema-v2 metadata contract**

Change the Qdrant unit-test metadata helper to require dataset and collection-instance identities:

```python
DATASET_ID = "8f7ad0c0-7ab7-4ec0-9025-22f40dd70c4a"
INDEX_INSTANCE_ID = "0f27f24e-7436-4c46-9260-4a37f419b3a6"


def expected_metadata(
    identity: EmbeddingIdentity = IDENTITY,
    *,
    dataset_instance_id: str = DATASET_ID,
    index_instance_id: str = INDEX_INSTANCE_ID,
) -> dict[str, object]:
    """Return the exact Phase 1E Winston collection metadata contract."""
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

Update all successful calls from:

```python
await index.ensure_compatible(IDENTITY)
```

to:

```python
session = await index.ensure_compatible(IDENTITY, DATASET_ID)
assert session.index_instance_id == INDEX_INSTANCE_ID
```

For a missing collection, assert the returned `index_instance_id` parses as a UUID and exactly matches the value persisted in `create_collection(..., metadata=...)`.

Add explicit failures:

```python
@pytest.mark.asyncio
async def test_ensure_compatible_rejects_collection_owned_by_another_dataset() -> None:
    metadata = expected_metadata(dataset_instance_id="6fd3c642-165f-4743-a871-a16df5df7cc7")
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="another dataset|dataset_instance_id"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_schema_one_collection() -> None:
    metadata = expected_metadata()
    metadata["winston_schema_version"] = 1
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="reindex"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)


@pytest.mark.asyncio
async def test_ensure_compatible_rejects_invalid_index_instance_id() -> None:
    metadata = expected_metadata(index_instance_id="not-a-uuid")
    client = FakeQdrantClient(exists=True, info=make_info(metadata=metadata))
    index = QdrantVisualIndex(QdrantSettings(), client=client)

    with pytest.raises(IncompatibleVisualIndexError, match="index_instance_id"):
        await index.ensure_compatible(IDENTITY, DATASET_ID)
```

Run:

```bash
uv run pytest tests/src/winston/index/test_qdrant.py tests/src/winston/index/test_qdrant_upsert.py -q
```

Expected: FAIL because the public protocol and Qdrant implementation still use the Phase 1D signature/schema.

- [ ] **Step 2: Add `VisualIndexSession` and evolve the storage-agnostic protocol**

Add to `src/winston/index/models.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisualIndexSession:
    """Identity of the exact compatible visual-index collection used by one run."""

    index_instance_id: str
```

Update `src/winston/index/base.py`:

```python
class VisualIndex(Protocol):
    """Storage-agnostic async contract for Winston visual embeddings."""

    async def ensure_compatible(
        self,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> VisualIndexSession:
        """Create or validate storage and return the exact collection-instance identity."""
        ...

    async def delete_old_revisions(
        self,
        *,
        source_path: str,
        current_asset_id: str,
    ) -> None:
        """Delete points for older asset revisions at one normalized source path."""
        ...

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        """Idempotently persist already-embedded visual candidates."""
        ...

    async def close(self) -> None:
        """Release storage resources held by the index session."""
        ...
```

Export `VisualIndexSession` from `src/winston/index/__init__.py`.

- [ ] **Step 3: Implement schema-v2 creation and validation in `QdrantVisualIndex`**

In `src/winston/index/qdrant.py`:

```python
from uuid import UUID, uuid4

WINSTON_SCHEMA_VERSION = 2
```

Change `ensure_compatible()` to generate a UUID only for a newly created collection and to return the existing UUID for an existing compatible collection:

```python
async def ensure_compatible(
    self,
    identity: EmbeddingIdentity,
    dataset_instance_id: str,
) -> VisualIndexSession:
    """Create or strictly validate one dataset-owned visual collection."""
    ...
```

The concrete logic must follow this order:

```text
collection missing
  -> index_instance_id = str(uuid4())
  -> create collection with schema=2 + dataset_instance_id + index_instance_id
  -> set self._identity
  -> return VisualIndexSession(index_instance_id)

collection exists
  -> validate named vector / dimension / cosine
  -> validate schema=2 / model / preprocessing / vector_name / distance
  -> validate metadata dataset_instance_id equals requested dataset
  -> parse metadata index_instance_id with UUID(...)
  -> set self._identity
  -> return VisualIndexSession(canonical UUID string)
```

Use a helper with an exact signature so tests can target validation independently if needed:

```python
def _collection_metadata(
    self,
    identity: EmbeddingIdentity,
    *,
    dataset_instance_id: str,
    index_instance_id: str,
) -> CollectionMetadata:
    """Build required Winston schema-v2 ownership and compatibility metadata."""
```

A schema-1 collection must remain incompatible. Do not mutate it in place.

- [ ] **Step 4: Write failing tests for stale-revision deletion**

Extend the fake Qdrant client with a captured delete call:

```python
@dataclass(frozen=True, slots=True)
class DeleteCall:
    collection_name: str
    points_selector: models.FilterSelector
    wait: bool
```

Add a fake method matching the SDK boundary:

```python
async def delete(
    self,
    collection_name: str,
    *,
    points_selector: models.FilterSelector,
    wait: bool,
) -> models.UpdateResult:
    """Capture one synchronous-from-Winston deletion request."""
    self.delete_calls.append(DeleteCall(collection_name, points_selector, wait))
    if self.delete_failure is not None:
        raise self.delete_failure
    return models.UpdateResult(operation_id=1, status=models.UpdateStatus.COMPLETED)
```

Add tests asserting:

```python
@pytest.mark.asyncio
async def test_delete_old_revisions_filters_same_path_and_excludes_current_asset() -> None:
    client = FakeQdrantClient(exists=True, info=make_info())
    index = QdrantVisualIndex(QdrantSettings(), client=client)
    await index.ensure_compatible(IDENTITY, DATASET_ID)

    await index.delete_old_revisions(
        source_path="cameras/cam01.mkv",
        current_asset_id="a" * 64,
    )

    assert len(client.delete_calls) == 1
    call = client.delete_calls[0]
    assert call.collection_name == "winston_visual"
    assert call.wait is True
    assert call.points_selector.filter.must == [
        models.FieldCondition(
            key="source_path",
            match=models.MatchValue(value="cameras/cam01.mkv"),
        )
    ]
    assert call.points_selector.filter.must_not == [
        models.FieldCondition(
            key="asset_id",
            match=models.MatchValue(value="a" * 64),
        )
    ]
```

Also add a failure test asserting `VisualIndexError`, preserved `__cause__`, and no swallowing of the provider error.

Run:

```bash
uv run pytest tests/src/winston/index -q
```

Expected: FAIL until the method exists.

- [ ] **Step 5: Implement `delete_old_revisions()`**

Add `delete(...)` to the internal `_QdrantClient` protocol and implement:

```python
async def delete_old_revisions(
    self,
    *,
    source_path: str,
    current_asset_id: str,
) -> None:
    """Delete older revisions for one source path before a replacement is indexed."""
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

Do not delete points belonging to `current_asset_id`; partial writes from an interrupted retry are intentionally overwritten by deterministic upserts.

- [ ] **Step 6: Run the full index unit suite and commit Task 2**

```bash
uv run pytest tests/src/winston/index -q
git add src/winston/index tests/src/winston/index
git commit -m "feat: add dataset-owned Qdrant index sessions"
```

Expected: all index tests PASS.

---

### Task 3: Implement durable dataset identity and append-only restart manifest

**Files:**
- Create: `src/winston/indexing/__init__.py`
- Create: `src/winston/indexing/manifest.py`
- Create: `tests/src/winston/indexing/test_manifest.py`

**Interfaces:**
- Produces: `DatasetIdentity(dataset_instance_id: str)`
- Produces: `ManifestError`
- Produces: `load_or_create_dataset_identity(root: Path) -> DatasetIdentity`
- Produces: `IndexManifest.completed_asset_ids(index_instance_id: str) -> set[str]`
- Produces: `IndexManifest.mark_completed(*, index_instance_id, asset_id, source_path) -> None`
- Consumed later by: `IndexingPipeline`

- [ ] **Step 1: Write failing tests for stable dataset identity**

Create `tests/src/winston/indexing/test_manifest.py` and start with:

```python
from pathlib import Path
from uuid import UUID

import pytest

from winston.indexing.manifest import (
    IndexManifest,
    ManifestError,
    load_or_create_dataset_identity,
)


def test_dataset_identity_is_created_once_and_reused(tmp_path: Path) -> None:
    """A dataset keeps one UUID independently of its absolute path spelling."""
    first = load_or_create_dataset_identity(tmp_path)
    second = load_or_create_dataset_identity(tmp_path)

    assert second == first
    assert str(UUID(first.dataset_instance_id)) == first.dataset_instance_id
    assert (tmp_path / ".winston" / "dataset.json").is_file()


def test_dataset_identity_rejects_invalid_existing_document(tmp_path: Path) -> None:
    """Corrupt ownership state must be fatal rather than silently replaced."""
    state_dir = tmp_path / ".winston"
    state_dir.mkdir()
    (state_dir / "dataset.json").write_text(
        '{"schema_version":1,"dataset_instance_id":"broken"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="dataset.json"):
        load_or_create_dataset_identity(tmp_path)
```

Run:

```bash
uv run pytest tests/src/winston/indexing/test_manifest.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 2: Implement atomic `dataset.json` creation and strict reload validation**

In `src/winston/indexing/manifest.py`, use strict Pydantic document models to avoid untyped JSON plumbing:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, ValidationError


DATASET_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    """Raised when Winston restart state cannot be trusted or persisted."""


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Stable logical identity stored with one indexing root."""

    dataset_instance_id: str


class _DatasetDocument(BaseModel):
    """Strict serialized representation of `.winston/dataset.json`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset_instance_id: UUID
```

`load_or_create_dataset_identity()` must:

1. resolve the root strictly and require a directory;
2. create `<root>/.winston` if needed;
3. if `dataset.json` exists, validate it and return its canonical UUID string;
4. otherwise generate `uuid4()`, write a complete JSON document to a temporary file in `.winston`, `flush()`, `os.fsync()`, then `os.replace()` it onto `dataset.json`;
5. clean up the temporary file if creation fails;
6. wrap read/write/validation problems in `ManifestError` with the state-file path.

No file locking is required because concurrent indexing processes are explicitly out of scope.

- [ ] **Step 3: Write failing completion-journal tests**

Add exact restart semantics:

```python
def test_manifest_round_trip_filters_by_index_instance(tmp_path: Path) -> None:
    manifest = IndexManifest(tmp_path)
    old_instance = "d44b56ad-b945-4dc7-afc3-5e9af7e19e26"
    current_instance = "ddbbd067-036e-4f45-a36c-69f702377c97"

    manifest.mark_completed(
        index_instance_id=old_instance,
        asset_id="a" * 64,
        source_path="old.jpg",
    )
    manifest.mark_completed(
        index_instance_id=current_instance,
        asset_id="b" * 64,
        source_path="current.jpg",
    )

    assert manifest.completed_asset_ids(current_instance) == {"b" * 64}


def test_manifest_ignores_only_a_corrupt_final_non_empty_line(tmp_path: Path) -> None:
    path = tmp_path / ".winston" / "index-state.jsonl"
    path.parent.mkdir()
    path.write_text(
        '{"schema_version":1,"index_instance_id":"ddbbd067-036e-4f45-a36c-69f702377c97",'
        '"asset_id":"' + "b" * 64 + '","source_path":"current.jpg","status":"completed"}\n'
        '{"schema_version":1,"index_instance_id":',
        encoding="utf-8",
    )

    manifest = IndexManifest(tmp_path)

    assert manifest.completed_asset_ids(
        "ddbbd067-036e-4f45-a36c-69f702377c97"
    ) == {"b" * 64}


def test_manifest_rejects_corruption_before_final_record(tmp_path: Path) -> None:
    path = tmp_path / ".winston" / "index-state.jsonl"
    path.parent.mkdir()
    path.write_text(
        '{broken}\n'
        '{"schema_version":1,"index_instance_id":"ddbbd067-036e-4f45-a36c-69f702377c97",'
        '"asset_id":"' + "b" * 64 + '","source_path":"current.jpg","status":"completed"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="line 1"):
        IndexManifest(tmp_path).completed_asset_ids(
            "ddbbd067-036e-4f45-a36c-69f702377c97"
        )
```

Add one fsync test by monkeypatching `os.fsync` and asserting it is called once during `mark_completed()` before the method returns.

Run:

```bash
uv run pytest tests/src/winston/indexing/test_manifest.py -q
```

Expected: FAIL because journal methods do not exist yet.

- [ ] **Step 4: Implement strict JSONL completion records and durable append**

Add an internal Pydantic record:

```python
class _CompletionRecord(BaseModel):
    """One durably completed asset for one exact Qdrant collection instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    index_instance_id: UUID
    asset_id: str
    source_path: str
    status: Literal["completed"]
```

Validate `asset_id` as exactly 64 lowercase hex characters and `source_path` as a non-empty normalized relative POSIX path before serializing/accepting a record.

Implement:

```python
class IndexManifest:
    """Append-only durable completion journal stored under one indexing root."""

    def __init__(self, root: Path) -> None:
        """Bind the journal to `<root>/.winston/index-state.jsonl`."""
        ...

    def completed_asset_ids(self, index_instance_id: str) -> set[str]:
        """Return completed asset IDs for exactly one Qdrant collection instance."""
        ...

    def mark_completed(
        self,
        *,
        index_instance_id: str,
        asset_id: str,
        source_path: str,
    ) -> None:
        """Append one completed record and fsync it before reporting success."""
        ...
```

For loading, parse non-empty lines in order. A validation/JSON failure is ignored only if that line is the final non-empty line; earlier failures raise `ManifestError` with a 1-based line number.

For append, write exactly:

```text
<compact JSON object>\n
```

then call `flush()` and `os.fsync(file.fileno())` before returning.

- [ ] **Step 5: Run manifest tests and commit Task 3**

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
- Consumes: `scan_media`, `identify_asset`, `probe_media`, `load_image`, `FrameSampler`, `generate_regions`, `MultimodalEmbedder`, `VisualIndex`, `IndexManifest`
- Produces: `PipelineStage`
- Produces: `AssetFailure`
- Produces: `IndexRunResult`
- Produces: `IndexingRunError`
- Produces: `IndexingPipeline.run() -> IndexRunResult`
- Produces: `run_indexing(root: Path, settings: Settings) -> IndexRunResult`

- [ ] **Step 1: Define failing tests around structured run results and a minimal fake stack**

Create `tests/src/winston/indexing/test_pipeline.py`. Define small fakes with exact existing contracts rather than mocking provider internals:

```python
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from winston.embeddings.models import EmbeddingBatch, EmbeddingIdentity
from winston.index.models import IndexedVisual, VisualIndexSession
from winston.ingest.models import ImageMetadata, MediaFile, MediaType, VideoMetadata
from winston.sampling.models import SampledFrame

IDENTITY = EmbeddingIdentity("jinaai/jina-clip-v1", 768, 1)
INDEX_INSTANCE_ID = "ddbbd067-036e-4f45-a36c-69f702377c97"


class FakeEmbedder:
    """Deterministic embedder that records orchestration batch sizes."""

    def __init__(self) -> None:
        self.image_batch_sizes: list[int] = []
        self.closed = False

    @property
    def identity(self) -> EmbeddingIdentity:
        return IDENTITY

    async def embed_images(self, images: Sequence[object]) -> EmbeddingBatch:
        self.image_batch_sizes.append(len(images))
        vectors = np.zeros((len(images), 768), dtype=np.float32)
        return EmbeddingBatch(vectors=vectors)

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise AssertionError("text embedding is not part of Phase 1E indexing")

    async def close(self) -> None:
        self.closed = True


class FakeVisualIndex:
    """Storage fake that records delete-before-upsert ordering."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.upserted: list[IndexedVisual] = []
        self.closed = False

    async def ensure_compatible(
        self,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> VisualIndexSession:
        self.events.append("ensure")
        assert identity == IDENTITY
        assert dataset_instance_id
        return VisualIndexSession(index_instance_id=INDEX_INSTANCE_ID)

    async def delete_old_revisions(
        self,
        *,
        source_path: str,
        current_asset_id: str,
    ) -> None:
        self.events.append(f"delete:{source_path}:{current_asset_id}")

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None:
        self.events.append("upsert")
        self.upserted.extend(visuals)

    async def close(self) -> None:
        self.closed = True
```

Use the existing `FrameSampler` structural contract for a fake video sampler. Keep media fixture dimensions small so region buffers stay tiny.

Run:

```bash
uv run pytest tests/src/winston/indexing/test_pipeline.py -q
```

Expected: FAIL because orchestration models/pipeline do not exist.

- [ ] **Step 2: Implement run-result models with explicit failure stages**

Create `src/winston/indexing/models.py`:

```python
from dataclasses import dataclass, field
from enum import StrEnum


class IndexingRunError(RuntimeError):
    """Raised for fatal initialization or cleanup failures that invalidate the whole run."""


class PipelineStage(StrEnum):
    """Asset pipeline stage retained in actionable failure diagnostics."""

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
    """One failed media asset with stage, message, and original exception object."""

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
        """Return 0 only when no asset failed."""
        return 1 if self.failures else 0
```

Export these public types from `src/winston/indexing/__init__.py`.

- [ ] **Step 3: Write the photo-path orchestration test first**

Monkeypatch `winston.indexing.pipeline.scan_media`, `probe_media`, `identify_asset`, and `load_image` so the test exercises orchestration without FFmpeg/Jina/Qdrant. Assert all of these exact outcomes:

```python
result = await pipeline.run()

assert result.indexed == 1
assert result.skipped == 0
assert result.failed == 0
assert index.events[0] == "ensure"
assert index.events[1].startswith("delete:photo.jpg:")
assert "upsert" in index.events
assert all(visual.media_type is MediaType.IMAGE for visual in index.upserted)
assert all(visual.sample_kind.value == "image" for visual in index.upserted)
assert all(visual.timestamp_seconds is None for visual in index.upserted)
```

The manifest must contain the asset ID only after all upserts return successfully.

- [ ] **Step 4: Implement the common region -> embedding -> IndexedVisual path**

In `src/winston/indexing/pipeline.py`, use `itertools.batched()` exactly at the per-sample region boundary:

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
    """Generate, embed, and persist one photo or keyframe without unbounded buffering."""
    region_batches = batched(
        generate_regions(width=width, height=height, rgb24=rgb24),
        int(self._settings.indexing.visual_batch_size),
    )

    while True:
        self._stage = PipelineStage.REGIONS
        try:
            regions = next(region_batches)
        except StopIteration:
            break

        self._stage = PipelineStage.EMBEDDING
        embedded = await self._embedder.embed_images(regions)
        if embedded.count != len(regions):
            raise IndexingRunError(
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

Do not build a list of all regions for a frame/video. `next(region_batches)` may create at most `visual_batch_size` region crops at once.

- [ ] **Step 5: Implement one-asset processing with delete-before-probe/index and durable completion last**

Use this exact ordering inside `_process_asset()`:

```text
identify_asset
if completed -> skip immediately
set stage=DELETE -> await delete_old_revisions
set stage=PROBE -> probe_media
image:
  set stage=DECODE -> load_image
  _index_sample(media_type=image, sample_kind=image, timestamp=None)
video:
  set stage=SAMPLING
  async for frame in frame_sampler.sample(video):
      _index_sample(media_type=video, sample_kind=keyframe, timestamp=frame.timestamp_seconds)
set stage=MANIFEST -> mark_completed
```

The skip check must occur before `delete_old_revisions`, `probe_media`, photo decode, FFmpeg sampling, embedding, or Qdrant upsert.

After `mark_completed()` succeeds, add the asset ID to the in-memory completed set so a duplicate media discovery in the same process cannot reprocess it.

- [ ] **Step 6: Add video streaming, batch-size, restart, and failure-isolation tests**

Add these concrete cases to `test_pipeline.py`:

1. `test_video_streams_each_keyframe_through_common_visual_pipeline`: fake sampler yields two `SampledFrame` objects; assert Qdrant visuals carry `MediaType.VIDEO`, `SampleKind.KEYFRAME`, and timestamps in source order.
2. `test_visual_batch_size_nine_is_never_exceeded`: choose image dimensions that produce more than nine regions and assert `max(embedder.image_batch_sizes) <= 9` and at least two embedding calls occur.
3. `test_completed_asset_is_skipped_before_probe`: pre-populate the manifest for the current `index_instance_id`, make fake `probe_media` raise if called, and assert `indexed=0`, `skipped=1`, `failed=0`.
4. `test_non_completed_asset_is_retried`: first run injects an upsert failure before manifest append; second run with the same deterministic identity succeeds and appends completion.
5. `test_changed_revision_deletes_old_points_before_first_upsert`: assert the fake index event order has `delete:` before every `upsert` event.
6. `test_delete_failure_prevents_new_upserts_for_that_asset`: fake delete raises; assert zero upserts and one failure with stage `DELETE`.
7. `test_asset_failure_does_not_stop_following_asset`: first media probe raises, second media succeeds; assert `indexed=1`, `failed=1` and both assets were attempted sequentially.
8. `test_manifest_failure_reports_asset_and_retries_next_run`: fake manifest append raises after upsert; assert failure stage `MANIFEST`; second run performs deterministic upsert again.
9. `test_assets_are_processed_sequentially`: record start/end events around fake video sampling for two videos and assert all events for asset A finish before asset B begins.

Run:

```bash
uv run pytest tests/src/winston/indexing/test_pipeline.py -q
```

Expected: PASS after the implementation is complete.

- [ ] **Step 7: Implement long-lived dependency lifecycle in `run_indexing()`**

`run_indexing(root, settings)` constructs exactly one embedder, one `QdrantVisualIndex`, and one `KeyframeSampler`, then reuses them for the run:

```python
async def run_indexing(root: Path, settings: Settings) -> IndexRunResult:
    """Build one reusable indexing stack, run it, and close resources without masking failures."""
    embedder = create_embedder(settings)
    visual_index = QdrantVisualIndex(settings.qdrant)
    sampler = KeyframeSampler()
    pipeline = IndexingPipeline(
        root=root,
        settings=settings,
        embedder=embedder,
        visual_index=visual_index,
        frame_sampler=sampler,
    )
    ...
```

Close the Qdrant index and embedder whether the pipeline succeeds or raises. If the pipeline already raised, attach cleanup failures with `BaseException.add_note()` and re-raise the primary exception. If the pipeline succeeded but either close fails, raise `IndexingRunError` describing the cleanup failure so the CLI returns non-zero.

Add tests that assert both `close()` methods are called after a successful run and after a fatal initialization/pipeline error. Add a test where the pipeline error and close error both occur; assert the primary exception remains the raised exception and the cleanup message appears in its notes.

- [ ] **Step 8: Run indexing unit tests and commit Task 4**

```bash
uv run pytest tests/src/winston/indexing -q
git add src/winston/indexing tests/src/winston/indexing
git commit -m "feat: orchestrate restartable media indexing"
```

Expected: PASS.

---

### Task 5: Wire `winston index` into the CLI with actionable summaries and exit codes

**Files:**
- Modify: `src/winston/cli.py`
- Modify: `tests/src/winston/test_cli.py`

**Interfaces:**
- Consumes: `run_indexing(root, settings) -> IndexRunResult`
- Produces: `winston index [path]`
- Exit codes: `0` when all assets indexed/skipped successfully, `1` for asset failures or fatal run-level failures

- [ ] **Step 1: Write failing parser and summary tests**

Extend `tests/src/winston/test_cli.py` with an injected fake `run_indexing`:

```python
def test_index_command_is_available_and_uses_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The high-level command must pass the resolved root into the async runner."""
    seen: list[Path] = []

    async def fake_run_indexing(root: Path, settings: Settings) -> IndexRunResult:
        seen.append(root)
        return IndexRunResult(indexed=1, skipped=0, failures=())

    monkeypatch.setattr("winston.cli.run_indexing", fake_run_indexing)

    assert main(["index", str(tmp_path)]) == 0
    assert seen == [tmp_path.resolve()]
```

Add a failure-summary test with:

```python
failure = AssetFailure(
    source_path="cameras/cam12.mkv",
    stage=PipelineStage.SAMPLING,
    message="ffmpeg failed to decode keyframes",
    cause=RuntimeError("decoder error"),
)
result = IndexRunResult(indexed=3, skipped=2, failures=(failure,))
```

Assert output contains exactly the meaningful fields:

```text
Indexed: 3
Skipped: 2
Failed: 1

FAILED cameras/cam12.mkv [sampling]
  ffmpeg failed to decode keyframes
```

and `main(...) == 1`.

Add a fatal-run test where `run_indexing` raises `IndexingRunError("manifest corrupt")`; assert exit `1` and stderr identifies the fatal message.

- [ ] **Step 2: Refactor parser construction so one settings object supplies defaults and execution**

Change the parser signature to accept an optional settings instance:

```python
def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """Build Winston's CLI parser from one runtime settings snapshot."""
    config = settings or get_config()
    ...
```

Register:

```python
index_parser = subparsers.add_parser(
    "index",
    help="Index local videos and photos into the configured visual index",
)
index_parser.add_argument("path", nargs="?", type=Path, default=config.data_dir)
```

Change `main()` to build one `Settings` object and pass it both to `build_parser()` and `index_command()` so environment/config cannot differ between default-path parsing and runtime construction.

- [ ] **Step 3: Implement `index_command()` and summary formatting**

Add:

```python
def _print_index_result(result: IndexRunResult) -> None:
    """Print one concise Phase 1E indexing summary and actionable asset failures."""
    print(f"Indexed: {result.indexed}")
    print(f"Skipped: {result.skipped}")
    print(f"Failed: {result.failed}")
    for failure in result.failures:
        print()
        print(f"FAILED {failure.source_path} [{failure.stage.value}]")
        print(f"  {failure.message}")


def index_command(root: Path, settings: Settings) -> int:
    """Run the asynchronous indexing pipeline and map its outcome to a CLI exit code."""
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError(f"indexing root is not a directory: {resolved_root}")

    try:
        result = asyncio.run(run_indexing(resolved_root, settings))
    except Exception as exc:
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1

    _print_index_result(result)
    return result.exit_code
```

Do not catch `KeyboardInterrupt`; interruption should retain normal shell semantics while manifest/Qdrant idempotency makes the next invocation safe.

- [ ] **Step 4: Run CLI/config/indexing tests and commit Task 5**

```bash
uv run pytest tests/src/winston/test_cli.py tests/src/winston/test_config.py tests/src/winston/indexing -q
git add src/winston/cli.py tests/src/winston/test_cli.py
git commit -m "feat: expose restartable index CLI"
```

Expected: PASS.

---

### Task 6: Add real-Qdrant Phase 1E integration confidence and finish the PR

**Files:**
- Modify: `tests/integration/test_qdrant_visual_index.py`
- Create: `tests/integration/test_restartable_index_pipeline.py`
- Modify only if required by existing CI wiring: `.github/workflows/*`

**Interfaces:**
- Verifies: schema-v2 ownership metadata against Qdrant 1.18.x
- Verifies: stale-revision deletion
- Verifies: first run indexes and second run skips without increasing point count
- Verifies: recreated collection receives a new `index_instance_id`, causing prior manifest completions to be ignored

- [ ] **Step 1: Update the existing real-Qdrant test to schema version 2**

In `tests/integration/test_qdrant_visual_index.py`, add a stable dataset UUID for the test and call:

```python
DATASET_ID = "8f7ad0c0-7ab7-4ec0-9025-22f40dd70c4a"

session = await index.ensure_compatible(IDENTITY, DATASET_ID)
```

Assert collection metadata includes:

```python
assert info.config.metadata is not None
assert info.config.metadata["winston_schema_version"] == 2
assert info.config.metadata["dataset_instance_id"] == DATASET_ID
assert info.config.metadata["index_instance_id"] == session.index_instance_id
assert str(UUID(session.index_instance_id)) == session.index_instance_id
```

Keep the existing vector-size, cosine-distance, payload, and deterministic-upsert checks.

- [ ] **Step 2: Add a real-Qdrant stale-revision deletion test**

Create two `IndexedVisual` values with the same `source_path` and different 64-char `asset_id` values. Upsert both, then call:

```python
await index.delete_old_revisions(
    source_path="cameras/integration/cam01.mkv",
    current_asset_id="f" * 64,
)
```

Query/count with Qdrant and assert only the `f * 64` revision remains for that path. This proves the real SDK filter matches the unit-test fake contract.

- [ ] **Step 3: Add an end-to-end restart smoke test with a fake embedder and real Qdrant**

Create `tests/integration/test_restartable_index_pipeline.py` using:

- `tmp_path` as the dataset root;
- one Pillow-generated JPG;
- the real `scan_media`, `probe_media`, `load_image`, regions, asset identity, manifest, and `QdrantVisualIndex`;
- a deterministic fake embedder returning 768-dimensional float32 zero vectors;
- a unique integration collection name such as `winston_restartable_index_integration`.

First run assertions:

```python
first = await pipeline.run()
assert first.indexed == 1
assert first.skipped == 0
assert first.failed == 0
first_count = (await client.count(collection, exact=True)).count
assert first_count > 0
```

Construct a new `IndexingPipeline` against the same root/collection and run again:

```python
second = await pipeline_again.run()
assert second.indexed == 0
assert second.skipped == 1
assert second.failed == 0
second_count = (await client.count(collection, exact=True)).count
assert second_count == first_count
```

Then delete/recreate the test collection, run against the same local manifest, and assert the new collection's `index_instance_id` differs and the asset is indexed again rather than skipped.

Use `try/finally` in the integration test to delete its dedicated test collection so reruns start cleanly.

- [ ] **Step 4: Run the full locked unit suite**

```bash
uv lock --check
uv run pytest tests/src -q
```

Expected: all unit tests PASS.

- [ ] **Step 5: Run real-Qdrant integration tests against the pinned service**

Start only the existing Qdrant service:

```bash
docker compose up -d qdrant
```

Run:

```bash
uv run pytest tests/integration/test_qdrant_visual_index.py tests/integration/test_restartable_index_pipeline.py -q
```

Expected: PASS against Qdrant 1.18.x.

Stop the local test service only if the execution environment owns it; do not tear down unrelated developer services.

- [ ] **Step 6: Run full regression and syntax checks**

```bash
uv run pytest -q
uv run python -m compileall -q src tests
uv lock --check
```

Expected: PASS with no regression in ingest, sampling, embeddings, recorder, index, or search tests.

- [ ] **Step 7: Review the diff against the Phase 1E spec before opening the PR**

Check all of these explicitly:

```text
[ ] visual_batch_size defaults to 9
[ ] photos decode through sampling, not ingest
[ ] one asset at a time
[ ] video keyframes remain streamed
[ ] dataset.json is atomic and stable
[ ] index-state.jsonl only records completed assets
[ ] completion append flushes + fsyncs
[ ] only final malformed JSONL line may be ignored
[ ] collection metadata schema is 2
[ ] dataset_instance_id mismatch is fatal
[ ] index_instance_id scopes manifest completions
[ ] schema-1 Qdrant collection is rejected, never migrated automatically
[ ] old source-path revisions are deleted before indexing a replacement
[ ] delete failure prevents new writes for that asset
[ ] deterministic point IDs remain unchanged
[ ] failed asset does not stop later assets
[ ] close failures produce non-zero outcome without masking an earlier exception
[ ] CLI prints Indexed / Skipped / Failed and per-asset stage/path
[ ] media mutation during indexing is not implemented
```

- [ ] **Step 8: Commit integration coverage**

```bash
git add tests/integration
git commit -m "test: cover restartable indexing with Qdrant"
```

- [ ] **Step 9: Push the feature branch and open one draft PR for issue #9**

```bash
git push -u origin feat/restartable-index-cli
```

Open a draft PR titled:

```text
feat: add restartable index CLI
```

The PR body must include at least:

```markdown
## Summary
- compose ingest, sampling, deterministic regions, embeddings, and Qdrant behind `winston index`
- add asset-level durable restart state and dataset/collection ownership identities
- support both videos and JPG/JPEG photos with bounded region batching
- delete stale revisions before indexing changed assets

## Validation
- `uv lock --check`
- `uv run pytest -q`
- real Qdrant 1.18 integration tests

Closes #9
```

Keep the PR in draft until implementation review, automated tests, and relevant GitHub Actions are green.
