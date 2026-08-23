# Phase 1E — Restartable index CLI design

## Status

Approved design for GitHub issue #9: **Phase 1E — Add restartable index CLI**.

This phase composes Winston's existing ingest, sampling, deterministic-region, embedding, and Qdrant layers behind one high-level command:

```text
winston index <path>
```

The command is restartable at asset granularity and idempotent across repeated runs.

## Goals

Phase 1E must:

- expose the complete V0 media-to-Qdrant indexing flow through `winston index <path>`;
- support both recorded videos and JPG/JPEG photos;
- process assets with bounded memory usage;
- skip assets that already completed against the current Qdrant collection instance;
- safely retry an asset that was interrupted before completion;
- preserve deterministic Qdrant point identity across retries;
- remove stale points when a file at an existing `source_path` has a new `asset_id`;
- continue indexing later assets when one asset fails;
- report failures with the affected media path and return a non-zero exit status when any asset fails;
- prevent one dataset from accidentally sharing a visual collection with a different dataset;
- remain testable without requiring real FFmpeg, Jina, or Qdrant for orchestration tests.

## Non-goals and preconditions

Phase 1E does not implement:

- concurrent processing of multiple assets;
- distributed indexing workers;
- per-keyframe or per-region checkpoints;
- a generic job/task framework;
- concurrent `winston index` processes against the same dataset/collection;
- automatic re-creation of incompatible Qdrant collections;
- semantic search;
- automatic handling of media files that change while they are being indexed.

The V0 indexing command has one explicit data precondition:

> Source media are immutable for the duration of one `winston index` run.

The recorder or another producer must therefore stop writing a file before that file is indexed. Winston does not re-stat the media at the end of processing to detect mid-run mutation.

## Chosen architecture

The CLI remains thin. A dedicated indexing orchestration layer composes the already-existing domain components.

```text
src/winston/
├── cli.py
├── ingest/
│   ├── identity.py
│   ├── probe.py
│   └── scanner.py
├── sampling/
│   ├── images.py          # decode JPG/JPEG to RGB24
│   ├── keyframes.py
│   └── regions.py
├── embeddings/
│   └── ...
├── indexing/
│   ├── __init__.py
│   ├── manifest.py        # dataset identity + append-only completion journal
│   ├── models.py          # run results/failures
│   └── pipeline.py        # end-to-end orchestration
└── index/
    ├── base.py
    ├── models.py
    └── qdrant.py          # collection lifecycle, revision cleanup, upsert
```

The new `indexing` package owns orchestration state. The existing `index` package remains the persistence boundary for visual vectors.

The alternative of placing the entire flow in `cli.py` was rejected because it would couple argument parsing, process lifecycle, restart state, media processing, embedding, and persistence in one place. A generic job framework was also rejected as unnecessary for V0.

## High-level data flow

```text
winston index <root>
        ↓
load/create dataset identity
        ↓
create embedder once
        ↓
open/validate Qdrant collection once
        ↓
load completion manifest for current index_instance_id
        ↓
scan_media(root)
        ↓
process each asset sequentially
        │
        ├─ identify_asset()
        ├─ already completed? ──────────────→ skip
        ├─ delete old revisions for source_path
        ├─ probe_media()
        ├─ video → stream keyframes
        │   image → decode one RGB24 photo
        ├─ generate deterministic regions
        ├─ batch up to 9 visual regions
        ├─ embed_images(batch)
        ├─ build IndexedVisual values
        ├─ Qdrant upsert
        └─ durable manifest append: completed
        ↓
close embedder + Qdrant
        ↓
print summary and return exit status
```

Assets are processed one at a time. The command never intentionally runs multiple FFmpeg decoders or multiple media assets concurrently.

## Dataset identity

### Purpose

`source_path` and `asset_id` are relative to the indexing root. Two independent datasets can therefore contain the same relative path, for example:

```text
/dataset-A/cameras/cam01.mkv
/dataset-B/cameras/cam01.mkv
```

Both would otherwise produce:

```text
source_path = "cameras/cam01.mkv"
```

Because Phase 1E deletes stale revisions by `source_path`, allowing unrelated datasets to share one Qdrant collection would be unsafe.

### V0 rule

One Qdrant visual collection belongs to exactly one Winston dataset identity.

The dataset stores a stable UUID in:

```text
<root>/.winston/dataset.json
```

Example:

```json
{
  "schema_version": 1,
  "dataset_instance_id": "9bd91e6f-2bff-4f3b-9420-aacddc7198d9"
}
```

The file is created once using an atomic write. The UUID is not derived from the absolute filesystem path.

This means a complete dataset can be moved to another directory or machine while preserving `.winston/` and still retain the same dataset identity.

Copying `.winston/dataset.json` intentionally copies the dataset identity as well. Winston treats such a copy as another location of the same logical dataset, not as a new independent dataset.

## Qdrant collection instance identity

### Why it is required

A local completion manifest alone is insufficient. If the Qdrant collection is deleted and recreated, a stale manifest could otherwise claim that assets are already indexed while the new collection is empty.

Each Qdrant collection therefore receives a random, stable:

```text
index_instance_id = UUIDv4
```

The value is generated only when Winston creates the collection and remains unchanged for that collection's lifetime.

### Collection ownership metadata

Phase 1E extends Winston collection metadata with:

```text
winston_schema_version = 2
dataset_instance_id = <dataset UUID>
index_instance_id = <collection UUID>
model_id = "jinaai/jina-clip-v1"
dimension = 768
preprocessing_version = 1
vector_name = "visual"
distance = "cosine"
```

The schema version is incremented from Phase 1D because the collection now carries required dataset ownership and restart-generation semantics.

A pre-Phase-1E collection that has schema version 1 or lacks `dataset_instance_id` / `index_instance_id` is incompatible. Winston must reject it with an explicit reindex-required error rather than silently adopting it.

This preserves the existing Phase 1D rule: Winston never automatically drops or recreates an incompatible visual collection.

### Collection validation

Opening the collection requires all of the following to match:

- named vector name;
- vector dimension;
- cosine distance;
- embedding model identity;
- preprocessing version;
- Winston schema version;
- `dataset_instance_id`.

`index_instance_id` must exist and be a valid UUID, but it is read from the existing collection rather than compared to a locally preselected value. It identifies that exact collection instance.

If `dataset_instance_id` differs, Winston refuses to index and reports that the collection belongs to another dataset.

## VisualIndex session contract

Phase 1E needs the persistence layer to expose the collection instance identity and stale-revision cleanup without leaking Qdrant SDK types.

The Winston-owned interface evolves conceptually to:

```python
@dataclass(frozen=True, slots=True)
class VisualIndexSession:
    index_instance_id: str


class VisualIndex(Protocol):
    async def ensure_compatible(
        self,
        identity: EmbeddingIdentity,
        dataset_instance_id: str,
    ) -> VisualIndexSession: ...

    async def delete_old_revisions(
        self,
        *,
        source_path: str,
        current_asset_id: str,
    ) -> None: ...

    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None: ...

    async def close(self) -> None: ...
```

The exact value type may use UUIDs internally, but public Winston contracts must remain independent from Qdrant models.

`delete_old_revisions()` removes points satisfying:

```text
source_path == current source path
AND
asset_id != current asset id
```

Deletion is synchronous from the orchestration point of view: Winston waits for Qdrant to confirm the delete before indexing the new revision.

If deletion fails, the new revision is not indexed during that run.

## Restart manifest

### Files

Phase 1E stores local restart state under the indexing root:

```text
<root>/.winston/
├── dataset.json
└── index-state.jsonl
```

The scanner already only accepts supported media extensions, so `.winston` cannot become an indexable media asset.

### Journal format

`index-state.jsonl` is append-only. It records only successfully completed assets.

Example record:

```json
{"schema_version":1,"index_instance_id":"0f27f24e-7436-4c46-9260-4a37f419b3a6","asset_id":"0b9d...","source_path":"cameras/cam01.mkv","status":"completed"}
```

There are intentionally no durable `running` or `failed` records. They do not contribute to restart correctness.

### Completion rule

A completion record is appended only after the complete asset pipeline has succeeded, including all Qdrant upserts.

The append is considered successful only after:

```text
write complete JSON line + newline
flush()
os.fsync()
```

If manifest persistence fails after Qdrant writes succeeded, the asset is reported as failed and is retried on the next run. Deterministic point IDs make that retry idempotent.

### Loading state

At startup the manifest loader reads only completion records whose `index_instance_id` matches the collection session returned by `ensure_compatible()`.

Therefore:

- same dataset + same collection instance → completed assets can be skipped;
- collection deleted/recreated → new `index_instance_id`, so old completion records are ignored;
- old history may remain in the append-only file without affecting the current run.

Completion is keyed by `asset_id`, not only `source_path`. If the same file path receives a new size/mtime and therefore a new `asset_id`, it is not considered completed.

### Crash-damaged journal

A crash can interrupt the last append. The loader may ignore one malformed/truncated final non-empty line because it cannot represent a durably completed record.

Malformed JSON or an invalid record anywhere before the final line is treated as manifest corruption and causes a fatal, actionable error. Winston must not silently ignore corruption in the durable history.

No compaction is required in Phase 1E.

## Asset identity and changed revisions

Phase 1D already defines:

```text
asset_id = SHA-256(
    versioned canonical JSON of:
      relative_path
      file_size
      mtime_ns
)
```

Phase 1E keeps this identity unchanged.

When the current file has an `asset_id` that is not completed, Winston first asks the visual index to remove all other revisions for the same `source_path`.

The order is intentionally:

```text
identify current asset
        ↓
delete points for same source_path + different asset_id
        ↓
index current asset
        ↓
mark completed
```

This is a product decision for V0: the old revision does not need to remain searchable while its replacement is built.

If the process crashes after the delete and before the new asset completes, that media may temporarily be absent or only partially present in Qdrant. On restart, the asset is not completed and is processed again. Partial points for the same current `asset_id` are safely overwritten because point IDs are deterministic.

## Photo decoding

The ingest layer already discovers and probes JPG/JPEG files but does not currently expose decoded photo pixels.

Photo decoding belongs in `sampling`, not `ingest`:

```text
ImageMetadata
      ↓
sampling.images.load_image(...)
      ↓
RGB24 image sample
      ↓
generate_regions(...)
```

The decoder uses Pillow, which is already an explicit project dependency.

The photo loader must:

- decode the complete image;
- convert explicitly to RGB;
- expose width, height, and contiguous RGB24 bytes;
- validate the decoded dimensions against probed image metadata;
- report decode/validation failures with the affected path.

A small frozen `SampledImage` value type may mirror the existing `SampledFrame` structure without a timestamp.

Photo pixels are then fed into the same deterministic region-generation and embedding flow as video keyframes.

## Video processing

Videos continue to use `KeyframeSampler.sample(video)` as an asynchronous streaming iterator.

The orchestrator must not materialize all keyframes for a video. It consumes one keyframe at a time:

```text
video
  ↓
next keyframe
  ↓
regions
  ↓
embedding batches
  ↓
Qdrant
  ↓
next keyframe
```

This preserves the bounded-memory behavior established in Phase 1A.

## Region and embedding batching

Phase 1E adds an orchestration setting:

```text
indexing.visual_batch_size = 9
```

Configuration shape:

```text
Settings
├── embedding = ...
├── qdrant = ...
└── indexing
    └── visual_batch_size = 9
```

Environment override:

```text
INDEXING__VISUAL_BATCH_SIZE=9
```

The value is a positive integer.

This is an orchestration/memory bound, not a provider-specific embedding batch size. The embedding backend remains free to apply its own internal batching/backoff rules.

For each photo or keyframe:

```text
generate_regions(...)
        ↓
itertools.batched(..., visual_batch_size)
        ↓
embed_images(region_batch)
        ↓
map vectors back to regions
        ↓
IndexedVisual batch
        ↓
VisualIndex.upsert(...)
```

Winston never accumulates every region from every frame in a video.

## Common visual mapping

Both media types converge on the same per-sample region pipeline.

For a photo:

```text
media_type = image
sample_kind = image
timestamp_seconds = None
```

For a video keyframe:

```text
media_type = video
sample_kind = keyframe
timestamp_seconds = SampledFrame.timestamp_seconds
```

Each generated `VisualRegion` becomes one `IndexedVisual` using:

- current `asset_id`;
- normalized relative `source_path`;
- media type;
- sample kind;
- optional timestamp;
- region kind;
- region geometry;
- one embedding vector;
- the embedder's validated `EmbeddingIdentity`.

The existing deterministic `visual_point_id()` remains the source of Qdrant point identity.

## Orchestrator responsibilities

`IndexingPipeline` owns one indexing run and is responsible for:

1. loading/creating the dataset state;
2. constructing or receiving long-lived dependencies;
3. validating Qdrant compatibility and obtaining `index_instance_id`;
4. loading completed asset IDs for that collection instance;
5. scanning media in deterministic order;
6. processing each asset sequentially;
7. continuing after asset-level failures;
8. durably marking only successful assets completed;
9. producing a structured run result.

The implementation should use dependency injection / narrow Protocols where this materially improves tests. It must not build a generic dependency container or job framework.

## Resource lifecycle

The expensive/reusable components are created once per CLI invocation:

- one multimodal embedder;
- one Qdrant visual index client.

They are reused for every asset and closed in `finally` paths.

Initialization errors that prevent a meaningful run are fatal, for example:

- dataset state cannot be read/created;
- manifest corruption;
- Qdrant is unreachable during initial compatibility setup;
- collection is incompatible;
- collection belongs to another dataset;
- embedder construction fails.

Fatal initialization errors stop before asset processing.

A close failure is a run-level failure and produces a non-zero exit status. Cleanup errors must not silently replace the diagnostic for an earlier primary failure.

## Asset-level failure policy

After successful initialization, failures are isolated per media asset whenever possible.

Examples include:

- asset identity error;
- media probe failure;
- stale-revision delete failure;
- image decode failure;
- FFmpeg/keyframe sampling failure;
- deterministic region generation failure;
- embedding failure;
- Qdrant upsert failure;
- manifest completion append failure.

The failed asset is not marked completed. Winston records the failure and continues to the next scanned media file.

The failure model should retain at least:

```text
source path
pipeline stage
human-readable exception message
```

The original exception chain should remain available for debugging/tests rather than being flattened too early.

## CLI behavior

`build_parser()` gains:

```text
winston index [path]
```

As with `scan`, the path defaults to configured `data_dir` when omitted.

The command resolves and validates the indexing root, runs the asynchronous indexing pipeline, and prints a concise final summary.

Example:

```text
Indexed: 127
Skipped: 34
Failed: 2

FAILED cameras/cam12.mkv [sampling]
  ffmpeg failed to decode keyframes: ...

FAILED photos/broken.jpg [decode]
  Pillow could not decode image: ...
```

Exit status:

```text
0  all discovered assets succeeded or were already completed
1  one or more asset failures or a fatal run-level failure occurred
```

No additional progress UI is required for Phase 1E.

## Restart and idempotency semantics

### Repeating a successful run

For an unchanged dataset and unchanged Qdrant collection instance:

```text
asset_id is in completed set
        ↓
skip before probe/decode/embedding
```

No duplicate Qdrant points are created.

### Interruption during one asset

If the process stops before the durable completion append:

- earlier completed assets remain skippable;
- the interrupted asset has no completion record;
- any already-written points use deterministic IDs;
- the next run processes that asset again and overwrites those points idempotently.

Checkpoint granularity is deliberately one complete media asset.

### Recreated Qdrant collection

A recreated collection receives a new `index_instance_id`.

Existing manifest lines refer to the former ID and are ignored. Every currently discovered asset is therefore indexed into the new empty collection.

### Changed source revision

A changed size or `mtime_ns` produces a new `asset_id`.

The new identity does not match the old completion record, so Winston deletes all prior points for that same `source_path` before indexing the replacement.

## Testing strategy

Tests follow Winston's existing convention of exercising small units and using fakes for external processes/services where possible.

### Dataset and manifest tests

Cover:

- new `dataset.json` creation;
- stable dataset UUID across runs;
- atomic/read validation behavior;
- append and reload of completed assets;
- completion filtering by `index_instance_id`;
- old collection instance records ignored;
- truncated final JSONL record ignored;
- corruption before the final line rejected;
- manifest append is flushed/fsynced before success is returned.

### Qdrant index tests

Extend Phase 1D coverage for:

- schema version 2 metadata;
- collection creation generates and persists `index_instance_id`;
- existing collection returns the stored instance ID;
- dataset ID mismatch is rejected;
- missing/invalid instance metadata is rejected;
- schema-1 collection is rejected with reindex guidance;
- `delete_old_revisions()` sends the intended source-path/current-asset filter;
- delete waits for completion;
- delete failure is surfaced without an upsert.

### Photo sampling tests

Cover:

- RGB JPEG decoding;
- non-RGB image conversion to RGB;
- RGB24 byte length and dimensions;
- decoded/probed dimension mismatch;
- corrupt image error includes the path.

### Orchestration tests

Using fake scanner/probe/sampler/embedder/index/manifest boundaries where appropriate, cover:

- photo → regions → embeddings → Qdrant;
- video → streamed keyframes → regions → embeddings → Qdrant;
- `visual_batch_size=9` is respected;
- assets are processed sequentially;
- successful asset is durably marked completed;
- already-completed asset is skipped before expensive stages;
- interrupted/non-completed asset is processed on the next run;
- partial prior upserts are harmless on retry;
- changed `source_path` revision triggers delete before any new upsert;
- delete failure prevents new writes for that asset;
- one asset failure does not prevent later assets from being attempted;
- run result counts indexed/skipped/failed correctly;
- embedder and visual index are closed on success and failure.

### CLI tests

Cover:

- parser exposes `index`;
- configured `data_dir` remains the default path;
- successful/no-op run exits 0;
- run with asset failures exits 1;
- summary and failure diagnostics include relative media paths.

### Integration confidence

The implementation PR should retain the full existing Winston test suite and add focused Phase 1E tests. A real Qdrant integration check should verify collection ownership metadata, deletion, upsert, restart, and second-run idempotency against the pinned Qdrant 1.18 line.

No test is required for a media file mutating during indexing because media immutability is an explicit V0 precondition.

## Acceptance criteria mapping

Issue #9 acceptance criteria map directly to the design:

- **same dataset twice does not duplicate points** → completion manifest + deterministic UUIDv5 point IDs;
- **interrupted indexing restarts safely** → asset-level durable completion records + idempotent upserts;
- **video keyframes and photos use one command** → common orchestration with Pillow photo decoding and streaming keyframes;
- **failures identify the asset** → structured per-asset failures and final CLI summary;
- **tests cover orchestration and idempotency** → manifest, Qdrant, sampling, pipeline, CLI, and real-Qdrant coverage above.

## PR contract

Phase 1E remains one focused implementation PR for issue #9.

The PR description must contain:

```text
Closes #9
```

The implementation PR should remain draft until its implementation, automated tests, and relevant GitHub Action checks are ready for review.
