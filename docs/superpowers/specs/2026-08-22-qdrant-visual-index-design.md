# Phase 1D — Qdrant visual index design

## Status

Approved design for GitHub issue #8: **Phase 1D — Implement Qdrant visual index**.

This phase adds the persistence boundary between Winston's class-agnostic visual embeddings and Qdrant. It does not yet build the complete media-to-index pipeline or search API.

## Goals

Phase 1D must provide a reliable visual index that:

- stores one Qdrant point for each full image/keyframe or deterministic region;
- uses deterministic identifiers so repeated indexing is idempotent;
- stores enough provenance to reconstruct the exact source media, timestamp, and region;
- rejects an existing collection when its vector or Winston embedding identity is incompatible;
- never deletes or recreates an incompatible collection automatically;
- keeps raw media outside Qdrant;
- exposes a Winston-owned interface rather than leaking Qdrant client types through the rest of the application.

## Non-goals

This phase does not implement:

- the complete `media -> sampling -> embedding -> Qdrant` orchestration pipeline;
- a CLI indexing command;
- semantic search against Qdrant;
- result grouping or reranking;
- automatic collection deletion or reindexing;
- indexing jobs or progress tracking;
- distributed workers.

Those remain later roadmap work.

## Architecture

Winston owns the contracts and value types. Qdrant is the V0 implementation behind those contracts.

```text
src/winston/
├── ingest/
│   └── identity.py          # stable asset identity relative to an index root
│
└── index/
    ├── __init__.py
    ├── base.py              # VisualIndex protocol
    ├── models.py            # Winston index value types and errors
    ├── identity.py          # deterministic UUIDv5 point identity
    └── qdrant.py            # AsyncQdrantClient implementation and mapping
```

The intended data flow is:

```text
Media
  ↓
asset_id
  ↓
SampledFrame / photo
  ↓
VisualRegion
  ↓
EmbeddingBatch
  ↓
IndexedVisual
  ↓
VisualIndex.upsert(...)
  ↓
QdrantVisualIndex
```

The index layer accepts already-produced visual embeddings. It does not own decoding, sampling, region generation, or embedding inference.

## VisualIndex contract

The application-facing contract is a small protocol:

```python
class VisualIndex(Protocol):
    async def ensure_compatible(self, identity: EmbeddingIdentity) -> None: ...
    async def upsert(self, visuals: Sequence[IndexedVisual]) -> None: ...
    async def close(self) -> None: ...
```

`ensure_compatible()` establishes or validates the collection before indexing begins and records the compatible `EmbeddingIdentity` for the index session.

`upsert()` accepts Winston value types and performs bounded, idempotent writes. It requires a successful `ensure_compatible()` first. Every visual in the batch must use the same embedding identity that was validated for the session; otherwise the call fails before writing anything. An empty batch is a no-op.

`close()` releases the reusable Qdrant client.

No Qdrant SDK models appear in the public protocol.

## Configuration

`Settings` gains one nested Qdrant configuration object:

```text
Settings
├── embedding = ...
└── qdrant
    ├── url = "http://localhost:6333"
    ├── collection = "winston_visual"
    ├── vector_name = "visual"
    └── upsert_batch_size = 256
```

With the existing nested environment convention, these map to values such as:

```text
QDRANT__URL=http://localhost:6333
QDRANT__COLLECTION=winston_visual
QDRANT__VECTOR_NAME=visual
QDRANT__UPSERT_BATCH_SIZE=256
```

The server in `compose.yml` remains pinned to `qdrant/qdrant:v1.18.2`.

The Python dependency is constrained to the matching client line:

```text
qdrant-client >=1.18,<1.19
```

This phase uses `AsyncQdrantClient` and reuses one client for the index lifetime.

## Qdrant collection contract

### Collection

```text
collection: winston_visual
named vector: visual
size: 768
metric: Cosine
```

The vector name and collection name are configurable, but these are the V0 defaults required by issue #8.

Qdrant 1.18.2 supports arbitrary collection-level metadata. Winston uses that native mechanism instead of creating a sentinel metadata point.

### Collection metadata

The collection carries these Winston-owned compatibility fields:

```text
winston_schema_version = 1
model_id = "jinaai/jina-clip-v1"
dimension = 768
preprocessing_version = 1
vector_name = "visual"
distance = "cosine"
```

Compatibility requires every Winston field above to exist and match the requested index identity. Unrelated additional collection metadata may exist without making the collection incompatible.

The vector configuration itself is also validated independently from metadata. Metadata cannot claim compatibility when the actual Qdrant vector configuration differs.

## Collection lifecycle

`ensure_compatible()` follows this exact behavior:

```text
collection exists?
    │
    ├─ no
    │   └─ create collection
    │       ├─ named vector = configured vector name
    │       ├─ size = EmbeddingIdentity.dimension
    │       ├─ distance = Cosine
    │       └─ collection metadata = Winston compatibility identity
    │
    └─ yes
        └─ read collection info
            ├─ validate named vector exists
            ├─ validate vector size
            ├─ validate distance
            └─ validate Winston metadata
```

An existing collection is rejected when any required vector or Winston identity property is missing or incompatible.

In particular:

- a collection with no Winston metadata is incompatible;
- a collection with incomplete Winston metadata is incompatible;
- a different model is incompatible;
- a different dimension is incompatible;
- a different preprocessing version is incompatible;
- a different vector name or distance is incompatible.

Winston never drops, clears, renames, or recreates an incompatible collection automatically. The error tells the caller that explicit reindexing is required.

## Asset identity

### Purpose

`asset_id` identifies one concrete source media revision without hashing the complete media content.

For V0 it is the SHA-256 digest of stable canonical file metadata:

```text
relative_path
file_size
mtime_ns
```

The relative path is always computed from the explicit indexing root.

### Path canonicalization

For an indexing root `data/`:

```text
data/cameras/brussels/cam01.mkv
        ↓
source_path = "cameras/brussels/cam01.mkv"
```

The algorithm resolves both paths, verifies the source is inside the indexing root, derives `source.relative_to(root)`, and serializes the relative path with POSIX separators.

Consequences:

- moving an otherwise unchanged Winston dataset from one absolute machine path to another does not change IDs when the relative path and file metadata are preserved;
- Windows and POSIX path separators do not create different identities;
- a source outside the indexing root is rejected instead of receiving an ambiguous identity.

### Canonical asset representation

The identity name is versioned with the prefix:

```text
winston:asset:v1:
```

followed by canonical JSON. Conceptually:

```json
{
  "file_size": 123456789,
  "mtime_ns": 1787412345678900000,
  "relative_path": "cameras/brussels/cam01.mkv"
}
```

The JSON representation uses UTF-8, sorted keys, fixed separators `(',', ':')`, and no insignificant whitespace. SHA-256 hashes the complete `winston:asset:v1:` prefix plus canonical JSON bytes.

The resulting `asset_id` is a lowercase hexadecimal SHA-256 string.

Changing the path, size, or nanosecond mtime changes `asset_id`. Re-reading the same unchanged file under the same indexing root produces exactly the same `asset_id`.

## IndexedVisual model

One `IndexedVisual` represents one already-embedded full image/keyframe or region.

It contains at least:

```text
asset_id
source_path          # normalized path relative to the indexing root
media_type           # image | video
sample_kind          # image | keyframe
timestamp_seconds    # None for photos
region_kind          # full | tile
region:
    x
    y
    width
    height
    scale
vector               # float32, exact embedding dimension
embedding_identity:
    model_id
    dimension
    preprocessing_version
```

The model validates enough invariants to prevent malformed points from crossing the Winston/Qdrant boundary, including:

- non-empty `asset_id` and relative `source_path`;
- valid media/sample-kind combinations (`image/image`, `video/keyframe`);
- `timestamp_seconds is None` for photos and a finite non-negative timestamp for video keyframes;
- positive region dimensions and non-negative coordinates;
- `full` regions use `x=0`, `y=0`, `scale=1.0`;
- the vector is one-dimensional `float32` and has exactly `embedding_identity.dimension` elements.

`sample_kind` is represented by a Winston enum rather than a free-form string.

## Timestamp canonicalization

Qdrant payloads keep human-friendly seconds:

```text
timestamp_seconds = 123.456
```

Deterministic identity never depends on the string representation of a binary float.

For identity purposes, Winston converts the finite non-negative timestamp to decimal using `Decimal(str(timestamp_seconds))`, multiplies by `1_000_000`, and rounds to the nearest integer microsecond with `ROUND_HALF_UP`.

```text
123.456 seconds
      ↓
123456000 microseconds
```

The canonical field is:

```text
timestamp_us: int | None
```

Photos use `None` for both timestamp fields.

## Deterministic Qdrant point IDs

Qdrant point IDs use UUIDv5 because it is deterministic and natively accepted by Qdrant.

Winston uses the standard namespace:

```python
uuid.NAMESPACE_URL
```

No custom Winston namespace UUID is created or maintained.

The UUID name begins with a versioned Winston prefix:

```text
winston:visual:v1:
```

and is followed by canonical UTF-8 JSON using sorted keys and fixed separators `(',', ':')` containing:

```text
asset_id
sample_kind
timestamp_us
region_kind
x
y
width
height
model_id
dimension
preprocessing_version
```

Conceptually:

```python
point_id = uuid.uuid5(uuid.NAMESPACE_URL, canonical_visual_name)
```

### Why `scale` is not part of the point ID

`scale` remains in the payload for provenance, but point identity uses the exact pixel rectangle (`x`, `y`, `width`, `height`).

If two sampling configurations happen to produce the exact same crop from the same temporal sample and embedding identity, Winston treats them as the same visual candidate rather than creating duplicate points solely because their declared scale differs.

### Embedding identity is part of the point ID

`model_id`, `dimension`, and `preprocessing_version` are part of point identity even though collection compatibility already prevents mixed incompatible vectors. This keeps the point identity semantically complete and deterministic independently of the current storage backend.

## Qdrant payload

Each point stores provenance only; raw media and RGB buffers are never included.

Example video tile:

```json
{
  "asset_id": "...sha256...",
  "source_path": "cameras/brussels/cam01.mkv",
  "media_type": "video",
  "sample_kind": "keyframe",
  "timestamp_seconds": 123.456,
  "timestamp_us": 123456000,
  "region_kind": "tile",
  "region": {
    "x": 1008,
    "y": 567,
    "width": 1344,
    "height": 756,
    "scale": 0.5
  },
  "model_id": "jinaai/jina-clip-v1",
  "dimension": 768,
  "preprocessing_version": 1
}
```

A photo uses:

```text
media_type = image
sample_kind = image
timestamp_seconds = null
timestamp_us = null
```

A full-frame candidate uses:

```text
region_kind = full
x = 0
y = 0
width = source width
height = source height
scale = 1.0
```

The embedding is stored only as the named Qdrant vector:

```text
visual -> float[768]
```

## Idempotent upsert behavior

`QdrantVisualIndex.upsert()` maps `IndexedVisual` values to Qdrant `PointStruct` values using deterministic UUIDv5 IDs.

Before constructing any Qdrant points, the whole supplied batch is checked against the embedding identity previously accepted by `ensure_compatible()`. A mixed or incompatible batch fails before the first network write.

Qdrant upsert semantics replace a point with the same ID, so repeated indexing of an unchanged candidate does not create duplicates.

The write path is:

```text
IndexedVisual[]
      ↓
validate complete batch identity
      ↓
map one bounded chunk
      ↓
sequential chunks of at most 256 points
      ↓
client.upsert(
    collection_name=...,
    points=...,
    wait=True,
)
```

The batch size is configurable and defaults to 256.

Writes are sequential in V0. No extra Qdrant write concurrency is introduced until measurement shows that Qdrant writes are a bottleneck.

`wait=True` is intentional: when `upsert()` returns, the operation has been applied rather than merely accepted for asynchronous execution. This simplifies restartable indexing semantics.

## Memory behavior

The index layer must not build a second unbounded copy of an indexing job.

It accepts a bounded sequence from the caller and materializes Qdrant point objects for at most one configured write chunk at a time. The default of 256 corresponds to roughly 0.75 MiB of raw data for 256 vectors of 768 `float32` values before Python/Qdrant object overhead.

No raw image bytes are retained or sent to Qdrant by this layer.

## Error model

The index package exposes focused errors:

```text
VisualIndexError
├── VisualIndexConfigurationError
└── IncompatibleVisualIndexError
```

`VisualIndexConfigurationError` covers invalid Winston-side index configuration, malformed index inputs, calling `upsert()` before compatibility has been established, or trying to upsert a batch with an embedding identity different from the active collection identity.

`IncompatibleVisualIndexError` covers an already-existing Qdrant collection whose vector configuration or Winston metadata does not match the requested embedding identity.

Compatibility errors should include useful expected/actual information, for example:

```text
Incompatible visual index collection 'winston_visual'.
Expected vector 'visual': size=768, distance=cosine.
Actual vector 'visual': size=512, distance=cosine.
Explicit reindexing is required.
```

Network and provider failures from Qdrant should retain their original exception as the cause when translated into a Winston index error.

## Qdrant cosine semantics

The collection contract remains `Cosine` because that is the semantic metric used by Phase 1C and the roadmap.

Qdrant documents that cosine vectors are automatically normalized during upload and similarity calculation is implemented as a dot product over normalized vectors. Winston therefore does not need to replace the collection contract with `Dot` merely to obtain dot-product execution internally.

Because Qdrant may return the normalized stored vector rather than the exact raw provider vector, Phase 1D integration tests verify identity, point count, vector configuration, and payload provenance rather than asserting byte-for-byte equality with the pre-upload embedding.

The persisted metric remains explicitly `cosine` in Winston collection metadata.

## Testing strategy

### Unit tests

Tests live under:

```text
tests/src/winston/index/
├── test_identity.py
├── test_models.py
└── test_qdrant.py
```

Required unit coverage includes:

- unchanged path/size/mtime under the same indexing root produces the same `asset_id`;
- changing path, size, or mtime changes `asset_id`;
- paths are stored relative to the indexing root with POSIX separators;
- a file outside the indexing root is rejected;
- the same visual candidate produces the same UUIDv5 point ID;
- changing timestamp changes the point ID;
- changing exact region geometry changes the point ID;
- changing model, dimension, or preprocessing version changes the point ID;
- `scale` does not change the point ID when exact pixel geometry is unchanged;
- timestamps use the specified deterministic microsecond conversion;
- photo, keyframe, full-frame, and tile payloads preserve complete provenance;
- invalid media/sample/timestamp combinations are rejected;
- vector dimension mismatches are rejected before Qdrant upsert;
- `upsert()` before `ensure_compatible()` is rejected;
- a mixed embedding-identity batch is rejected before any network write;
- collection creation uses the configured named vector, dimension, and Cosine metric;
- collection metadata is written on creation;
- missing or incompatible existing collection metadata is rejected;
- incompatible actual vector configuration is rejected even if metadata claims compatibility;
- the implementation never calls an automatic collection deletion path;
- upserts are split into batches no larger than the configured batch size;
- only one write batch is materialized at a time;
- `wait=True` is used;
- repeated logical candidates map to the same Qdrant point ID.

### Real Qdrant integration test

A temporary or dedicated GitHub Actions verification must run against an actual `qdrant/qdrant:v1.18.2` service.

The acceptance flow is:

```text
start Qdrant 1.18.2
      ↓
ensure_compatible(identity)
      ↓
verify collection:
  winston_visual
  named vector visual
  768 / Cosine
  Winston metadata present
      ↓
upsert one IndexedVisual
      ↓
upsert the exact same IndexedVisual again
      ↓
count == 1
      ↓
read point and verify complete payload provenance
```

A second integration case creates an incompatible collection first and verifies that Winston raises `IncompatibleVisualIndexError` without dropping or rewriting that collection.

Mocks alone are not sufficient acceptance evidence for issue #8 because collection metadata and Qdrant's real idempotent upsert semantics are part of the contract being implemented.

## Acceptance mapping to issue #8

Issue criterion: repeated indexing does not create duplicates.

- Satisfied by canonical UUIDv5 point IDs plus Qdrant upsert semantics and a real integration test proving final count remains one.

Issue criterion: exact source media/region/timestamp can be reconstructed.

- Satisfied by relative `source_path`, stable `asset_id`, media/sample kinds, exact timestamp fields, and complete region geometry in payload.

Issue criterion: incompatible model, dimension, or preprocessing state is rejected or requires explicit reindexing.

- Satisfied by strict collection vector + metadata validation and no automatic destructive migration.

Issue criterion: Qdrant configuration matches embedding dimension/metric.

- Satisfied by collection creation and validation against `EmbeddingIdentity.dimension` and the fixed Cosine metric.

Issue criterion: tests cover collection creation and idempotent upsert behavior.

- Satisfied by unit coverage plus a real Qdrant 1.18.2 integration workflow.

## Dependency and API boundaries

Phase 1D depends on the Phase 1C `EmbeddingIdentity` and embedding matrix shape but remains independent of the concrete Jina local/API engines.

The future indexing pipeline will assemble `IndexedVisual` values from ingestion, sampling, region, and embedding outputs. That orchestration is intentionally not added here.

The future search layer can later depend on a search-capable index contract without requiring callers to construct `qdrant_client` request models directly.

## Final decisions

The design deliberately fixes these V0 choices:

- SHA-256 asset identity from canonical relative path + file size + `mtime_ns`;
- versioned canonical asset identity material (`winston:asset:v1`);
- source paths relative to the indexing root;
- UUIDv5 Qdrant point IDs using `uuid.NAMESPACE_URL`;
- versioned canonical JSON as UUID name material (`winston:visual:v1`);
- deterministic integer microsecond timestamp identity using decimal `ROUND_HALF_UP`;
- exact pixel geometry, not region scale, as spatial point identity;
- named vector `visual`;
- 768 dimensions for Jina CLIP v1;
- Cosine collection metric;
- native Qdrant collection metadata for Winston index compatibility;
- strict refusal of unowned/incompatible existing collections;
- no automatic drop/recreate behavior;
- explicit compatibility establishment before upsert;
- no mixed embedding identities within an upsert session;
- `AsyncQdrantClient` reused for the index lifetime;
- sequential upsert batches, default size 256;
- `wait=True` for completed write semantics;
- real Qdrant 1.18.2 integration validation before the PR is considered complete.
