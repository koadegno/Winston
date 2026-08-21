# Winston architecture

## Purpose

Winston is a semantic search engine for recorded surveillance footage and photos.

The core requirement is that indexing must **not depend on knowing in advance what a future user will search for**. A user should be able to search for visual concepts that were never declared as detector classes during ingestion, for example:

- `man wearing a green Nike cap`
- `woman with a pink stroller`
- `red car running a red light`
- `blue car with license plate XXX-YYY-ZZZ`

These examples do not all require the same retrieval technique. Winston therefore separates coarse semantic retrieval from later specialized enrichment and temporal reasoning.

## Non-negotiable design principles

1. **No predefined object classes in the baseline semantic index.**
   Winston must not require YOLO classes, tracked objects, labels, or a classifier-defined ontology before an image can be indexed.
2. **Offline files first.**
   The first product surface is a dataset of existing video and image files. Live camera ingestion is a later concern.
3. **Coarse-to-fine search.**
   Build a cheap broad index over sparse video samples, retrieve candidate regions/windows, then spend more compute only around promising results.
4. **Raw media is not stored in Qdrant.**
   Videos and images remain on filesystem/object storage. Qdrant stores vectors and searchable metadata that point back to the source media.
5. **Model/index compatibility is explicit.**
   Every indexed vector must be traceable to the embedding model and preprocessing version that produced it. Changing either requires reindexing.
6. **Exact text and temporal actions are separate problems from visual similarity.**
   CLIP-style retrieval is the first-stage candidate generator; OCR/ALPR and temporal reasoning are added as dedicated later stages.

## Current repository structure

The repository already reserves the following package boundaries:

```text
src/winston/
├── ingest/
├── sampling/
├── embeddings/
├── index/
├── search/
├── utils/
├── cli.py
└── __init__.py
```

These boundaries remain the V0 architecture.

Future capabilities such as OCR/ALPR and temporal reasoning should only gain dedicated modules when their roadmap phase starts. They should not be mixed into the baseline semantic retrieval code.

## Existing workstreams

Two feature branches already exist and are part of this design.

### `feat/media-ingest`

Status: active branch, draft PR #1 open against `main`.

This branch implements the beginning of Winston's runtime ingestion layer:

- recursive discovery of supported local media;
- media metadata models;
- `ffprobe`-based video/image probing;
- Winston settings for the data directory and `ffprobe` binary;
- a `winston scan` CLI command;
- tests for scanning, probing, and CLI output.

This is the foundation for the indexing pipeline and should be merged before the semantic indexing phases are built on top of it.

### `feat/public-stream-recorder`

Status: active branch, not yet represented by a PR at the time this document was written.

This branch implements **one-shot dataset acquisition tooling**, not Winston's core runtime. It lives under `scripts/stream_recorder/` and:

- discovers public HLS webcam streams;
- resolves `.m3u8` candidates;
- records streams with FFmpeg;
- writes hourly MKV files plus source/camera metadata;
- intentionally excludes YouTube and YouTube Live.

This workstream feeds recorded media into Winston's offline dataset. It should stay under `scripts/` and must not become a dependency of the core semantic-search package.

## High-level system

```text
                         RECORDED MEDIA
                ┌──────────────┴──────────────┐
                │                             │
              VIDEO                         IMAGE
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
                            INGEST
                    discover + probe media
                               │
                               ▼
                           SAMPLING
              video keyframes / image candidate
                               │
                               ▼
                     SPATIAL REGIONS/CROPS
                  full image + fixed regions
                               │
                               ▼
                         EMBEDDINGS
                   Jina CLIP v1 image encoder
                               │
                               ▼
                            QDRANT
                 visual vectors + media payload
                               ▲
                               │
                   Jina CLIP v1 text encoder
                               ▲
                               │
                           USER QUERY
                               │
                               ▼
                            SEARCH
               retrieve + group + rank candidates
                               │
                               ▼
                      CANDIDATE WINDOWS
                               │
                    later refinement stages
               ┌───────────────┼────────────────┐
               ▼               ▼                ▼
            dense          OCR / ALPR         temporal
          re-decode                          reasoning
```

## 1. Ingestion

Responsibility: discover source assets and describe them deterministically before indexing.

The `feat/media-ingest` branch is the first implementation of this layer.

A media asset should expose at least:

```text
asset_id
path
media_type          # video | image
width
height
codec               # video only
duration_seconds    # video only
fps                  # video only
```

The initial source of truth can remain the filesystem plus a small local catalog/manifest. Qdrant is not the canonical media catalog.

## 2. Temporal sampling

### Baseline: video keyframes

For V0, recorded video is sampled using decoder keyframes rather than every decoded frame.

The important concept for Winston is the decoder's `key_frame == 1` signal, not merely `pict_type == I`. H.264/H.265 concepts such as I-frames, IDR frames and other random-access pictures are related but not identical. The implementation should therefore expose a generic `KeyframeSampler` abstraction.

Why use keyframes:

- they drastically reduce the number of frames that must be decoded and embedded;
- they are natural random-access points in compressed video;
- typical surveillance GOPs can reduce the sampled frame count by tens of times compared with indexing every frame.

Important limitation:

**keyframes are codec decisions, not semantic decisions.** A short visual event can occur entirely between two keyframes. Therefore keyframe-only sampling is the V0 baseline, not a permanent assumption.

The sampling interface should make future alternatives possible without changing embedding/index/search code:

```text
FrameSampler
├── KeyframeSampler      # V0
├── PeriodicSampler      # future
├── SceneChangeSampler   # future
└── HybridSampler        # future
```

Each sampled frame must keep its exact source timestamp.

### Photos

A photo is already a single temporal sample and enters directly into spatial-region generation.

## 3. Spatial candidate generation

Winston must remain class-agnostic at indexing time.

The V0 spatial stage therefore creates deterministic image regions without asking an object detector what is present. At minimum it should index:

- the full sampled image;
- a fixed set of overlapping regions/crops at one or more scales.

A stored `region` is simply geometry describing which pixels were embedded. It is **not** a detection bounding box and does not imply any object class.

Example metadata:

```text
region_kind = full | tile
x
y
width
height
scale
```

This preserves Winston's ability to retrieve an arbitrary future concept while still allowing small details to contribute strongly to an embedding.

## 4. Visual embedding

### Baseline model: Jina CLIP v1

V0 uses the Jina CLIP v1 multimodal embedding space:

```text
image / region
    ↓
Jina CLIP v1 image encoder
    ↓
768-dimensional visual embedding
```

At query time:

```text
natural-language query
    ↓
Jina CLIP v1 text encoder
    ↓
768-dimensional text embedding
```

Both encoders are two halves of the same CLIP-style model and produce vectors intended to be compared in the same embedding space.

The baseline similarity metric is cosine similarity/distance.

The embedding implementation must hide model details behind an interface so a later model can be benchmarked without rewriting sampling or search.

Suggested contract:

```text
MultimodalEmbedder
├── embed_images(images) -> vectors
├── embed_text(texts) -> vectors
├── dimension
├── model_id
└── preprocessing_version
```

## 5. Qdrant semantic index

Qdrant is Winston's semantic retrieval database.

The repository already starts Qdrant through `compose.yml` with persistent storage under `./volumes/qdrant`.

### Initial collection

Start with one collection dedicated to visual candidates, for example:

```text
winston_visual
```

Initial vector configuration:

```text
name: visual
size: 768
metric: cosine
```

A Qdrant point represents **one indexed visual candidate**: either a full sampled frame/image or one deterministic region of it.

Suggested payload:

```json
{
  "asset_id": "...",
  "source_path": "...",
  "media_type": "video",
  "timestamp_seconds": 123.456,
  "sample_kind": "keyframe",
  "region_kind": "tile",
  "region": {
    "x": 960,
    "y": 540,
    "width": 960,
    "height": 540
  },
  "model_id": "jinaai/jina-clip-v1",
  "preprocessing_version": 1
}
```

For photos, `timestamp_seconds` can be null/omitted.

The Qdrant point id must be deterministic so rerunning an index job is idempotent. It should derive from stable asset identity + sample timestamp + region identity + model/preprocessing version.

### Why Qdrant instead of Frigate's sqlite-vec design

Frigate can keep its vector count low because it indexes one representative thumbnail per tracked object. Winston intentionally does not require tracked objects, so the number of vectors can become much larger. Qdrant/HNSW is therefore a better fit for the expected scale and supports metadata filtering as part of retrieval.

## 6. Indexing pipeline

The first complete indexing path should be:

```text
asset discovery
    ↓
media probe
    ↓
keyframe extraction (video) / image input
    ↓
spatial regions
    ↓
batched Jina CLIP image embedding
    ↓
batched Qdrant upsert
```

The pipeline must be restartable and idempotent. Re-running the same indexing job must not create duplicate semantic candidates.

Index progress should eventually distinguish at least:

```text
discovered
probed
sampled
embedded
indexed
failed
```

V0 does not need a distributed worker system; correctness and measurable retrieval quality come first.

## 7. Search pipeline

### V0 visual semantic retrieval

For a query such as:

```text
woman with a pink stroller
```

Winston performs:

```text
query
  ↓
Jina CLIP v1 text encoder
  ↓
query vector
  ↓
Qdrant cosine search
  ↓
top visual candidates
  ↓
group nearby results by asset + time
  ↓
rank candidate frames/windows
  ↓
source path + timestamp + matching region
```

The grouping step matters because adjacent keyframes/regions from the same event should not fill the entire result list as independent user-facing hits.

The baseline ranking should preserve raw retrieval scores. UI-friendly percentages should not be invented until they are calibrated against a real evaluation set.

### Coarse-to-fine refinement

Once V0 retrieval works, Winston should use the sparse index only to locate promising temporal windows.

For a candidate around timestamp `t`:

```text
Qdrant hit at t
    ↓
decode a bounded window around t
    ↓
sample more densely
    ↓
re-embed / rerank locally
    ↓
return a more precise timestamp/range
```

This allows the global index to stay economical while avoiding the permanent recall limitations of keyframe-only sampling.

## 8. Exact text: OCR / ALPR

A query such as:

```text
blue car with license plate XXX-YYY-ZZZ
```

contains two different retrieval signals:

- semantic visual concepts: `blue car`;
- exact/fuzzy text identity: `XXX-YYY-ZZZ`.

The visual index should first produce candidate frames/windows. A later ALPR/OCR stage can then extract plate text from candidate media and support exact/fuzzy matching.

Do not force exact identifiers into the CLIP similarity score.

The search layer will eventually combine visual score + structured exact/fuzzy evidence.

## 9. Temporal actions

A query such as:

```text
red car running a red light
```

cannot be proven by one isolated image. It requires a temporal model or deterministic temporal analysis over several frames.

The intended architecture is:

```text
semantic visual retrieval
    ↓
candidate temporal windows
    ↓
dense decode / track local motion if useful
    ↓
temporal action reasoning
    ↓
rerank / verify
```

Tracking can be introduced **inside a candidate window as a refinement technique** without becoming a prerequisite for the global semantic index. This preserves the class-agnostic design.

## 10. Optional generated descriptions

A future VLM can generate textual descriptions of candidate windows. Their text embeddings may be indexed separately and fused with visual retrieval.

This is optional enrichment, not required for Winston's core `text -> image` semantic search.

If introduced, visual and description retrieval scores must be calibrated/fused deliberately rather than assumed to be directly comparable.

## 11. CLI and API boundaries

The first product interface is CLI-oriented:

```text
winston scan <path>
winston index <path>
winston search "woman with a pink stroller"
```

A web/API layer comes after the algorithm has measurable quality on a fixed dataset.

Later, an API can expose asset ingestion, indexing status, semantic search and result media endpoints without moving ML/indexing responsibilities into the web layer.

## 12. Dataset acquisition

`feat/public-stream-recorder` remains a separate one-shot acquisition utility:

```text
public HLS sources
    ↓
scripts/stream_recorder
    ↓
hourly MKV dataset
    ↓
Winston offline ingestion/indexing
```

It is useful for building realistic evaluation/training datasets, but Winston's core runtime must also work with arbitrary local files that were not produced by this script.

## 13. Evaluation strategy

Winston needs a reproducible test/evaluation dataset before optimization decisions are trusted.

The evaluation set should contain concepts intentionally absent from any predefined class list, including small and compositional targets. Example query families:

```text
pink stroller
person wearing a green cap
person carrying a cardboard box
red bicycle
blue car
person with an umbrella
```

Later evaluation families add:

```text
exact license plates     # OCR/ALPR phase
temporal actions         # temporal phase
```

Track at minimum:

- Recall@K for known relevant media;
- Precision@K;
- indexing throughput (minutes of footage indexed per wall-clock minute);
- total vector count per hour of video;
- Qdrant search latency;
- refinement latency.

Performance optimization is only accepted when retrieval quality is measured alongside throughput.

## 14. Initial deployment shape

For V0:

```text
host machine
├── Winston Python process / CLI
├── FFmpeg + ffprobe
├── local dataset filesystem
└── Docker Compose
    └── Qdrant
```

No Kubernetes, distributed queue, web frontend, authentication or live-camera service is required for the first algorithmic milestone.

## 15. Architectural invariants to protect

Future PRs should preserve these rules unless an explicit design decision replaces them:

- baseline indexing does not require object classification;
- baseline indexing does not require tracking;
- a `region` is deterministic image geometry, not a detector-defined bounding box;
- video sampling is replaceable behind a sampler abstraction;
- raw media remains outside Qdrant;
- query and image vectors must come from compatible multimodal embedding spaces;
- model/preprocessing changes require reindexing;
- exact text and temporal actions are specialized stages layered on top of semantic candidate retrieval;
- one-shot dataset acquisition scripts stay outside `src/winston` runtime code.
