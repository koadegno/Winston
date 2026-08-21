# Winston architecture

## 1. Purpose

Winston is a semantic search engine for large collections of recorded surveillance video and photos.

The system must let a user search for things that were **not declared as classes at indexing time**, for example:

```text
man wearing a green Nike cap
woman with a pink stroller
red car running a red light
blue car with license plate XXX-YYY-ZZZ
```

Those examples span three different problems:

- visual semantic retrieval;
- exact text recognition such as license plates;
- temporal/action understanding.

Winston therefore uses a broad class-agnostic visual index as its first-stage retriever, then adds specialized verification/enrichment only where needed.

---

## 2. Architectural invariants

These rules define the project unless a later design decision explicitly replaces them.

1. **No predefined object classes for the baseline index.**
   YOLO classes, tracked objects, labels, or a fixed ontology must not decide what can be searched later.
2. **Recorded files first.**
   V0 indexes existing video/image files. Continuous RTSP/HLS ingestion comes later.
3. **Coarse-to-fine retrieval.**
   Keep the global index sparse, retrieve candidate locations cheaply, then spend extra compute only around the best candidates.
4. **Raw media stays outside Qdrant.**
   Qdrant stores vectors and searchable metadata pointing back to source media.
5. **Sampling is replaceable.**
   Keyframes are the initial strategy, not a permanent assumption.
6. **A region is geometry, not a detection.**
   Coordinates describe a deterministic crop/tile and do not imply that an object detector found anything there.
7. **Embedding compatibility is explicit.**
   Model identity, vector dimension, and preprocessing version are part of the index contract.
8. **Exact identifiers and actions are specialized stages.**
   CLIP-style similarity should not be forced to solve exact license plates or multi-frame actions alone.

---

## 3. Existing repository workstreams

### `feat/media-ingest`

At the time of writing, this branch has draft PR #1 open against `main` and is mergeable.

It already implements the beginning of Winston's core ingestion layer:

- recursive local media discovery;
- typed media metadata;
- `ffprobe` probing;
- Winston settings for data directory and `ffprobe`;
- `winston scan [path]`;
- tests for scanning, probing, and CLI output.

This branch is a **runtime foundation** and should be merged before the indexing pipeline is built on top of it.

### `feat/public-stream-recorder`

At the time of writing, this branch has PR #2 open against `main` and is mergeable.

It implements one-shot dataset acquisition tooling under:

```text
scripts/stream_recorder/
```

It:

- discovers public HLS webcam streams;
- resolves `.m3u8` streams, including JavaScript-driven players;
- records with FFmpeg stream copy;
- writes UTC hour-aligned MKV files;
- persists source/camera metadata;
- excludes YouTube and YouTube Live;
- includes unit/local integration tests and a live-webcam smoke test.

This workstream is **not Winston runtime code**. It produces realistic recorded media that the normal Winston ingestion/indexing pipeline can consume.

The two branches are therefore complementary:

```text
feat/public-stream-recorder
        ↓
recorded dataset
        ↓
feat/media-ingest
        ↓
Winston indexing/search
```

But the recorder does not block semantic-index development; Winston must also work with arbitrary local media not produced by that script.

---

## 4. High-level system

```text
                          MEDIA
                 ┌─────────┴─────────┐
                 │                   │
               VIDEO               IMAGE
                 │                   │
                 └─────────┬─────────┘
                           ▼
                         INGEST
                  discover + probe assets
                           │
                           ▼
                        SAMPLING
                  video keyframes / photo
                           │
                           ▼
                        REGIONS
                 full frame + fixed crops
                           │
                           ▼
                       EMBEDDINGS
                Jina CLIP v1 image encoder
                           │
                           ▼
                         QDRANT
                 vectors + media metadata
                           ▲
                           │
                Jina CLIP v1 text encoder
                           ▲
                           │
                         QUERY
                           │
                           ▼
                         SEARCH
                retrieve + group + rerank
                           │
                           ▼
                   CANDIDATE WINDOWS
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          dense local    OCR/ALPR     temporal
          refinement                  reasoning
```

---

## 5. Package boundaries

The repository already reserves the V0 package shape:

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

Target V0 implementation:

```text
src/winston/
├── ingest/
│   ├── models.py
│   ├── probe.py
│   └── scanner.py
├── sampling/
│   ├── base.py
│   ├── keyframes.py
│   └── regions.py
├── embeddings/
│   ├── base.py
│   └── jina_clip.py
├── index/
│   └── qdrant.py
├── search/
│   ├── engine.py
│   ├── grouping.py
│   └── models.py
├── utils/
├── config.py
├── cli.py
└── __init__.py
```

This is a design target, not a request to create empty placeholder files in advance.

Future modules should appear only when implemented, for example:

```text
ocr/          # exact text / ALPR
 temporal/    # action understanding
 descriptions/# optional VLM-generated descriptions
```

---

## 6. Ingestion

Responsibility: discover media and extract deterministic metadata before indexing.

The `feat/media-ingest` branch already establishes this layer.

A source asset needs at least:

```text
asset_id
path
media_type          # video | image
width
height
codec               # video
fps                  # video
duration_seconds    # video
```

The initial source of truth can be the filesystem plus a small local manifest/catalog. Qdrant is not the authoritative media catalog.

Asset identity must be stable enough to support idempotent reindexing. The exact strategy can be path + file metadata initially, and later a stronger content identity if needed.

---

## 7. Temporal sampling

### V0: decoder keyframes

Winston does not embed every frame of every video.

The first sampler uses decoder keyframes discovered through FFmpeg/ffprobe. The implementation should rely on the decoder-level `key_frame == 1` signal rather than equating every useful random-access point with `pict_type == I`.

Why keyframes are useful:

- far fewer frames need to be decoded;
- they are natural random-access points in compressed video;
- surveillance footage commonly has GOPs that reduce the sample count by tens of times compared with indexing every frame.

Important limitation:

**keyframes are compression decisions, not semantic decisions.**

A short event can occur entirely between two keyframes. Therefore the code must hide sampling behind an abstraction:

```text
FrameSampler
├── KeyframeSampler      # V0 baseline
├── PeriodicSampler      # future experiment
├── SceneChangeSampler   # future experiment
└── HybridSampler        # future experiment
```

Every sampled video frame keeps its exact source timestamp.

A photo is already a single temporal sample.

---

## 8. Spatial regions

The V0 index remains class-agnostic.

Each sampled image produces deterministic visual candidates such as:

```text
full image
fixed overlapping tiles/crops
possibly multiple scales
```

No detector determines which regions exist.

For each region store geometry such as:

```text
region_kind = full | tile
x
y
width
height
scale
```

These coordinates only tell Winston which pixels produced an embedding and later allow the result viewer/refiner to return to the matching part of the image.

---

## 9. Multimodal embedding

### Baseline: Jina CLIP v1

V0 uses the Jina CLIP v1 shared text/image embedding space.

Image side:

```text
sampled image / region
        ↓
Jina CLIP v1 image encoder
        ↓
768-D vector
```

Query side:

```text
natural-language text
        ↓
Jina CLIP v1 text encoder
        ↓
768-D vector
```

The two encoders belong to the same CLIP-style model and are trained so their vectors are directly comparable.

Baseline metric:

```text
cosine
```

The implementation should expose a model-independent interface:

```text
MultimodalEmbedder
├── embed_images(images)
├── embed_text(texts)
├── dimension
├── model_id
└── preprocessing_version
```

Changing model/preprocessing invalidates compatible index assumptions and must trigger explicit reindexing.

---

## 10. Qdrant index

Qdrant is Winston's semantic retrieval database.

`main` already starts Qdrant through `compose.yml` and persists it under:

```text
./volumes/qdrant
```

### V0 collection

Start with one visual collection:

```text
winston_visual
```

Vector configuration:

```text
name: visual
size: 768
metric: cosine
```

One Qdrant point represents one indexed visual candidate: either a full sampled image or one deterministic region.

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

For photos, timestamp is omitted/null.

Point IDs must be deterministic so repeated indexing is idempotent. They should derive from stable asset identity + temporal sample identity + region identity + model/preprocessing version.

### Why Qdrant

Frigate can use brute-force `sqlite-vec` because it heavily reduces the index to tracked-object thumbnails. Winston deliberately indexes content without requiring tracked objects, so vector counts can be much larger. Qdrant/HNSW plus payload filtering is a better match for the expected scale.

---

## 11. V0 indexing pipeline

```text
asset discovery
    ↓
media probe
    ↓
keyframe extraction (video) / photo
    ↓
deterministic regions
    ↓
batched Jina CLIP image embedding
    ↓
batched Qdrant upsert
```

Properties:

- restartable;
- idempotent;
- raw media never inserted into Qdrant;
- explicit failures per asset/sample;
- model/preprocessing metadata stored with points.

Index progress can eventually expose states such as:

```text
discovered
probed
sampled
embedded
indexed
failed
```

A distributed worker system is not required for V0.

---

## 12. V0 search pipeline

Example:

```text
woman with a pink stroller
```

Path:

```text
query
  ↓
Jina CLIP v1 text encoder
  ↓
query vector
  ↓
Qdrant cosine retrieval
  ↓
top visual candidates
  ↓
group nearby candidates by asset/time
  ↓
rank user-facing hits
  ↓
source + timestamp + matching region
```

Grouping is necessary so adjacent keyframes/tiles from one event do not flood the result list as separate user-facing hits.

V0 should return raw semantic scores. Do not invent a human confidence percentage before calibration on an evaluation set.

---

## 13. Coarse-to-fine refinement

The global sparse index is a candidate generator, not necessarily the final timestamp detector.

For a hit around time `t`:

```text
Qdrant hit at t
    ↓
create bounded time window around t
    ↓
decode intermediate frames
    ↓
sample more densely
    ↓
local embedding/reranking
    ↓
precise timestamp/range
```

This is how Winston can keep indexing cost low while recovering short events that keyframe-only indexing can miss.

Dense refinement should initially happen at query time and should not globally persist every intermediate frame unless measurements justify it.

---

## 14. Exact text and ALPR

Example:

```text
blue car with license plate XXX-YYY-ZZZ
```

This contains two signals:

```text
blue car        -> semantic visual retrieval
XXX-YYY-ZZZ     -> exact/fuzzy text identity
```

Planned path:

```text
semantic candidate retrieval
    ↓
candidate frames/windows
    ↓
plate/text localization
    ↓
OCR / ALPR
    ↓
normalized exact/fuzzy match
    ↓
combined ranking
```

Do not rely on CLIP similarity to reproduce exact characters reliably.

Whether ALPR later becomes an offline enrichment index or remains mostly candidate-time refinement should be decided from performance measurements.

---

## 15. Temporal/action understanding

Example:

```text
red car running a red light
```

A single image containing a red car and a red traffic light does not prove the action.

Planned path:

```text
visual semantic candidates
    ↓
candidate temporal windows
    ↓
dense multi-frame representation
    ↓
temporal verification/reasoning
    ↓
rerank
```

Candidate-local tracking is allowed as one possible verification tool. It must not become the global prerequisite that determines which concepts are indexable.

Possible research tracks include:

- video/multiframe embedding models;
- VLM reasoning over selected frames;
- local motion/tracking + explicit event logic;
- combinations of the above.

---

## 16. Optional generated descriptions

A future VLM may generate descriptions for frames/regions/windows.

Those descriptions can be embedded separately and fused with direct visual retrieval:

```text
query -> visual embedding search
query -> description embedding search
      -> score fusion/reranking
```

This remains optional enrichment. Winston's core visual search must work without descriptions.

---

## 17. Interfaces

V0 is CLI-first:

```text
winston scan <path>
winston index <path>
winston search "woman with a pink stroller"
```

Only after search quality is demonstrated should a stable HTTP API be added.

Potential later API:

```text
POST /assets
POST /index
GET  /index/{job_id}
POST /search
GET  /media/{asset_id}
```

A frontend comes after the backend/search contract is useful and stable.

---

## 18. Deployment shape

V0:

```text
host
├── Winston Python CLI/process
├── FFmpeg + ffprobe
├── local recorded-media dataset
└── Docker Compose
    └── Qdrant
```

No Kubernetes, auth system, distributed queue, frontend, or live-camera service is necessary for the first algorithmic milestone.

---

## 19. Evaluation

A fixed evaluation dataset must be created before optimizing architecture choices.

Initial query families should intentionally include concepts not represented by any predefined class list:

```text
pink stroller
person wearing a green cap
person carrying a cardboard box
red bicycle
umbrella
```

Later suites add:

```text
license plates   # OCR/ALPR
multi-frame actions
```

Track at minimum:

- Recall@K;
- Precision@K;
- indexing throughput;
- vectors per hour of video;
- Qdrant query latency;
- refinement latency.

Any speed optimization that reduces recall must be visible in these metrics.

---

## 20. Definition of the core design

The core Winston architecture is:

```text
recorded media
    ↓
class-agnostic sparse visual sampling
    ↓
Jina CLIP v1 image embeddings
    ↓
Qdrant
    ↑
Jina CLIP v1 text query
    ↓
semantic candidate windows
    ↓
specialized refinement only where necessary
```

The essential property is that a future search concept does not need to have been anticipated when the media was indexed.
