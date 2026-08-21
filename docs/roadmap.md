# Winston roadmap

This roadmap defines the implementation order for Winston and records how the two feature branches that already exist fit into the project.

The roadmap is intentionally staged so that retrieval quality can be measured before adding OCR, temporal reasoning, a frontend, or live ingestion.

## Project goal

Index large collections of recorded surveillance videos and photos, then search them in natural language without requiring the searched concepts to have been declared as object classes during indexing.

Target end-state examples:

```text
man wearing a green Nike cap
woman with a pink stroller
red car running a red light
blue car with license plate XXX-YYY-ZZZ
```

The first two are primarily visual semantic retrieval. The third requires temporal understanding. The fourth requires semantic retrieval plus exact/fuzzy OCR/ALPR.

---

## Phase 0 — Foundations already in progress

### 0A. Local media ingestion — `feat/media-ingest`

**Current status:** implemented on `feat/media-ingest`; draft PR #1 is open against `main` and is mergeable.

Scope already present on the branch:

- recursive media discovery;
- support for local MKV/JPG/JPEG discovery;
- `ffprobe` metadata extraction;
- typed media metadata models;
- data/ffprobe settings;
- `winston scan [path]` CLI command;
- tests for scanner, probe and CLI behavior.

### Acceptance criteria before the next core phase

- PR #1 reviewed and merged into `main`;
- scan/probe behavior remains deterministic;
- malformed/unreadable media produces explicit errors without corrupting the scan;
- source footage remains ignored by Git.

This workstream becomes the canonical beginning of `src/winston/ingest`.

### 0B. Public dataset acquisition — `feat/public-stream-recorder`

**Current status:** implemented on `feat/public-stream-recorder`; no PR is currently open for this branch.

Scope already present on the branch:

- one-shot public HLS discovery;
- `.m3u8` discovery/resolution;
- FFmpeg recording;
- hourly UTC-aligned MKV output;
- source and camera metadata;
- Playwright support for JavaScript-driven players;
- explicit exclusion of YouTube/YouTube Live;
- unit tests for the recorder components.

### Integration rule

This branch is **dataset tooling**, not Winston runtime code.

It should remain under:

```text
scripts/stream_recorder/
```

Its output becomes ordinary recorded-media input for Winston:

```text
public streams
    ↓
stream recorder
    ↓
hourly MKV files
    ↓
Winston scan/index
```

Before merge, open a dedicated PR to `main` and validate that its dependency choices do not leak Playwright/collection-only dependencies into Winston's runtime package.

---

## Phase 1 — V0 semantic retrieval baseline

**Objective:** prove that Winston can retrieve a concept that was never defined as a detector class.

Example success case:

```text
query: "woman with a pink stroller"
```

must return relevant video timestamps/images without a `stroller` class, YOLO classification, or pre-existing track.

### 1A. Keyframe sampling

Implement `src/winston/sampling` with a sampler abstraction.

V0 implementation:

```text
FrameSampler
└── KeyframeSampler
```

Requirements:

- use FFmpeg/ffprobe-derived decoder keyframes (`key_frame == 1`);
- preserve exact timestamps;
- batch/extract keyframes efficiently;
- do not assume keyframes are semantically important;
- expose the sampler behind an interface so later strategies can replace/augment it.

### Tests

- known fixture has deterministic sampled timestamps;
- no duplicate timestamps;
- keyframe metadata remains stable across repeated runs;
- unsupported/corrupt video fails clearly.

### 1B. Deterministic spatial regions

Add class-agnostic spatial candidate generation for each sampled image.

V0 candidates:

- full image;
- deterministic overlapping regions/crops.

Rules:

- no object detector required;
- no class labels required;
- no tracking required;
- region coordinates describe deterministic image geometry only.

### Tests

- expected region count for known dimensions;
- every region remains inside image bounds;
- deterministic region ids/order;
- adequate overlap/coverage at image borders.

### 1C. Jina CLIP v1 embeddings

Implement `src/winston/embeddings` around an explicit multimodal embedder interface.

Baseline:

```text
model: jinaai/jina-clip-v1
image vector dimension: 768
text vector dimension: 768
```

Requirements:

- official preprocessing/tokenization path;
- batched image embedding;
- text embedding for queries;
- model id + preprocessing version exposed programmatically;
- cosine-compatible output;
- deterministic behavior within inference tolerances.

### Tests

- output dimensions are 768;
- text and image encoders share the configured model identity;
- simple known image/text pairs rank semantically sensible matches above unrelated ones;
- model/preprocessing metadata is attached to index records.

### 1D. Qdrant visual index

Implement `src/winston/index` using Qdrant.

The existing Compose service remains the local V0 deployment.

Initial collection:

```text
winston_visual
```

Named vector:

```text
visual: 768-D, cosine
```

Each point represents one full sampled image or deterministic region.

Payload includes at minimum:

```text
asset_id
source_path
media_type
timestamp_seconds
sample_kind
region_kind
region coordinates
model_id
preprocessing_version
```

Requirements:

- deterministic point ids;
- idempotent reindex/upsert;
- no raw media bytes stored in Qdrant;
- explicit collection compatibility checks for model/dimension/version.

### Tests

- collection creation;
- upsert + retrieve;
- rerun does not duplicate candidates;
- incompatible vector dimension/model metadata is rejected or forces explicit reindexing.

### 1E. First semantic search CLI

Implement `src/winston/search` and expose:

```text
winston index <path>
winston search "query"
```

Search path:

```text
query
  ↓
Jina CLIP text embedding
  ↓
Qdrant cosine retrieval
  ↓
top visual candidates
  ↓
group by asset + temporal proximity
  ↓
rank user-facing hits
```

Result output should expose:

```text
source file
video timestamp (when applicable)
region
raw semantic score
```

Do not manufacture a confidence percentage in V0.

### Phase 1 exit criteria

On a fixed evaluation dataset Winston can retrieve unseen concepts such as:

```text
pink stroller
person wearing a green cap
person carrying a cardboard box
red bicycle
umbrella
```

with no corresponding detector/class ontology used during indexing.

The evaluation records Recall@K, Precision@K, indexing throughput, vector count/hour and query latency.

---

## Phase 2 — Coarse-to-fine temporal refinement

**Objective:** keep the global index sparse while improving recall and timestamp precision.

Problem:

A visual event can occur entirely between two video keyframes. Keyframe-only indexing can therefore miss short events.

Design:

```text
sparse keyframe index
    ↓
Qdrant candidate hit
    ↓
construct bounded time window
    ↓
decode intermediate frames densely
    ↓
local semantic reranking
    ↓
precise timestamp/range
```

### Work items

- temporal clustering/grouping of nearby semantic hits;
- candidate-window construction;
- bounded dense decoding around top candidates;
- configurable local sampling frequency;
- local embedding/reranking without globally persisting every decoded frame;
- benchmark recall improvement versus added query cost.

### Sampling experiments

Add and benchmark alternative implementations behind `FrameSampler`:

```text
PeriodicSampler
SceneChangeSampler
HybridSampler
```

Do not replace `KeyframeSampler` without evaluation evidence.

### Phase 2 exit criteria

- known short events missed by the sparse index are recovered by refinement at an acceptable rate;
- user-facing results return a useful video interval, not just an isolated sparse keyframe;
- query refinement latency is measured and bounded.

---

## Phase 3 — Exact text and license plates

**Objective:** support queries containing exact/fuzzy textual identity, especially license plates.

Example:

```text
blue car with license plate XXX-YYY-ZZZ
```

Architecture:

```text
semantic query / visual candidate retrieval
    ↓
candidate frames/windows
    ↓
plate/text region discovery
    ↓
OCR / ALPR
    ↓
exact/fuzzy text match
    ↓
combined ranking
```

### Work items

- benchmark OCR/ALPR options on surveillance-camera quality footage;
- normalize plate formats;
- preserve recognized text + confidence + timestamp;
- exact match and typo-tolerant/fuzzy match;
- structured storage outside the CLIP vector itself;
- query parsing/routing so an exact plate token is not treated as pure semantic similarity.

### Important rule

Do not require ALPR during Phase 1 indexing. It is specialized enrichment layered on top of broad semantic candidate retrieval unless measurements later justify an offline ALPR index.

### Phase 3 exit criteria

A query combining appearance and exact plate text returns the correct vehicle when the plate is visually readable in the source footage.

---

## Phase 4 — Temporal/action understanding

**Objective:** answer queries that cannot be proven by one image.

Example:

```text
red car running a red light
```

This requires reasoning over a sequence, not only retrieving frames containing both a red car and a red traffic light.

Architecture:

```text
semantic candidate retrieval
    ↓
candidate temporal windows
    ↓
dense frame/video representation
    ↓
temporal reasoning / verification
    ↓
rerank
```

### Research tracks

Benchmark at least these strategies before choosing one:

- video/multiframe embedding model;
- VLM over selected frames from a candidate window;
- local motion/tracking plus explicit event logic where appropriate;
- hybrid approach: semantic retrieval -> local tracking/temporal verifier.

Tracking is allowed here as a **local verification tool**. It must not become the prerequisite that determines which concepts Winston can search globally.

### Phase 4 exit criteria

The system distinguishes:

```text
red car next to red light
```

from:

```text
red car actually crossing against the red light
```

on a controlled evaluation set.

---

## Phase 5 — VLM descriptions and hybrid retrieval

**Objective:** improve abstract/action/context retrieval with generated descriptions while keeping direct visual search.

Potential path:

```text
candidate image/window
    ↓
VLM description
    ↓
text embedding/index
```

Search can then fuse:

```text
text -> visual embeddings
text -> generated-description embeddings
structured OCR evidence
temporal-verifier evidence
```

### Work items

- choose description granularity: frame, region or temporal window;
- measure cost and retrieval gain;
- keep visual and description scores separately calibrated;
- define score fusion/reranking strategy;
- store source/provenance for every generated description.

This phase is optional if direct visual + temporal/OCR retrieval already meets product requirements.

---

## Phase 6 — API and minimal UI

**Objective:** expose the validated engine without coupling frontend work to early ML experimentation.

Only begin this phase once Phase 1/2 search quality is demonstrably useful.

Potential backend endpoints:

```text
POST /assets
POST /index
GET  /index/{job_id}
POST /search
GET  /media/{asset_id}
```

Minimal user flow:

```text
select/upload indexed media
        ↓
natural-language search
        ↓
results with thumbnails
        ↓
open video at matching timestamp/range
```

Frontend technology is deliberately not fixed in the core roadmap. A small Python UI or lightweight React frontend can be selected after the backend contract stabilizes.

---

## Phase 7 — Continuous/live ingestion

**Objective:** extend the proven offline engine to ongoing camera sources.

This is separate from `feat/public-stream-recorder`, which is one-shot dataset collection tooling.

Future live ingestion may support RTSP/HLS sources and continuously generate indexable media chunks.

Requirements before implementation:

- defined chunking/retention model;
- backpressure strategy;
- indexing lag target;
- recovery after process restart;
- storage lifecycle policy;
- camera/source metadata model.

Do not introduce this complexity before the offline indexing/search pipeline is measured and stable.

---

# Planned package evolution

## V0 / Phase 1

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

This is a design target, not a requirement to create placeholder files before implementation.

## Later phases

Only add modules when they have concrete implementations, for example:

```text
src/winston/
├── ocr/          # Phase 3
├── temporal/     # Phase 4
└── descriptions/ # Phase 5, if adopted
```

Avoid empty speculative package trees.

---

# Dependency/integration order

Recommended merge/development sequence from the repository's current state:

```text
main
 │
 ├─ review + merge PR #1 / feat/media-ingest
 │
 ├─ open/review/merge feat/public-stream-recorder
 │    (independent dataset-tooling workstream)
 │
 ├─ feat/keyframe-sampling
 │
 ├─ feat/spatial-regions
 │
 ├─ feat/jina-clip-v1
 │
 ├─ feat/qdrant-visual-index
 │
 ├─ feat/semantic-search
 │
 ├─ feat/search-evaluation
 │
 ├─ feat/temporal-refinement
 │
 ├─ feat/alpr-ocr
 │
 ├─ feat/temporal-actions
 │
 └─ API/UI/live-ingest only after core retrieval quality is proven
```

`feat/public-stream-recorder` does not block semantic indexing; it is useful for obtaining more realistic footage but is not a runtime prerequisite.

---

# Milestones

## M0 — Media understood

- local files can be discovered and probed;
- realistic recorded datasets can be acquired independently;
- no media is committed to Git.

## M1 — Search unseen visual concepts

- recorded videos and photos are indexed;
- keyframes + deterministic regions are embedded with Jina CLIP v1;
- Qdrant returns semantic candidates;
- natural-language CLI search works;
- `pink stroller`-style queries do not depend on predefined classes.

## M2 — Reliable video localization

- candidate hits are grouped into temporal windows;
- dense local refinement improves short-event recall and timestamp precision.

## M3 — Exact visual text

- OCR/ALPR adds exact/fuzzy license-plate matching and similar textual evidence.

## M4 — Actions

- candidate windows can be verified/reranked using temporal reasoning;
- action queries such as running a red light become distinct from static co-occurrence.

## M5 — Product surface

- stable API;
- minimal search UI;
- later optional continuous camera ingestion.

---

# Definition of success

Winston is successful when a future query does not need to have been anticipated at indexing time.

The baseline semantic architecture must therefore remain capable of searching arbitrary visual concepts, while specialized components improve exact identifiers and temporal actions without narrowing the global index to a predefined ontology.
