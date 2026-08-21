# Winston roadmap

This roadmap defines the implementation order for Winston and incorporates the work already present on the repository's two active feature branches.

The guiding rule is simple: **prove useful class-agnostic semantic retrieval first, then add more expensive/specialized capabilities only when the baseline is measurable.**

## End-state goal

Winston should eventually support queries such as:

```text
man wearing a green Nike cap
woman with a pink stroller
red car running a red light
blue car with license plate XXX-YYY-ZZZ
```

The first two are mainly visual semantic retrieval. The third requires temporal understanding. The fourth requires visual retrieval plus OCR/ALPR.

---

# Phase 0 — Foundations already in progress

## 0A. Local media ingestion — `feat/media-ingest`

**Current state:** draft PR #1 is open against `main` and mergeable.

Already implemented:

- recursive local media discovery;
- MKV/JPG/JPEG support;
- typed media metadata;
- `ffprobe` metadata extraction;
- Winston data/ffprobe settings;
- `winston scan [path]`;
- tests for scanner, probe, and CLI behavior.

This branch becomes the canonical beginning of `src/winston/ingest`.

### Exit criteria

- PR #1 reviewed and merged;
- malformed/unreadable media fails explicitly;
- repeated scans are deterministic;
- media files and runtime volumes remain outside Git.

## 0B. Public dataset acquisition — `feat/public-stream-recorder`

**Current state:** PR #2 is open against `main` and mergeable.

Already implemented:

- one-shot public HLS stream discovery;
- `.m3u8` resolution;
- Playwright observation for JavaScript-driven players;
- FFmpeg stream-copy recording;
- UTC hour-aligned MKV files;
- source/camera metadata;
- recovery/re-discovery after expired stream URLs;
- exclusion of YouTube/YouTube Live;
- unit/local integration tests plus live-webcam smoke testing.

### Integration rule

This branch is dataset tooling and stays under:

```text
scripts/stream_recorder/
```

It must not become a runtime dependency of `src/winston`.

Its relationship to the main pipeline is:

```text
public HLS cameras
      ↓
stream recorder
      ↓
recorded MKV dataset
      ↓
Winston scan/index
```

It is useful for generating realistic test data but does not block core semantic-index work.

---

# Phase 1 — V0 class-agnostic semantic retrieval

**Objective:** demonstrate that Winston can retrieve a visual concept that was never defined as a detector class.

Reference success query:

```text
woman with a pink stroller
```

No `stroller` class, YOLO detector, object label, or pre-existing track may be required.

## 1A. Keyframe sampling

Implement the first sampler in `src/winston/sampling`.

```text
FrameSampler
└── KeyframeSampler
```

Requirements:

- obtain decoder keyframes via FFmpeg/ffprobe;
- use `key_frame == 1` semantics rather than assuming every useful random-access frame is simply `pict_type == I`;
- preserve exact source timestamps;
- extract/decode samples efficiently;
- expose a sampler abstraction for later alternatives.

### Tests

- deterministic timestamps on fixture video;
- no duplicate samples;
- corrupt/unsupported video fails clearly;
- timestamps point back to valid source locations.

## 1B. Deterministic spatial regions

For every sampled image, generate class-agnostic visual candidates:

```text
full image
fixed overlapping regions/crops
```

Rules:

- no object detector;
- no object class;
- no tracking;
- region coordinates are deterministic image geometry only.

### Tests

- deterministic region count/order;
- full image coverage;
- region bounds always valid;
- border areas covered with the expected overlap.

## 1C. Jina CLIP v1 embeddings

Implement the multimodal embedding layer.

Baseline:

```text
model: jinaai/jina-clip-v1
text dimension: 768
image dimension: 768
metric: cosine
```

Requirements:

- official Jina preprocessing/tokenization;
- batched image inference;
- query text inference;
- explicit `model_id`, dimension, and preprocessing version;
- model-independent interface so alternatives can later be benchmarked.

### Tests

- correct vector dimensions;
- simple semantically related text/image fixtures rank above unrelated fixtures;
- index metadata records model/preprocessing identity.

## 1D. Qdrant visual index

Implement `src/winston/index` against the Qdrant service already defined in `compose.yml`.

Initial collection:

```text
winston_visual
```

Named vector:

```text
visual
size = 768
metric = cosine
```

Each Qdrant point represents one full image/keyframe or one deterministic region.

Payload includes at least:

```text
asset_id
source_path
media_type
timestamp_seconds
sample_kind
region_kind
region geometry
model_id
preprocessing_version
```

Requirements:

- deterministic point IDs;
- idempotent upserts;
- raw media remains outside Qdrant;
- incompatible model/dimension/preprocessing state is rejected or forces explicit reindexing.

## 1E. Index CLI

Expose:

```text
winston index <path>
```

Pipeline:

```text
discover
  ↓
probe
  ↓
sample keyframes / photo
  ↓
generate regions
  ↓
batch image embeddings
  ↓
Qdrant upsert
```

Indexing must be restartable and idempotent.

## 1F. Semantic search CLI

Expose:

```text
winston search "woman with a pink stroller"
```

Pipeline:

```text
query
  ↓
Jina text embedding
  ↓
Qdrant cosine retrieval
  ↓
top visual candidates
  ↓
group by asset + temporal proximity
  ↓
user-facing results
```

Return at least:

```text
source file
timestamp when applicable
matching region
raw semantic score
```

Do not invent a percentage confidence before calibration.

## 1G. Evaluation baseline

Create a fixed recorded-video/photo evaluation set with concepts that are intentionally not detector classes used by Winston.

Initial query families:

```text
pink stroller
person wearing a green cap
person carrying a cardboard box
red bicycle
umbrella
blue car
```

Measure:

- Recall@K;
- Precision@K;
- indexing throughput;
- vectors/hour of footage;
- Qdrant search latency.

### Phase 1 exit criteria

Winston can reliably retrieve at least several unseen concepts in the fixed evaluation set without any predefined object ontology.

---

# Phase 2 — Coarse-to-fine temporal refinement

**Objective:** improve recall and timestamp precision without globally indexing every video frame.

Problem:

A short event may occur completely between two keyframes because keyframes are codec/compression decisions, not semantic-event boundaries.

Design:

```text
sparse global keyframe index
        ↓
Qdrant candidate at time t
        ↓
bounded temporal window around t
        ↓
decode intermediate frames
        ↓
local denser sampling
        ↓
local embedding/reranking
        ↓
precise timestamp/range
```

## Work items

- group nearby sparse hits into candidate windows;
- configurable refinement window around hits;
- dense local decoding;
- local semantic reranking;
- avoid globally persisting every refinement frame unless measurements justify it.

## Sampling experiments

Add alternatives behind the same `FrameSampler` contract:

```text
PeriodicSampler
SceneChangeSampler
HybridSampler
```

Benchmark them against `KeyframeSampler`; do not replace the baseline based only on intuition.

### Phase 2 exit criteria

- short-event recall measurably improves;
- returned timestamps/ranges are more precise;
- added query latency is measured and acceptable.

---

# Phase 3 — OCR and ALPR

**Objective:** support exact/fuzzy textual identifiers visible in media.

Reference query:

```text
blue car with license plate XXX-YYY-ZZZ
```

Separate the query into:

```text
blue car        -> semantic visual evidence
XXX-YYY-ZZZ     -> exact/fuzzy OCR evidence
```

Candidate architecture:

```text
semantic visual retrieval
        ↓
candidate frames/windows
        ↓
plate/text localization
        ↓
OCR / ALPR
        ↓
normalization + exact/fuzzy matching
        ↓
combined ranking
```

## Work items

- benchmark OCR/ALPR models on actual surveillance quality;
- normalize license-plate formats;
- retain recognized text, confidence, asset, and timestamp;
- exact match + fuzzy/typo-tolerant search;
- determine whether ALPR remains candidate-time refinement or becomes an offline enrichment index.

### Phase 3 exit criteria

A query combining visual attributes and a readable plate identifies the correct vehicle on the evaluation set.

---

# Phase 4 — Temporal/action understanding

**Objective:** answer queries that cannot be proven from one frame.

Reference query:

```text
red car running a red light
```

Static co-occurrence of a red car and red traffic light is insufficient.

Architecture:

```text
semantic candidate retrieval
        ↓
candidate temporal windows
        ↓
dense multiframe representation
        ↓
temporal verification/reasoning
        ↓
rerank
```

## Research tracks

Benchmark before selecting one:

- video/multiframe embedding model;
- VLM over selected frames;
- local motion/tracking plus explicit event logic;
- hybrid semantic retrieval -> local temporal verifier.

Tracking is acceptable here as a **local verification technique**. It must not become a global prerequisite for what Winston can search.

### Phase 4 exit criteria

The controlled evaluation can distinguish:

```text
red car near a red light
```

from:

```text
red car actually crossing against the red light
```

---

# Phase 5 — Optional VLM descriptions and hybrid retrieval

**Objective:** determine whether generated textual descriptions improve abstract/context/action retrieval enough to justify their cost.

Potential flow:

```text
candidate frame/window
      ↓
VLM description
      ↓
text embedding/index
```

Search may later combine:

```text
text -> visual embeddings
text -> description embeddings
OCR/ALPR evidence
temporal verifier evidence
```

## Work items

- choose description granularity: region, frame, or temporal window;
- measure indexing cost and retrieval gain;
- preserve provenance of generated text;
- calibrate/fuse scores instead of assuming different retrieval distributions are directly comparable.

This phase remains optional if the direct visual pipeline already performs well enough.

---

# Phase 6 — API and minimal UI

**Objective:** expose a validated engine without distracting from early ML/indexing work.

Start only after the Phase 1/2 backend is demonstrably useful.

Potential API:

```text
POST /assets
POST /index
GET  /index/{job_id}
POST /search
GET  /media/{asset_id}
```

Minimal user flow:

```text
add/select media
      ↓
search in natural language
      ↓
see ranked visual results
      ↓
open video at matching timestamp/range
```

Frontend technology remains deliberately undecided until the backend contract is stable.

---

# Phase 7 — Continuous/live ingestion

**Objective:** extend the proven offline engine to continuous camera streams.

This is separate from `feat/public-stream-recorder`, which is one-shot dataset acquisition tooling.

Future live ingestion may support RTSP/HLS and continuously generate indexable media chunks.

Before implementation define:

- chunking strategy;
- retention/storage lifecycle;
- backpressure;
- indexing-lag target;
- restart/recovery semantics;
- camera/source identity model.

Do not introduce this operational complexity before offline retrieval quality is stable.

---

# Recommended implementation/merge order

From the current repository state:

```text
main
 │
 ├─ review + merge PR #1  feat/media-ingest
 │
 ├─ review + merge PR #2  feat/public-stream-recorder
 │    └─ independent dataset-tooling workstream
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
 └─ API/UI/live ingestion after retrieval quality is proven
```

The recorder PR does not need to block keyframe/embedding/index work; it is independent except for providing useful real-world media.

---

# Milestones

## M0 — Media foundation

- recorded media can be discovered/probed;
- public-recording tooling can create realistic datasets independently;
- no footage is committed to the repository.

## M1 — Unseen visual concept search

- videos/photos indexed;
- keyframes + class-agnostic regions embedded with Jina CLIP v1;
- Qdrant semantic retrieval works;
- natural-language CLI search works;
- unseen concept queries such as `pink stroller` require no predefined class.

## M2 — Reliable video localization

- sparse results grouped into temporal windows;
- dense refinement recovers more short events and improves timestamps.

## M3 — Exact visual text

- OCR/ALPR provides reliable exact/fuzzy plate retrieval.

## M4 — Actions

- multi-frame temporal verification handles action queries beyond static co-occurrence.

## M5 — Product surface

- stable API;
- minimal UI;
- optional continuous camera ingestion afterward.

---

# Definition of success

Winston succeeds when a future query does not need to have been anticipated when the video or photo was indexed.

The global semantic index remains broad and class-agnostic; specialized OCR and temporal components add precision without turning Winston back into a predefined object-class search system.
