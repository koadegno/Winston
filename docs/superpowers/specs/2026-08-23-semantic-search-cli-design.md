# Phase 1F — Semantic search CLI design

## Status

Approved design direction for GitHub issue #10: **Phase 1F — Add semantic search CLI**.

This phase adds Winston's first natural-language retrieval path over the visual index created by Phase 1E:

```text
winston search "woman with a pink stroller"
```

The design deliberately adapts the temporal result-selection idea from the Semantic File Explorer reference video:

```text
similarity timeline
  ↓
moving average
  ↓
z-score
  ↓
maximum subarray / Kadane
  ↓
selected temporal interval
```

Winston does not copy the reference implementation literally. The original explorer scores the I-frames of each video directly. Winston already has a large-scale Qdrant ANN index and multiple class-agnostic regions per keyframe, so Phase 1F uses Qdrant as a coarse candidate generator and applies the temporal algorithm only inside bounded candidate neighborhoods.

## Goals

Phase 1F must:

- expose `winston search <query>`;
- embed the query exactly once with the configured `MultimodalEmbedder.embed_texts()` implementation;
- require the same embedding identity used by the existing visual collection;
- retrieve visual candidates from Qdrant using the existing named `visual` vector and cosine similarity;
- return both photos and video passages;
- collapse multiple regions from one photo into one user-facing photo result;
- collapse multiple regions from one video keyframe into one temporal score before temporal processing;
- group nearby video hits into coherent temporal neighborhoods rather than returning many almost-identical tiles/keyframes;
- smooth the per-keyframe similarity timeline, z-score it, and use Kadane's maximum-subarray algorithm to select the strongest contiguous interval within a candidate neighborhood;
- keep the strongest matching region/keyframe as the representative visual for each result;
- expose the representative raw cosine semantic score rather than inventing a confidence percentage;
- keep all Qdrant SDK types inside the Qdrant implementation boundary;
- keep memory bounded while rescoring temporal neighborhoods;
- remain deterministic and thoroughly testable without a real embedding service for unit tests.

## Non-goals

Phase 1F does not implement:

- decoding P-frames or any frame that was not already indexed;
- dense temporal refinement around a hit;
- tracking or object detection;
- OCR or ALPR;
- action understanding;
- VLM-generated descriptions;
- calibrated confidence percentages;
- relevance thresholds learned from evaluation data;
- persistent query/result history;
- a frontend or API;
- concurrent search refinement across many videos;
- automatic migration or recreation of incompatible Qdrant collections.

Dense local decoding remains Phase 2. Phase 1F works exclusively with the keyframes and deterministic regions already persisted in the Phase 1E Qdrant collection.

## Reference algorithm and Winston adaptation

The Semantic File Explorer reference algorithm is conceptually:

```text
all I-frames in one video
        ↓
CLIP similarity(query, frame)
        ↓
moving average
        ↓
z-score among that video's frame scores
        ↓
maximum subarray / Kadane
        ↓
average selected subarray scores
        ↓
best video passage
```

The reference material does not specify the moving-average width. Winston therefore treats that width as an explicit configurable V0 parameter rather than embedding an invented claim about the source implementation.

A literal full-dataset implementation would undermine the reason Winston uses Qdrant. Conversely, applying Kadane directly to only Qdrant's top-K hits is invalid because low-scoring intermediate keyframes would be absent from the timeline even though they are necessary to determine whether a semantic event is temporally continuous.

The chosen adaptation is therefore coarse-to-fine:

```text
query text
    ↓
Jina text embedding
    ↓
Qdrant cosine ANN
    ↓
coarse visual hits
    ↓
photo hits ───────────────────────────────→ collapse per photo
    │
    └─ video hits
          ↓
       collapse duplicate regions at same timestamp
          ↓
       seed bounded temporal neighborhoods
          ↓
       merge overlapping neighborhoods per asset
          ↓
       fetch all indexed vectors in each neighborhood
          ↓
       exact cosine score for every stored region
          ↓
       max region score per keyframe timestamp
          ↓
       moving average
          ↓
       z-score inside the neighborhood
          ↓
       Kadane maximum subarray
          ↓
       representative strongest raw region
          ↓
       video passage result
    ↓
rank user-facing results by representative raw cosine score
    ↓
limit
```

This preserves Qdrant as the global scalable retrieval engine while ensuring the temporal algorithm receives a complete local sequence rather than a sparse list of only the strongest ANN hits.

## Alternatives considered

### A. Score every indexed visual in every video for every query

This is the closest conceptual reproduction of the reference explorer.

It was rejected for Phase 1F because it turns every query into a dataset-wide scan and discards the scaling advantage already gained by Qdrant/HNSW.

### B. Run temporal grouping directly over Qdrant top-K hits

This is cheap and simple, but it is semantically wrong for moving-average/z-score/Kadane processing. Qdrant intentionally omits weak candidates. Missing intermediate keyframes would make unrelated strong points appear adjacent and would prevent weak valleys from splitting passages.

This approach was rejected.

### C. Qdrant coarse retrieval plus complete bounded local timelines

This is the chosen design.

Qdrant identifies promising assets/timestamps. Winston then retrieves all already-indexed vectors inside small temporal neighborhoods and performs exact local scoring before smoothing and interval selection.

This adds bounded query-time work but retains global ANN scaling and keeps the temporal input complete inside each candidate neighborhood.

## Package boundaries

Phase 1F adds a dedicated `search` package while extending the existing `index` persistence boundary.

```text
src/winston/
├── cli.py
├── config.py
├── embeddings/
│   └── ...
├── index/
│   ├── base.py
│   ├── models.py
│   └── qdrant.py          # read-only search + bounded vector-window retrieval
└── search/
    ├── __init__.py
    ├── models.py          # Winston-owned query/result values
    ├── temporal.py        # score collapse, smoothing, z-score, Kadane
    └── pipeline.py        # query orchestration and resource lifecycle
```

The `search` package may depend on Winston-owned `index` and embedding contracts. It must not import or expose `qdrant_client.models`.

## Search configuration

Add a nested search configuration:

```text
Settings
├── embedding
├── qdrant
├── indexing
└── search
    ├── result_limit = 10
    ├── candidate_limit = 200
    ├── temporal_context_seconds = 15.0
    ├── moving_average_frames = 3
    └── timeline_page_size = 256
```

Environment overrides follow the existing nested-settings convention:

```text
SEARCH__RESULT_LIMIT=10
SEARCH__CANDIDATE_LIMIT=200
SEARCH__TEMPORAL_CONTEXT_SECONDS=15
SEARCH__MOVING_AVERAGE_FRAMES=3
SEARCH__TIMELINE_PAGE_SIZE=256
```

Rules:

- all integer counts are positive;
- `temporal_context_seconds` is finite and strictly positive;
- `moving_average_frames` is a positive odd integer so the smoothing window can be centered deterministically;
- defaults are V0 baselines to be measured and tuned in Phase 1G, not claimed as calibrated values.

The reference video does not disclose its moving-average width. Winston's default of three indexed keyframes is therefore an explicit project choice.

## Read-only visual-index session

The current Phase 1E `ensure_compatible()` contract is write-oriented: it creates the collection if it does not exist and requires the local dataset identity.

Search must never create an empty collection. Phase 1F therefore adds a separate read-only open/validation operation conceptually like:

```python
@dataclass(frozen=True, slots=True)
class VisualSearchSession:
    dataset_instance_id: str
    index_instance_id: str


class VisualIndex(Protocol):
    async def open_search(
        self,
        identity: EmbeddingIdentity,
    ) -> VisualSearchSession: ...
```

`open_search()` must:

- require the configured collection to already exist;
- validate Winston schema version 2;
- validate vector name, dimension, cosine distance, model ID, and preprocessing version;
- validate that stored `dataset_instance_id` and `index_instance_id` are UUIDs;
- return those IDs as Winston-owned session metadata;
- never create, drop, migrate, or otherwise mutate the collection.

`winston search` does not take a dataset path. The configured Qdrant collection already declares which dataset it owns through schema-v2 metadata.

A missing collection produces an actionable error telling the user to index media first.

## Winston-owned retrieval models

Qdrant results are translated at the persistence boundary into Winston domain values.

A useful shape is:

```python
@dataclass(frozen=True, slots=True)
class ScoredVisual:
    visual: IndexedVisual
    score: float
```

`score` is cosine similarity, where larger values are more similar.

Using `IndexedVisual` for returned records has two useful properties:

- payload provenance continues to pass the same strict Winston validation used for writes;
- the search layer receives NumPy float32 vectors and Winston enums/models, never Qdrant SDK structures.

The Qdrant backend reconstructs `IndexedVisual` values from payload + stored named vector + the already validated embedding identity.

All returned scores and vectors must be finite. Malformed Winston-owned payload/vector data is a search failure, not silently ignored corruption.

## Coarse Qdrant retrieval

The query is embedded once:

```text
embedder.embed_texts([query])
```

The result must contain exactly one vector matching the embedder identity dimension.

The Qdrant backend then performs one ANN query against:

```text
collection = configured collection
vector     = configured named `visual` vector
metric     = cosine
limit      = effective coarse candidate limit
```

For the normal CLI path:

```text
effective coarse limit = max(search.candidate_limit, requested result limit)
```

The backend requests payload data and the stored vector needed to reconstruct Winston domain values.

No percentage conversion is performed.

## Photo result collapse

A photo can have a full-image vector plus several deterministic crop vectors. Returning each region as a separate result would duplicate the same source photo.

For coarse photo hits:

```text
all returned hits for one asset_id
        ↓
choose highest raw cosine score
        ↓
one photo SearchResult
```

The winning hit preserves:

- `source_path`;
- region kind;
- exact region geometry;
- raw cosine score.

Photo results have no temporal interval or timestamp.

## Video coarse-hit collapse

A single keyframe can also produce many Qdrant hits because full frame and crops share the same timestamp.

Before creating temporal neighborhoods, coarse video hits are collapsed by:

```text
(asset_id, timestamp_us)
```

For each timestamp, retain only the highest-scoring region as the seed representative.

This prevents one visually strong keyframe with many matching crops from creating many redundant temporal seeds.

## Candidate temporal neighborhoods

Each collapsed coarse video timestamp `t` seeds:

```text
[t - temporal_context_seconds,
 t + temporal_context_seconds]
```

The lower bound is clipped to zero.

For the same asset, overlapping or touching seed windows are merged. Distant semantic events in the same video therefore remain separate candidate neighborhoods and can become separate user-facing results.

Example with a 15-second context:

```text
coarse hits:
  100 s
  108 s
  310 s

seed windows:
   85 ───── 115
   93 ───── 123
  295 ───── 325

merged:
   85 ───────── 123
  295 ──────── 325
```

The context value is intentionally configurable because event duration and keyframe cadence vary by source encoding.

## Complete neighborhood retrieval

Kadane cannot operate correctly on only the high-scoring ANN hits. For every merged video neighborhood, the persistence layer therefore exposes a bounded iterator conceptually like:

```python
async def iter_visuals(
    *,
    asset_id: str,
    start_timestamp_us: int,
    end_timestamp_us: int,
) -> AsyncIterator[IndexedVisual]: ...
```

The Qdrant implementation uses filtered scroll/pagination with:

```text
asset_id == candidate asset
AND media_type == video
AND timestamp_us >= start
AND timestamp_us <= end
```

and requests the named vector.

`timeline_page_size` bounds each Qdrant page. Vectors from one page are scored and discarded before the next page is processed.

Winston does not materialize every region vector for the neighborhood in memory. It retains only the best scored region seen for each keyframe timestamp.

No schema-version bump is required: Phase 1E already persists `asset_id`, `media_type`, `timestamp_us`, region provenance, and the named vector required by this flow.

Phase 1F does not add payload indexes solely for this first baseline; that optimization can be measured before changing persisted index requirements.

## Exact local cosine scoring

The coarse Qdrant scores identify promising neighborhoods, but every visual inside a selected neighborhood needs a score to build a complete timeline.

For each stored region vector `v` and query vector `q`, Winston computes:

```text
cosine(q, v) = (q · v) / (||q|| ||v||)
```

using the stored float32 vector and the same query embedding.

The implementation must reject zero-norm or non-finite vectors rather than producing NaN/Inf scores.

This local score has the same semantic meaning as the Qdrant cosine similarity used for coarse retrieval. Integration tests with real Qdrant must assert that Winston's exact local score agrees with Qdrant's returned cosine score within floating-point tolerance for known vectors.

The score is not clamped or converted into a confidence percentage.

## One temporal score per keyframe

After local scoring, multiple regions belonging to the same keyframe are reduced to one timeline observation:

```text
keyframe @ 12.0 s

full      0.27
tile 1    0.31
tile 2    0.61
tile 3    0.29

        ↓ max

timestamp 12.0 s → score 0.61
                    representative = tile 2
```

This is essential for Winston's class-agnostic crops: a small object can drive the timestamp score without forcing the complete frame to match strongly.

The representative region is preserved for the final result.

The timeline is sorted by exact integer `timestamp_us` before temporal processing.

## Moving average

Let the collapsed raw keyframe scores be:

```text
s[0], s[1], ..., s[n-1]
```

Winston applies a centered moving average with odd width `w = moving_average_frames`.

For each index, the average uses the available indices inside the centered window and truncates naturally at the beginning/end of the candidate neighborhood.

With the default width `3`:

```text
m[i] = mean(s[max(0, i-1) : min(n, i+2)])
```

The moving average affects temporal interval selection only. It does not replace the raw representative cosine score exposed to the user.

If the neighborhood has fewer than three distinct keyframe timestamps, Winston skips smoothing/z-score/Kadane and returns the strongest raw keyframe as a single-timestamp passage.

## Z-score normalization

The smoothed scores are normalized inside each candidate neighborhood:

```text
mean = average(m)
std  = population standard deviation(m)

z[i] = (m[i] - mean) / std
```

Population standard deviation (`ddof=0`) is used because the complete bounded neighborhood is the sequence being normalized, not a statistical sample from an unknown larger population.

If the standard deviation is zero (or numerically indistinguishable from zero), the neighborhood carries no useful relative temporal signal. Winston falls back to the strongest raw keyframe rather than manufacturing an arbitrary multi-frame range.

The z-score is an internal temporal-selection signal. It is not a user-facing confidence and is not compared directly to raw cosine scores from other results.

## Kadane maximum-subarray selection

Kadane's algorithm runs over the z-score sequence to find the contiguous interval with maximum total normalized evidence.

Deterministic tie-breaking is:

1. larger subarray sum;
2. if equal, shorter interval;
3. if still equal, earlier start timestamp.

The shorter-interval tie-break applies only when sums are equal; it does not otherwise alter the reference maximum-subarray objective.

If no positive signal can be selected, Winston falls back to the strongest raw keyframe.

The reference video describes averaging the scores in the selected subarray after Kadane. Winston may compute the mean selected z-score internally for diagnostics/tests, but Phase 1F does not expose it as confidence and does not use it to replace the raw semantic score contract required by Issue #10.

## Video SearchResult mapping

For one selected temporal interval:

```text
start_timestamp = first selected keyframe timestamp
end_timestamp   = last selected keyframe timestamp
```

Winston does not infer unobserved P-frame boundaries or pretend the event begins/ends between sampled keyframes.

The representative visual is the highest raw cosine-scoring region among the selected keyframe timestamps.

A video result therefore contains at least:

```text
source_path
media_type = video
start_timestamp_seconds
end_timestamp_seconds
representative_timestamp_seconds
representative region kind + geometry
raw_score
```

`raw_score` is the representative region's uncalibrated cosine similarity.

One merged candidate neighborhood produces at most one passage result. Separate non-overlapping neighborhoods from the same asset can produce separate results.

## User-facing result model

A single Winston-owned result type can represent both media types:

```python
@dataclass(frozen=True, slots=True)
class SearchResult:
    source_path: str
    media_type: MediaType
    start_timestamp_seconds: float | None
    end_timestamp_seconds: float | None
    representative_timestamp_seconds: float | None
    region_kind: RegionKind
    region: RegionGeometry
    raw_score: float
```

Invariants:

- images have all timestamp fields `None`;
- videos have all three timestamp fields present and non-negative;
- video start <= representative <= end;
- score must be finite;
- region/source provenance follows the same strict rules as indexed visuals.

No `confidence` field exists in Phase 1F.

## Final ranking

Moving-average/z-score/Kadane processing chooses the temporal extent of a video result. It does not redefine cross-result semantic ranking.

All photo/video results are sorted by:

```text
raw_score DESC
source_path ASC
start_timestamp_seconds ASC (videos)
```

The raw score is always the strongest representative region's cosine similarity.

This keeps Issue #10's ranking contract explicit and avoids comparing per-neighborhood z-scores, whose distributions are intentionally local.

After deterministic ranking:

```text
results = results[:requested_limit]
```

No threshold is applied in Phase 1F. A later evaluation phase may establish useful cutoffs.

## Search pipeline orchestration

The high-level search flow is:

```text
validate non-empty query
        ↓
create embedder once
        ↓
open existing visual index read-only
        ↓
validate embedding/index compatibility
        ↓
embed query once
        ↓
coarse Qdrant ANN
        ↓
collapse photos + video seed timestamps
        ↓
create/merge video candidate neighborhoods
        ↓
process neighborhoods sequentially
        │
        ├─ paginated vector retrieval
        ├─ exact cosine scoring
        ├─ max region per timestamp
        ├─ moving average
        ├─ z-score
        └─ Kadane
        ↓
combine photo + video results
        ↓
raw-score deterministic ranking
        ↓
limit
        ↓
close embedder + visual index
```

The embedder and Qdrant client are constructed once and reused for the complete command.

Candidate video neighborhoods are refined sequentially in V0. Query-time parallelism is deliberately deferred until measurements show it is needed.

## Resource and failure semantics

Search failures are run-level failures because there is no durable per-asset work to continue after a failed query.

Examples:

- embedding backend cannot be created;
- query embedding fails;
- Qdrant is unavailable;
- collection is missing or incompatible;
- stored Winston payload/vector is malformed;
- coarse retrieval fails;
- bounded neighborhood retrieval fails;
- local scoring encounters invalid vector data.

The command returns exit status `1` and an actionable stderr diagnostic.

An empty but otherwise valid search result is not an error and returns exit status `0`.

Cleanup follows the Phase 1E principle: a close failure must not hide an earlier primary exception. Cleanup errors are attached as diagnostic notes when a primary error already exists.

## CLI contract

Add:

```text
winston search QUERY [--limit N]
```

Examples:

```text
winston search "woman with a pink stroller"
winston search "red bicycle" --limit 20
```

`--limit` defaults to `Settings.search.result_limit` and must be positive.

A blank/whitespace-only query is rejected before constructing expensive dependencies.

Example output shape:

```text
1. camera/parking.mkv
   passage: 00:18:39.500 -> 00:18:47.200
   representative: 00:18:42.320
   region: tile x=896 y=504 width=896 height=504 scale=0.333333
   raw score: 0.421893

2. photos/stroller.jpg
   region: tile x=0 y=0 width=896 height=504 scale=0.333333
   raw score: 0.398411
```

The output must call the value `raw score` or otherwise make clear it is an uncalibrated semantic similarity. It must not display a percent sign.

No progress logger is required for the initial search path. Search diagnostics/errors go to stderr; results go to stdout.

## Testing strategy

Development follows strict TDD: failing contract test first, implementation second, then targeted GitHub Actions validation.

### Search configuration and CLI

Test:

- search defaults;
- nested environment overrides;
- invalid moving-average widths;
- `winston search QUERY` parser behavior;
- `--limit` validation;
- blank query rejection;
- result formatting for image and video passages;
- empty result behavior;
- fatal error exit status.

### Qdrant read path

With a fake typed client, test:

- missing collection fails without creating it;
- schema/model/vector/distance mismatches are rejected;
- dataset/index instance IDs are validated and returned;
- coarse query uses the configured named vector and requested limit;
- Qdrant payload/vector values map to strict Winston domain models;
- malformed payload/vector/score is rejected;
- temporal scroll uses asset/time filters and bounded page size;
- pagination does not materialize all vectors at once;
- Qdrant SDK exceptions are wrapped as Winston index/search errors.

### Temporal algorithm

Use deterministic synthetic timelines to test:

- multiple regions at one timestamp collapse to the maximum raw score;
- representative geometry follows that maximum region;
- centered moving-average output;
- population z-score values;
- zero-variance fallback;
- short-neighborhood fallback;
- Kadane selects the expected contiguous interval;
- deterministic tie-breaking;
- an isolated raw spike is reduced by smoothing compared with a sustained cluster;
- two distant coarse-hit groups in one asset remain two result neighborhoods;
- overlapping seed windows merge;
- selected passage preserves exact keyframe timestamps.

### Pipeline

With fake embedder/index implementations, test:

- query is embedded exactly once;
- embedder identity is passed to read-only index validation;
- photo hits are deduplicated by asset;
- video regions are deduplicated by timestamp before seed generation;
- each merged neighborhood is refined exactly once;
- photo/video results are combined and sorted by raw score;
- result limit applies after grouping/refinement;
- resources close on success and failure;
- cleanup errors do not replace primary diagnostics.

### Real Qdrant integration

Use `qdrant/qdrant:v1.18.2` in GitHub Actions to verify:

- an existing schema-v2 collection can be opened read-only;
- known query vectors retrieve the expected ordered coarse points;
- bounded temporal filtered retrieval returns all expected points;
- local cosine scores agree with Qdrant cosine scores within tolerance;
- photos and video regions map through the real payload/vector representation correctly.

### Real Jina end-to-end validation

During development, use a dedicated ephemeral GitHub Actions validation workflow/branch with the existing Jina API secret to exercise a tiny real semantic example, such as two simple images where a matching text query must rank the semantically corresponding image above the other.

This validation is evidence for the complete text-embedding → Qdrant-search path. It should not force a flaky external-network test into every normal unit-test run.

Any temporary validation workflow/branch must be removed after successful validation, following the Winston repository workflow rules.

## CI integration

The retained Phase 1F CI should extend the existing Qdrant-focused workflow or add a narrowly named permanent search workflow only if that produces a clearer stable boundary.

Permanent CI must include:

- locked dependency check;
- Phase 1F unit tests;
- real Qdrant 1.18.2 integration tests;
- relevant prior index regression tests;
- `python -m compileall -q src tests`.

A temporary full-regression workflow may be used during development when useful, but temporary workflows are removed before final PR review.

## Compatibility and persistence

Phase 1F is read-only with respect to indexed media data.

It does not change:

- deterministic point IDs;
- schema-v2 collection ownership;
- embedding model identity;
- vector dimension;
- preprocessing version;
- stored payload fields.

Therefore Phase 1F does **not** require a Winston Qdrant schema-version bump or reindex solely to enable search.

Search must continue to reject older/incompatible collections explicitly rather than attempting implicit adoption.

## Memory and scaling bounds

The global search is ANN through Qdrant.

Local refinement is bounded by:

- a finite coarse candidate count;
- bounded temporal context around coarse hits;
- merged neighborhoods rather than full-dataset rescoring;
- paginated Qdrant vector retrieval;
- processing neighborhoods sequentially;
- retaining only one best scored visual per timestamp after page processing.

The algorithm never intentionally stores all region vectors for all candidate videos in memory at once.

The primary V0 cost to measure in Phase 1G is the amount of vector data transferred from Qdrant for temporal refinement. If this becomes material, possible later optimizations include payload indexes, larger/coarser seed selection, server-side grouping, or other retrieval strategies. Those optimizations are not introduced before measurement.

## Relationship to Phase 2

Phase 1F returns passages based on already-indexed sparse keyframes.

It does not solve the known failure mode where a short event occurs entirely between keyframes.

Phase 2 remains:

```text
Phase 1F selected sparse passage
        ↓
bounded temporal refinement window
        ↓
decode intermediate frames from original media
        ↓
denser local embeddings
        ↓
rerank / refine timestamps
```

Phase 1F should therefore expose precise source path and representative/start/end timestamps so Phase 2 can build directly on its output without redesigning search provenance.

## Success criteria

Phase 1F is complete when:

1. `winston search "woman with a pink stroller"` embeds the query with the configured Jina CLIP v1 family and searches the existing schema-v2 visual collection;
2. indexed photos and video regions can both be retrieved;
3. multiple regions from the same photo do not appear as duplicate photo results;
4. multiple nearby regions/keyframes from one video are converted into a coherent passage using the approved smoothing → z-score → Kadane design;
5. separate distant neighborhoods in the same video can remain separate results;
6. each result exposes enough provenance to inspect the original media: source path, temporal location where applicable, region geometry, and raw cosine score;
7. final cross-result ranking is deterministic and based on raw cosine similarity, not a fabricated confidence percentage;
8. Qdrant types remain confined to the persistence implementation;
9. unit tests cover embedding orchestration, retrieval mapping, temporal processing, ranking, CLI behavior, and cleanup;
10. real Qdrant integration verifies retrieval and local/Qdrant cosine agreement;
11. a real Jina development validation demonstrates text-to-image retrieval end to end;
12. all retained GitHub Actions are green before PR review.

## Explicit V0 decisions to revisit with measurements

The following are intentional baselines, not permanent truths:

```text
candidate_limit = 200
temporal_context_seconds = 15
moving_average_frames = 3
timeline_page_size = 256
final result limit = 10
```

Phase 1G should measure their effect on:

- Recall@K;
- Precision@K;
- duplicate-result rate;
- temporal passage width;
- Qdrant search latency;
- local refinement latency;
- vector bytes transferred per query.

The reference Semantic File Explorer itself notes that its temporal algorithm can select ranges that are too large. Winston therefore keeps all these temporal parameters explicit and measurable rather than presenting them as calibrated constants.