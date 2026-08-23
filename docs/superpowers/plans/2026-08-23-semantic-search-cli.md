# Phase 1F — Semantic Search CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GitHub issue #10 by adding `winston search QUERY [--limit N]`, using Jina CLIP text embeddings, Qdrant coarse retrieval, and bounded temporal grouping inspired by the Semantic File Explorer reference algorithm.

**Architecture:** Keep `cli.py` thin. Extend Winston's `VisualIndex` boundary with read-only retrieval methods, then add a dedicated `winston.search` package. Global candidate discovery remains ANN through Qdrant. Video candidates are refined only inside bounded temporal neighborhoods by rescoring all already-indexed regions, collapsing to one score per keyframe, applying a centered moving average, population z-score normalization, and Kadane maximum-subarray selection. Cross-result ranking always uses the representative raw cosine similarity, never a fabricated confidence percentage.

**Tech Stack:** Python 3.13+, argparse, asyncio, NumPy, Pydantic/Pydantic Settings, existing Jina CLIP v1 embedding engines, qdrant-client 1.18.x, Qdrant 1.18.2, pytest, pytest-asyncio, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-23-semantic-search-cli-design.md`

## Global Constraints

- Implement only GitHub issue #10 / Phase 1F. Dense intermediate-frame decoding remains Phase 2.
- The implementation stays in one draft PR titled `feat: add semantic search CLI`; the PR body contains `Closes #10`.
- Do not merge or delete `feat/semantic-search-cli` without explicit user approval.
- For Winston, do not use Mac Studio / local MCP execution. All test execution happens through GitHub Actions.
- Temporary validation workflows/branches are allowed only for testing and must be removed after successful validation.
- Follow strict GitHub-only TDD for every implementation task:

```text
write focused failing tests
        ↓
commit/push test contract (`test: ...`)
        ↓
GitHub Action proves RED for the intended reason
        ↓
implement minimal behavior
        ↓
commit/push implementation (`feat:` / `fix:`)
        ↓
GitHub Action proves GREEN
        ↓
next task
```

- A RED run must fail because the new behavior is absent, not because of syntax/import/workflow mistakes.
- Do not introduce `typing.Any` or `object` in new Winston production code or new Phase 1F tests. Use concrete domain/Qdrant types and narrow `Protocol`s.
- Every new function/method gets a useful docstring. Non-obvious temporal math gets concise comments/docstrings with numeric examples.
- In particular, document with examples:
  - multiple regions at one timestamp collapsing to the maximum raw score;
  - `temporal_context_seconds` creating/merging candidate windows;
  - centered `moving_average_frames=3` behavior at edges and interior points;
  - z-score being an internal local signal rather than confidence;
  - Kadane finding the strongest contiguous positive normalized interval.
- Explicitly distinguish parameters that can change semantic output (`temporal_context_seconds`, `moving_average_frames`) from parameters that mainly bound retrieval/runtime cost (`candidate_limit`, `timeline_page_size`).
- Query text is embedded exactly once per search invocation.
- Search opens the existing schema-v2 collection read-only. It must never create, drop, migrate, clear, rename, or adopt a collection.
- Keep Qdrant SDK types confined to `src/winston/index/qdrant.py` and Qdrant adapter tests.
- Preserve current schema version 2 and existing deterministic point IDs/payload format; Phase 1F requires no reindex solely to enable search.
- Preserve Jina CLIP v1 embedding identity, 768 dimensions, preprocessing version 1, and cosine metric.
- Do not convert cosine similarity into a percentage or confidence score.
- Final photo/video ranking uses representative raw cosine similarity descending; local z-scores never participate in cross-result ranking.
- Refine candidate video neighborhoods sequentially in V0 and page Qdrant reads with `timeline_page_size`; do not materialize all candidate vectors across all videos at once.
- Do not add Qdrant payload indexes until Phase 1G measurements justify them.

## V0 Search Defaults

```text
result_limit             = 10
candidate_limit          = 200
temporal_context_seconds = 15.0
moving_average_frames    = 3
timeline_page_size       = 256
```

These are explicit V0 baselines to measure in Phase 1G, not calibrated constants and not values copied from the reference video.

## File Map

### Create

```text
src/winston/search/models.py
src/winston/search/temporal.py
src/winston/search/pipeline.py

tests/src/winston/search/test_models.py
tests/src/winston/search/test_temporal.py
tests/src/winston/search/test_pipeline.py
tests/src/winston/index/test_qdrant_search.py

tests/integration/test_semantic_search_qdrant.py
```

### Modify

```text
src/winston/config.py
src/winston/index/base.py
src/winston/index/models.py
src/winston/index/__init__.py
src/winston/index/qdrant.py
src/winston/search/__init__.py
src/winston/cli.py

tests/src/winston/test_config.py
tests/src/winston/test_cli.py
.github/workflows/qdrant-index.yml
```

### Delete

```text
tests/src/winston/search/.gitkeep
```

---

## Task 1: Search configuration and user-facing result models

**Files:** `src/winston/config.py`, `src/winston/search/models.py`, `src/winston/search/__init__.py`, `tests/src/winston/test_config.py`, `tests/src/winston/search/test_models.py`.

**Interfaces:**

```python
class SearchSettings(BaseSettings):
    result_limit: PositiveInt = 10
    candidate_limit: PositiveInt = 200
    temporal_context_seconds: Annotated[FiniteFloat, Field(gt=0)] = 15.0
    moving_average_frames: PositiveInt = 3
    timeline_page_size: PositiveInt = 256


class SearchRunError(RuntimeError): ...


class SearchResult(BaseModel):
    source_path: str
    media_type: MediaType
    start_timestamp_seconds: float | None
    end_timestamp_seconds: float | None
    representative_timestamp_seconds: float | None
    region_kind: RegionKind
    region: RegionGeometry
    raw_score: float
```

- [ ] Write config tests for defaults, nested `SEARCH__...` overrides, non-positive counts/context, non-finite context, and even `moving_average_frames` rejection.
- [ ] Commit the tests as `test: define semantic search configuration contracts`; run the focused GitHub Action and verify intended RED.
- [ ] Implement `SearchSettings` with the existing nested Pydantic settings convention and an odd-width validator; no direct environment reads.
- [ ] Add tests for strict/frozen `SearchResult`: image timestamps all `None`; video timestamps all present; `start <= representative <= end`; finite raw score; normalized path; invalid mixed state rejected.
- [ ] Implement result validation by reusing `MediaType`, `RegionKind`, and `RegionGeometry`.
- [ ] Commit as `feat: add semantic search configuration and result models`; verify focused Action GREEN.

---

## Task 2: Winston-owned visual-index search contracts

**Files:** `src/winston/index/base.py`, `src/winston/index/models.py`, `src/winston/index/__init__.py`, `tests/src/winston/index/test_qdrant_search.py`.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class VisualSearchSession:
    dataset_instance_id: str
    index_instance_id: str


@dataclass(frozen=True, slots=True)
class ScoredVisual:
    visual: IndexedVisual
    score: float


class VisualIndex(Protocol):
    async def open_search(self, identity: EmbeddingIdentity) -> VisualSearchSession: ...
    async def search_visuals(
        self,
        query_vector: VisualVector,
        *,
        limit: int,
    ) -> Sequence[ScoredVisual]: ...
    def iter_visuals(
        self,
        *,
        asset_id: str,
        start_timestamp_us: int,
        end_timestamp_us: int,
        page_size: int,
    ) -> AsyncIterator[IndexedVisual]: ...
```

Existing indexing methods remain unchanged.

- [ ] Write tests for UUID-valid search-session identities, finite `ScoredVisual.score`, and public exports/protocol shape.
- [ ] Commit `test: define visual search index contracts`; verify intended RED in Actions.
- [ ] Implement the Winston-owned types and extend `VisualIndex`, without importing Qdrant types into domain modules.
- [ ] Commit `feat: define visual search index contracts`; run index/search focused tests in Actions and require GREEN.

---

## Task 3: Read-only Qdrant opening and coarse ANN retrieval

**Files:** `src/winston/index/qdrant.py`, `tests/src/winston/index/test_qdrant_search.py`.

`open_search(identity)` must require an existing collection, validate schema v2/vector name/dimension/cosine/model/preprocessing metadata, validate stored dataset/index instance UUIDs, set the accepted embedding identity, and return a `VisualSearchSession`. A missing collection raises an actionable error saying media must be indexed first. No mutating Qdrant operation may be called.

`search_visuals()` uses Qdrant 1.18 `query_points` with:

```text
query         = dense query vector
using         = configured named vector
limit         = requested coarse limit
with_payload  = True
with_vectors  = [configured vector name]
```

- [ ] Add a fully typed fake Qdrant client and tests for missing collection, successful read-only open, every compatibility mismatch, invalid UUID metadata, and proof that no collection-creation path is used.
- [ ] Add coarse-query tests for exact `query_points` arguments, positive limit, float32 finite dimension-valid query vector, image/video payload reconstruction, named-vector extraction, finite score, malformed payload/vector rejection, and provider-error wrapping.
- [ ] Commit `test: define Qdrant semantic retrieval contracts`; verify intended RED in Actions.
- [ ] Refactor collection validation only as needed to share compatibility checks while keeping indexing's dataset-ownership validation intact.
- [ ] Implement `open_search()` and `search_visuals()`, reconstructing strict `IndexedVisual` + `ScoredVisual` values at the adapter boundary.
- [ ] Commit `feat: add read-only Qdrant semantic retrieval`; require all index adapter tests GREEN.

---

## Task 4: Bounded filtered retrieval for complete video neighborhoods

**Files:** `src/winston/index/qdrant.py`, `tests/src/winston/index/test_qdrant_search.py`.

The iterator uses Qdrant filtered scroll/pagination over:

```text
asset_id == requested asset
media_type == "video"
timestamp_us >= start_timestamp_us
timestamp_us <= end_timestamp_us
```

and requests payload + the configured named vector on every page.

- [ ] Write typed-fake tests for the exact filter, named-vector request, `page_size`, continuation offset, multi-page yielding, invalid ranges/page sizes, and wrapped provider failures.
- [ ] Commit `test: define bounded visual timeline retrieval`; verify intended RED.
- [ ] Implement `iter_visuals()` as an async iterator that yields strict `IndexedVisual` records page-by-page without accumulating the complete Qdrant result set.
- [ ] Commit `feat: add bounded visual timeline retrieval`; require focused + existing index tests GREEN.

---

## Task 5: Exact cosine, timestamp collapse, and temporal candidate windows

**Files:** `src/winston/search/temporal.py`, `tests/src/winston/search/test_temporal.py`.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class TemporalObservation:
    timestamp_us: int
    match: ScoredVisual


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    asset_id: str
    source_path: str
    start_timestamp_us: int
    end_timestamp_us: int


def cosine_similarity(query: VisualVector, candidate: VisualVector) -> float: ...
def collapse_best_by_timestamp(matches: Sequence[ScoredVisual]) -> tuple[TemporalObservation, ...]: ...
def build_temporal_windows(
    seeds: Sequence[TemporalObservation],
    *,
    context_seconds: float,
) -> tuple[TemporalWindow, ...]: ...
```

- [ ] Write cosine tests with known vectors; reject wrong dimensions, non-finite vectors, and zero-norm vectors.
- [ ] Write timestamp-collapse tests using the documentation example: at one keyframe `full=0.27`, `tile1=0.31`, `tile2=0.61`, `tile3=0.29`; expected temporal score/representative is tile2 at `0.61`.
- [ ] Write window tests using coarse timestamps `100s`, `108s`, `310s` with 15-second context; expected merged windows are `85..123s` and `295..325s`. Cover separate assets, touching windows, and clipping at zero.
- [ ] Commit `test: define semantic temporal candidate windows`; verify intended RED.
- [ ] Implement exact cosine + collapse + merging. Put the numeric examples in docstrings/comments. Explain that `temporal_context_seconds` can change grouping, whereas `candidate_limit`/`timeline_page_size` primarily bound discovery/runtime cost.
- [ ] Commit `feat: build semantic temporal candidate windows`; require temporal tests GREEN.

---

## Task 6: Moving average, z-score, Kadane, and passage selection

**Files:** `src/winston/search/temporal.py`, `tests/src/winston/search/test_temporal.py`.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class SelectedPassage:
    start_timestamp_us: int
    end_timestamp_us: int
    representative: ScoredVisual


def centered_moving_average(values: Sequence[float], *, width: int) -> tuple[float, ...]: ...
def population_z_scores(values: Sequence[float]) -> tuple[float, ...] | None: ...
def maximum_subarray(values: Sequence[float]) -> tuple[int, int] | None: ...
def select_passage(
    observations: Sequence[TemporalObservation],
    *,
    moving_average_frames: int,
) -> SelectedPassage: ...
```

- [ ] Write moving-average tests. Width 3 must demonstrate: interior = previous/current/next; first/last = only available neighboring values.
- [ ] Write population-z-score tests with deterministic expected values and zero-variance fallback.
- [ ] Write Kadane tests for normal selection, non-positive fallback, equal-sum shorter-range tie-break, and equal-length earlier-start tie-break. Use the documented example `[-0.8, -0.3, 1.2, 1.5, 0.9, -0.2, -1.0]`, selecting `1.2, 1.5, 0.9`.
- [ ] Write passage tests for fewer than three timestamps, zero variance, sustained cluster vs isolated spike, exact sampled boundaries, and representative raw region selected from inside the chosen interval.
- [ ] Commit `test: define semantic video passage selection`; verify intended RED.
- [ ] Implement the algorithms. Docstrings must state: smoothing changes temporal interval selection, z-score is local/not confidence, and Kadane only detects a contiguous numerical signal—it knows nothing about images/events.
- [ ] Commit `feat: select semantic video passages`; require temporal suite GREEN.

---

## Task 7: End-to-end search orchestration

**Files:** `src/winston/search/pipeline.py`, `src/winston/search/__init__.py`, `tests/src/winston/search/test_pipeline.py`.

**Flow:**

```text
validate query + result limit
      ↓
create embedder + visual index once
      ↓
open existing index read-only
      ↓
embed query exactly once
      ↓
Qdrant coarse retrieval
      ↓
photos: best hit per asset
videos: collapse seed timestamps -> merge windows
      ↓
for each video window sequentially:
    page all stored visuals
    exact cosine each vector
    retain best region per timestamp
    moving average -> z-score -> Kadane
      ↓
combine photo + video results
      ↓
raw_score DESC + deterministic ties
      ↓
limit
```

- [ ] Build precise fake embedder/index types and tests proving query embedded exactly once, identity passed to `open_search`, effective coarse limit `max(candidate_limit, requested_limit)`, photo deduplication, video-seed deduplication, window refinement once, and sequential iterator consumption.
- [ ] Add tests for photo/video result mapping, raw-score ranking, deterministic source/time ties, result limit after grouping, empty results, and cleanup behavior.
- [ ] Add lifecycle tests proving one embedder + one Qdrant index per command, closure on success/failure, and primary exception preservation with cleanup errors attached as notes.
- [ ] Commit `test: define semantic search orchestration`; verify intended RED.
- [ ] Implement `SearchPipeline.run()` and `run_search()` using only Winston-owned contracts outside the Qdrant adapter; follow the Phase 1E cleanup pattern.
- [ ] Commit `feat: orchestrate semantic visual search`; require search pipeline suite GREEN.

---

## Task 8: Expose `winston search` in the CLI

**Files:** `src/winston/cli.py`, `tests/src/winston/test_cli.py`.

**CLI:**

```text
winston search QUERY [--limit N]
```

Example:

```text
1. camera/parking.mkv
   passage: 00:18:39.500 -> 00:18:47.200
   representative: 00:18:42.320
   region: tile x=896 y=504 width=896 height=504 scale=0.333333
   raw score: 0.421893
```

- [ ] Write parser/dispatch tests for query, configured default limit, explicit positive `--limit`, and invalid non-positive limit.
- [ ] Write behavior tests proving whitespace-only query rejection before expensive search construction, stderr + exit 1 on failure, empty valid search exit 0, image output without passage, video output with passage/representative timestamp, region geometry, raw score, and no `%`/confidence label.
- [ ] Commit `test: define semantic search CLI behavior`; verify intended RED.
- [ ] Implement parser, dispatch, command wrapper, and formatting; reuse `_format_duration()`.
- [ ] Commit `feat: expose semantic search CLI`; require CLI + search suite GREEN.

---

## Task 9: Real Qdrant integration and permanent CI coverage

**Files:** `tests/integration/test_semantic_search_qdrant.py`, `.github/workflows/qdrant-index.yml`.

- [ ] Write real-Qdrant integration tests against `qdrant/qdrant:v1.18.2` for schema-v2 read-only opening, known-vector coarse ordering, bounded filtered timeline retrieval, image/video payload reconstruction, and local-cosine agreement with Qdrant scores within tolerance.
- [ ] Extend the permanent Qdrant workflow to run Phase 1F config/CLI/search/index-adapter tests plus the new integration test while retaining relevant Phase 1E regressions and compileall.
- [ ] Commit `test: cover semantic search with Qdrant`; verify the new integration contract in GitHub Actions. If implementation gaps make it RED, the failure must be the expected behavioral gap.
- [ ] Fix only concrete integration issues with focused regression tests first, then commit `fix:` changes and require the retained workflow GREEN.

---

## Task 10: Real Jina end-to-end validation, full regression, and draft PR

Create a dedicated temporary branch from the final feature head:

```text
test/semantic-search-validation
```

Add a temporary workflow **only on that branch** with two jobs:

```text
full-regression
real-jina-search
```

The real Jina job uses the existing `JINA_API_KEY` secret and Qdrant 1.18.2. It creates tiny red/blue image fixtures, indexes them with the Jina API engine, searches for a red-square query, and verifies the red image ranks above the blue image. It also verifies output contains a raw score and no fabricated percentage.

- [ ] Create the temporary validation branch/workflow using GitHub only.
- [ ] `full-regression`: run `uv lock --check`, locked dependency sync, complete Winston pytest regression, and `python -m compileall -q src tests`.
- [ ] `real-jina-search`: execute actual Jina text/image -> Qdrant -> Winston search using `JINA_API_KEY`.
- [ ] Inspect Actions results. If either job exposes a real bug, add a focused RED regression test on `feat/semantic-search-cli`, prove RED in Actions, implement the fix, prove GREEN, then rerun the validation branch from the updated feature head.
- [ ] Once both jobs are GREEN, delete the temporary validation branch/workflow.
- [ ] Compare `main...feat/semantic-search-cli` and check scope, `Any`/`object`, docstrings/comments, accidental Qdrant imports outside the adapter, and temporary CI files.
- [ ] Confirm retained permanent Actions GREEN on final feature head.
- [ ] Open a draft PR titled `feat: add semantic search CLI` with concise summary, exact validation evidence, and `Closes #10`.
- [ ] Leave the PR draft/unmerged for user review.

## Final Definition of Done

```text
winston search QUERY works for photos and videos
query embedding happens exactly once
existing Qdrant collection is opened read-only
coarse ANN retrieval uses raw cosine similarity
photo regions collapse to one user result per asset
video seeds create bounded/merged neighborhoods
all stored keyframe regions in each neighborhood are rescored
one best region becomes the score for each keyframe
moving average -> z-score -> Kadane selects a sparse-keyframe passage
raw representative cosine score drives final ranking
no percentage confidence is shown
Qdrant SDK types stay behind the index adapter
no schema bump/reindex is introduced
memory/network reads are bounded by pagination + sequential windows
important temporal math is documented with concrete numeric examples
all Phase 1F + relevant regression tests pass
real Qdrant 1.18.2 integration passes
real Jina end-to-end validation passes
full regression + compileall pass
temporary validation branch/workflow are deleted
draft PR contains Closes #10 and remains unmerged
```
