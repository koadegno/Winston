# Phase 1F — Semantic Search CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GitHub issue #10 by adding `winston search QUERY [--limit N]`, using Jina CLIP text embeddings plus Qdrant coarse retrieval and bounded temporal grouping inspired by the Semantic File Explorer reference algorithm.

**Architecture:** Keep `cli.py` thin. Extend Winston's `VisualIndex` boundary with read-only retrieval methods, then add a dedicated `winston.search` package. Global candidate discovery stays ANN through Qdrant; video candidates are refined only inside bounded temporal neighborhoods by rescoring all already-indexed regions, collapsing to one score per keyframe, applying a centered moving average, population z-score normalization, and Kadane maximum-subarray selection. Cross-result ranking always uses the representative raw cosine similarity, never a fabricated confidence percentage.

**Tech Stack:** Python 3.13+, argparse, asyncio, NumPy, Pydantic/Pydantic Settings, existing Jina CLIP v1 embedding engines, qdrant-client 1.18.x, Qdrant 1.18.2, pytest, pytest-asyncio, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-23-semantic-search-cli-design.md`

## Global Constraints

- Implement only GitHub issue #10 / Phase 1F. Dense intermediate-frame decoding remains Phase 2.
- The implementation stays in one draft PR titled `feat: add semantic search CLI`; PR body includes `Closes #10`.
- Do not merge or delete `feat/semantic-search-cli` without explicit user approval.
- For Winston, do not use Mac Studio / local MCP execution. Validation happens through GitHub and GitHub Actions.
- Temporary validation workflows/branches are allowed only for testing; remove them after successful validation.
- Follow strict TDD for every implementation task: failing test -> verify RED -> minimal implementation -> verify GREEN -> coherent commit.
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

## Task 1: Add search configuration and user-facing result models

**Files:**
- Modify: `src/winston/config.py`
- Create: `src/winston/search/models.py`
- Modify: `src/winston/search/__init__.py`
- Modify: `tests/src/winston/test_config.py`
- Create: `tests/src/winston/search/test_models.py`
- Delete: `tests/src/winston/search/.gitkeep`

**Interfaces:**

```python
class SearchSettings(BaseSettings):
    result_limit: PositiveInt = 10
    candidate_limit: PositiveInt = 200
    temporal_context_seconds: Annotated[float, Field(gt=0)] = 15.0
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

- [ ] **Step 1: Write failing config tests** for all defaults, nested `SEARCH__...` overrides, non-positive values, non-finite/non-positive context, and rejection of even `moving_average_frames`.
- [ ] **Step 2: Run `uv run pytest tests/src/winston/test_config.py -q` and verify RED.**
- [ ] **Step 3: Implement `SearchSettings`** using the existing nested Pydantic settings convention. Add a field validator requiring an odd moving-average width. Do not add direct environment reads.
- [ ] **Step 4: Run config tests and verify GREEN.**
- [ ] **Step 5: Write failing `SearchResult` tests** covering image timestamps all `None`, video timestamps all present, `start <= representative <= end`, finite raw score, normalized source path, and invalid mixed image/video timestamp state.
- [ ] **Step 6: Implement the strict frozen result model and `SearchRunError`.** Reuse Winston-owned `MediaType`, `RegionKind`, and `RegionGeometry` rather than duplicating validation.
- [ ] **Step 7: Run focused tests and verify GREEN.**
- [ ] **Step 8: Commit:** `feat: add semantic search configuration and result models`.

---

## Task 2: Add Winston-owned read/search contracts to the visual index

**Files:**
- Modify: `src/winston/index/base.py`
- Modify: `src/winston/index/models.py`
- Modify: `src/winston/index/__init__.py`
- Create: `tests/src/winston/index/test_qdrant_search.py` with only contract/model tests in this task

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

- [ ] **Step 1: Write failing tests** for finite `ScoredVisual.score` and exported read-session/search types.
- [ ] **Step 2: Run focused tests and verify RED.**
- [ ] **Step 3: Add the Winston-owned types and extend `VisualIndex`.** Validate both session UUID strings and finite scores without importing Qdrant types into the domain layer.
- [ ] **Step 4: Run focused tests and existing index tests; verify GREEN.**
- [ ] **Step 5: Commit:** `feat: define visual search index contracts`.

---

## Task 3: Implement read-only Qdrant opening and coarse ANN retrieval

**Files:**
- Modify: `src/winston/index/qdrant.py`
- Extend: `tests/src/winston/index/test_qdrant_search.py`

**Behavior:**

`open_search(identity)` must require an existing collection, validate schema v2/vector name/dimension/cosine/model/preprocessing metadata, validate stored dataset/index instance UUIDs, set the accepted embedding identity, and return a `VisualSearchSession`. A missing collection must raise an actionable Winston error mentioning that media must be indexed first. No mutation method may be called.

`search_visuals(query_vector, limit=...)` uses Qdrant 1.18 `query_points` with:

```text
query         = dense query vector
using         = configured named vector
limit         = requested coarse limit
with_payload  = True
with_vectors  = [configured vector name]
```

- [ ] **Step 1: Add a fully typed fake Qdrant client and failing tests** for missing collection, successful schema-v2 read-only open, every compatibility mismatch, invalid UUID metadata, and proof that search opening never calls collection creation.
- [ ] **Step 2: Run the new Qdrant search tests and verify RED.**
- [ ] **Step 3: Refactor collection validation only as much as needed** so indexing still validates dataset ownership while search validates stored ownership without a caller-provided dataset ID.
- [ ] **Step 4: Implement `open_search()` and verify GREEN.**
- [ ] **Step 5: Add failing coarse-query tests** checking exact `query_points` arguments, positive limit validation, float32 finite query-vector/dimension validation, image/video payload reconstruction, named-vector extraction, finite score validation, missing/malformed payload or vector rejection, and provider-error wrapping.
- [ ] **Step 6: Implement Qdrant -> Winston mapping** by reconstructing strict `IndexedVisual` values from payload + stored vector + the already accepted `EmbeddingIdentity`, then wrap them as `ScoredVisual`.
- [ ] **Step 7: Run all index adapter tests and verify GREEN.**
- [ ] **Step 8: Commit:** `feat: add read-only Qdrant semantic retrieval`.

---

## Task 4: Implement bounded filtered retrieval for complete video neighborhoods

**Files:**
- Modify: `src/winston/index/qdrant.py`
- Extend: `tests/src/winston/index/test_qdrant_search.py`

The iterator uses Qdrant filtered scroll/pagination over:

```text
asset_id == requested asset
media_type == "video"
timestamp_us >= start_timestamp_us
timestamp_us <= end_timestamp_us
```

with payload + the configured named vector requested on every page.

- [ ] **Step 1: Add failing typed-fake tests** proving the exact asset/media/time filter, configured named vector retrieval, page-size bound, continuation-offset propagation, multi-page yielding, invalid range/page-size rejection, and wrapped provider failures.
- [ ] **Step 2: Run focused tests and verify RED.**
- [ ] **Step 3: Implement `iter_visuals()` as an async iterator.** Yield strict `IndexedVisual` records page-by-page; do not accumulate every Qdrant record before yielding.
- [ ] **Step 4: Run focused + existing index tests and verify GREEN.**
- [ ] **Step 5: Commit:** `feat: add bounded visual timeline retrieval`.

---

## Task 5: Implement temporal seed collapse, candidate windows, and exact cosine scoring

**Files:**
- Create: `src/winston/search/temporal.py`
- Create/extend: `tests/src/winston/search/test_temporal.py`

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

- [ ] **Step 1: Write failing cosine tests** using known vectors; reject wrong dimensions, non-finite values, and zero-norm vectors.
- [ ] **Step 2: Verify RED, implement exact cosine, verify GREEN.**
- [ ] **Step 3: Write failing timestamp-collapse tests** with the concrete documentation example `full=0.27`, `tile1=0.31`, `tile2=0.61`, `tile3=0.29` at the same timestamp; expected score/representative is tile2 at `0.61`.
- [ ] **Step 4: Write failing window tests** using coarse timestamps `100s`, `108s`, `310s` with 15-second context; expected merged windows are `85..123s` and `295..325s`. Also cover different assets, touching windows, and clipping below zero.
- [ ] **Step 5: Implement collapse/window logic with those examples in docstrings/comments.** Explain that `temporal_context_seconds` can change semantic grouping, while `candidate_limit` and `timeline_page_size` do not define the math of one already-selected neighborhood.
- [ ] **Step 6: Run temporal tests and verify GREEN.**
- [ ] **Step 7: Commit:** `feat: build semantic temporal candidate windows`.

---

## Task 6: Implement moving average, z-score, Kadane, and passage selection

**Files:**
- Modify: `src/winston/search/temporal.py`
- Extend: `tests/src/winston/search/test_temporal.py`

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

- [ ] **Step 1: Write failing centered-moving-average tests.** The docstring/example must show width 3 explicitly: at an interior point use previous/current/next score; at the first/last point use only available values.
- [ ] **Step 2: Verify RED, implement moving average, verify GREEN.** State in the docstring that smoothing changes interval selection but never replaces the raw cosine score shown/ranked to users.
- [ ] **Step 3: Write failing population-z-score tests** with deterministic values and zero-variance fallback. Document that z-score compares a keyframe to its local neighborhood and is not a calibrated confidence.
- [ ] **Step 4: Verify RED, implement z-score, verify GREEN.**
- [ ] **Step 5: Write failing Kadane tests** for normal selection, all-non-positive fallback, equal-sum shorter-interval tie-break, and equal-length earlier-start tie-break. Include a simple numeric example such as `[-0.8, -0.3, 1.2, 1.5, 0.9, -0.2, -1.0]`, whose strongest positive contiguous interval is `1.2, 1.5, 0.9`.
- [ ] **Step 6: Verify RED, implement Kadane, verify GREEN.** Explain in code that Kadane only finds a contiguous numerical signal; it has no knowledge of images/events.
- [ ] **Step 7: Add failing passage-selection tests** for fewer than three timestamps, zero variance, sustained cluster vs isolated spike, exact selected keyframe boundaries, and representative raw region chosen from the selected interval.
- [ ] **Step 8: Implement `select_passage()` and verify GREEN.**
- [ ] **Step 9: Commit:** `feat: select semantic video passages`.

---

## Task 7: Implement the end-to-end search orchestration

**Files:**
- Create: `src/winston/search/pipeline.py`
- Modify: `src/winston/search/__init__.py`
- Create/extend: `tests/src/winston/search/test_pipeline.py`

**Flow:**

```text
validate query + result limit
      ↓
create/open embedder + visual index once
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
raw_score DESC + deterministic tie-breaks
      ↓
limit
```

- [ ] **Step 1: Build precise fake embedder/index types and write failing orchestration tests** proving query embedded once, identity passed to `open_search`, effective coarse limit `max(candidate_limit, requested_limit)`, photo deduplication, video-seed deduplication, window refinement exactly once, and sequential iterator consumption.
- [ ] **Step 2: Run pipeline tests and verify RED.**
- [ ] **Step 3: Implement `SearchPipeline.run()` minimally and verify GREEN.** Keep Qdrant types out of this module.
- [ ] **Step 4: Add failing ranking/result-mapping tests** for photo/video combination, raw-score ranking, deterministic source/time tie-breaks, final limit after grouping, and empty results.
- [ ] **Step 5: Implement mapping/ranking and verify GREEN.**
- [ ] **Step 6: Add failing lifecycle tests** for `run_search()` constructing one embedder + one Qdrant index, closing both on success/failure, and preserving a primary failure while attaching cleanup failures as exception notes.
- [ ] **Step 7: Implement resource lifecycle following the Phase 1E cleanup pattern and verify GREEN.**
- [ ] **Step 8: Commit:** `feat: orchestrate semantic visual search`.

---

## Task 8: Expose `winston search` in the CLI

**Files:**
- Modify: `src/winston/cli.py`
- Modify: `tests/src/winston/test_cli.py`

**CLI:**

```text
winston search QUERY [--limit N]
```

Example result:

```text
1. camera/parking.mkv
   passage: 00:18:39.500 -> 00:18:47.200
   representative: 00:18:42.320
   region: tile x=896 y=504 width=896 height=504 scale=0.333333
   raw score: 0.421893
```

- [ ] **Step 1: Write failing parser/dispatch tests** for query, configured default limit, explicit positive `--limit`, and invalid non-positive limit.
- [ ] **Step 2: Verify RED, implement parser/dispatch, verify GREEN.**
- [ ] **Step 3: Write failing behavior tests** proving whitespace-only queries fail before expensive search construction, errors go to stderr with exit code 1, empty valid search exits 0, image output omits passage lines, video output includes passage/representative timestamps, region geometry is printed, raw score is printed with no `%` or confidence label.
- [ ] **Step 4: Implement formatting/command wrapper and verify GREEN.** Reuse `_format_duration()`.
- [ ] **Step 5: Commit:** `feat: expose semantic search CLI`.

---

## Task 9: Add real Qdrant semantic-search integration and permanent CI coverage

**Files:**
- Create: `tests/integration/test_semantic_search_qdrant.py`
- Modify: `.github/workflows/qdrant-index.yml`

- [ ] **Step 1: Write real-Qdrant integration tests** against `qdrant/qdrant:v1.18.2` for read-only opening of schema-v2 state, known-vector coarse ordering, bounded filtered timeline retrieval, image/video payload reconstruction, and local cosine agreement with Qdrant scores within floating-point tolerance.
- [ ] **Step 2: Extend the permanent Qdrant workflow** to run Phase 1F config/CLI/search/index adapter tests plus the new integration test while retaining relevant Phase 1E regressions and compileall.
- [ ] **Step 3: Push the RED integration/CI commit only if needed to observe the intended failure in GitHub Actions; otherwise preserve the normal local-contract RED/GREEN sequence through branch commits.** All execution remains on GitHub Actions, not Mac Studio.
- [ ] **Step 4: Implement/fix only issues exposed by the real server, then require the retained workflow to be GREEN.**
- [ ] **Step 5: Commit:** `test: cover semantic search with Qdrant`.

---

## Task 10: Run real Jina end-to-end validation, full regression, and open the draft PR

**Temporary validation:** create a dedicated branch from the final feature head, for example:

```text
test/semantic-search-validation
```

Add a temporary workflow only on that branch with two jobs:

```text
full-regression
real-jina-search
```

The real Jina job uses the existing `JINA_API_KEY` secret and Qdrant 1.18.2. It generates tiny red/blue image fixtures, indexes them with the Jina API engine, then searches for a red-square query and verifies the red image ranks above the blue image. It must also verify output contains a raw score and no fabricated percentage.

- [ ] **Step 1: Create the dedicated temporary validation branch/workflow using GitHub only.**
- [ ] **Step 2: `full-regression` runs at least `uv lock --check`, locked dependency sync, complete Winston pytest regression, and `python -m compileall -q src tests`.**
- [ ] **Step 3: `real-jina-search` performs an actual Jina text/image -> Qdrant -> Winston search path using `JINA_API_KEY`.**
- [ ] **Step 4: Inspect logs/results; fix implementation on `feat/semantic-search-cli` with TDD if either validation exposes a real bug, then rerun validation from the updated head.**
- [ ] **Step 5: Once both jobs are GREEN, delete the temporary workflow/branch. Confirm it is absent from the feature branch.**
- [ ] **Step 6: Review `main...feat/semantic-search-cli` for scope creep, `Any`/`object` additions, missing docstrings/comments, accidental Qdrant imports outside the adapter, and temporary CI files.**
- [ ] **Step 7: Confirm the retained permanent GitHub Actions are GREEN on the final feature head.**
- [ ] **Step 8: Open a draft PR titled `feat: add semantic search CLI` with a concise summary, exact validation evidence, and `Closes #10`.**
- [ ] **Step 9: Leave the PR draft/unmerged for user review.**

## Final Definition of Done

Phase 1F is ready for user review only when all of the following are true:

```text
winston search QUERY works for photos and videos
query embedding happens exactly once
existing Qdrant collection is opened read-only
coarse ANN retrieval uses raw cosine similarity
photo regions collapse to one user result per asset
video seeds create bounded/merged neighborhoods
all stored keyframe regions in a neighborhood are rescored
one best region becomes the score for each keyframe
moving average -> z-score -> Kadane selects a sparse-keyframe passage
raw representative cosine score drives final ranking
no percentage confidence is shown
Qdrant SDK types stay behind the index adapter
no schema bump/reindex is introduced
memory/network reads are bounded by pagination + sequential windows
important temporal math is documented with concrete examples
all Phase 1F + relevant regression tests pass
real Qdrant 1.18.2 integration passes
real Jina end-to-end validation passes
full regression + compileall pass
temporary validation branch/workflow are deleted
draft PR contains Closes #10 and remains unmerged
```
