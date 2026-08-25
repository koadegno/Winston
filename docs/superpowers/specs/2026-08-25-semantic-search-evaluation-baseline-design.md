# Phase 1G — Semantic-search evaluation baseline design

## Status

Approved design direction for GitHub issue #11: **Phase 1G — Establish semantic-search evaluation baseline**.

This design also establishes the measurement foundation for issue #32, **Investigate semantic search quality: multilingual queries, attribute binding, false positives, and reranking**. Phase 1G does not implement the experiments proposed in #32. It creates the fixed benchmark, contracts, metrics, and reproducible execution path that those later experiments must use.

The guiding rule is:

```text
measure the current Phase 1F system first
        ↓
freeze a reproducible baseline
        ↓
change one hypothesis at a time
        ↓
compare against the same benchmark
```

The baseline must not tune itself to the results it is measuring.

## Goals

Phase 1G must:

- create a versioned semantic-search benchmark manifest backed by fixed real Winston media;
- pin the benchmark media to an immutable release asset digest rather than commit footage to Git;
- represent positive events, known hard negatives, query language, query family, and cross-language/synonym equivalence groups explicitly;
- evaluate the existing Phase 1F search behavior without changing its ranking, temporal selection, crops, embedding model, or score semantics;
- compute deterministic per-query and aggregate semantic-quality metrics;
- separate semantic retrieval quality from temporal localization quality;
- measure indexing throughput, vectors per hour of video, end-to-end query latency, and Qdrant ANN latency separately;
- preserve enough run provenance to compare later models, region strategies, query processing, rerankers, and no-match policies;
- provide a CLI that emits machine-readable JSON suitable for GitHub Actions artifacts and later comparison;
- keep normal CI fast and provider-free by testing evaluation logic with deterministic fakes/fixtures;
- provide an explicit GitHub Actions path for the real baseline using the pinned release, Qdrant, and the configured Jina provider;
- remain class-agnostic: benchmark labels describe what is visible, but they do not become detector classes or indexing requirements.

## Non-goals

Phase 1G does not:

- switch from Jina CLIP v1 to v2 or another embedding model;
- translate French queries, expand synonyms, or fuse multiple query embeddings;
- change fixed crops/tiles or add region proposals;
- add a VLM/reranker;
- add a no-match threshold or convert cosine into a confidence percentage;
- change ANN candidate limits, temporal grouping, local cosine scoring, moving average, z-score, or Kadane behavior to improve benchmark numbers;
- decode additional P-frames or implement Phase 2 temporal refinement;
- introduce MLflow, notebooks as the canonical benchmark, a database of experiment runs, or a dashboard;
- commit the 844 MB media release to the repository;
- infer ground truth from the current search ranking.

A benchmark that is labeled by accepting the current model's own top results as truth would be circular and is explicitly forbidden.

## Why a dedicated evaluation subsystem

A one-off script would be enough to print a few Recall@K values, but issue #32 already requires repeated comparisons across language handling, crop strategies, embedding models, compositional scoring, reranking, and no-match behavior. The benchmark therefore needs stable typed boundaries without becoming a full experiment-management platform.

The chosen architecture is:

```text
benchmarks/semantic-search/v1/manifest.json
        │
        ├─ dataset provenance
        ├─ fixed queries
        ├─ positive truth events
        ├─ annotated hard negatives
        └─ metric cutoffs
        │
        v
src/winston/evaluation/
├── models.py
├── manifest.py
├── relevance.py
├── metrics.py
├── instrumentation.py
└── runner.py
        │
        v
winston evaluate semantic ...
        │
        v
machine-readable baseline JSON
```

The core search/indexing subsystems remain the system under test. Evaluation code consumes their public Winston-owned contracts and does not move benchmark-specific rules into `src/winston/search` or `src/winston/indexing`.

## Fixed dataset provenance

The first benchmark version uses the existing public release:

```text
release tag:   test-dataset-v1
release name:  Winston test dataset v1
asset:         videos.zip
size:          844175165 bytes
sha256:        94fa6b655e4118401fb724736d5e21f5e34d52b677f91562af9297e4a08201b6
```

The manifest records the release tag, asset name, byte size, SHA-256 digest, and the exact source paths it expects after extraction. A real benchmark run must fail before indexing if the downloaded asset digest or expected media inventory does not match the manifest.

This provides two independent identities:

```text
benchmark manifest version
        +
media release digest
```

Changing annotations/queries creates a new benchmark manifest version. Changing media creates a new pinned dataset identity. Historical benchmark results therefore remain interpretable.

## Benchmark manifest

Canonical path:

```text
benchmarks/semantic-search/v1/manifest.json
```

The JSON is data, not executable configuration. It is parsed into strict Winston-owned models with unknown fields rejected.

Conceptual structure:

```json
{
  "schema_version": 1,
  "benchmark_id": "winston-semantic-v1",
  "dataset": {
    "release_tag": "test-dataset-v1",
    "asset_name": "videos.zip",
    "asset_size_bytes": 844175165,
    "sha256": "94fa6b655e4118401fb724736d5e21f5e34d52b677f91562af9297e4a08201b6",
    "expected_source_paths": ["..."]
  },
  "cutoffs": [1, 3, 5, 10],
  "queries": [
    {
      "id": "...",
      "text": "red bicycle",
      "language": "en",
      "family": "attribute_binding",
      "equivalence_group": "red-bicycle",
      "expect_no_match": false,
      "positives": ["event-red-bicycle-001"],
      "hard_negatives": ["negative-red-person-black-bike-001"]
    }
  ],
  "events": ["... typed event objects ..."]
}
```

The actual schema uses typed objects rather than stringly-typed arbitrary dictionaries.

### Query identity

Each query has:

- stable `id`;
- exact query `text` sent to Phase 1F unchanged;
- explicit `language`, initially `en` or `fr`;
- one `family`;
- optional `equivalence_group` for translations/synonyms that express the same intent;
- `expect_no_match`;
- referenced positive event IDs;
- referenced hard-negative event IDs.

The query text is benchmark data. The runner must not translate, rewrite, normalize beyond the existing search pipeline's normal whitespace handling, or generate extra prompts.

### Query families

The initial controlled vocabulary is:

```text
simple_object
person_object
object_attribute
attribute_binding
person_clothing
person_object_relation
confusable_object
synonym
absent
```

Language is stored independently rather than encoded into the family, so the same family can be compared between French and English.

These categories are evaluation metadata only. Winston indexing/search never receives them.

### Equivalence groups

Queries that should mean the same thing share an `equivalence_group`, for example:

```text
stroller / pram / pushchair
une poussette / a stroller
une personne avec un t-shirt blanc / a person with a white T-shirt
```

Equivalence groups permit paired language/synonym reporting without merging their actual retrieval runs. Each text query is still executed independently.

## Ground-truth event model

Ground truth describes real visible events in the pinned release. It must be human-verified independently of Winston's rank ordering. Search output may help navigate footage during annotation, but it is evidence to inspect, not authority for the label.

A truth event contains:

```text
id
source_path
media_type
start_timestamp_seconds
end_timestamp_seconds
optional region annotation
annotation note
```

For video, timestamps are finite, non-negative, and ordered. For still images, timestamps are absent and source identity is sufficient.

An optional manually verified region can document where the relevant object is visible, but Phase 1G semantic relevance does **not** require tile/box IoU. Current Phase 1F regions are intentionally coarse deterministic crops; making exact region overlap part of the first pass would conflate crop localization with whether Winston found the correct visual event.

Region-specific evaluation can be added as a later benchmark version without changing Phase 1G's semantic definition.

### Positive events

A positive event means the query is genuinely satisfied in that media interval.

Examples include the interval in which a cyclist is visible or the interval in which a person is actually wearing the requested clothing color.

### Hard negatives

A hard negative is an explicitly annotated interval that is visually/semantically tempting but does **not** satisfy the query. It records a reason from a controlled vocabulary such as:

```text
wrong_attribute_binding
wrong_color
wrong_object
confusable_object
wrong_relation
other
```

Examples from issue #32 include:

```text
query: red bicycle
hard negative: red clothing/person next to a non-red bicycle

query: person wearing a yellow T-shirt
hard negative: person carrying a yellow bag while the shirt is not yellow

query: stroller
hard negative: bicycle/cyclist scene
```

Hard negatives make failures diagnosable instead of treating every wrong result as one undifferentiated false positive.

### Absent queries

An absent query has:

```text
expect_no_match = true
positives = []
```

It can still reference known hard negatives. A manifest entry may not simultaneously declare `expect_no_match=true` and positive events.

Phase 1F currently returns best-available neighbors rather than abstaining, so poor absent-query behavior is an expected baseline result, not something Phase 1G should hide with a threshold.

## Semantic relevance rule

Phase 1G deliberately evaluates the representative visual selected by Phase 1F rather than allowing a very wide returned passage to count as correct merely because it overlaps a truth interval somewhere.

A video result semantically matches a positive truth event when:

```text
result.source_path == truth.source_path
AND
truth.start <= result.representative_timestamp <= truth.end
```

A still-image result matches when its source path equals the truth source path.

This rule isolates the question:

> Did semantic retrieval select a visual moment that actually satisfies the query?

Returned passage boundaries are evaluated separately.

### Deterministic one-to-one matching

Within top K, ranked results are matched to positive truth events in order. Each positive truth event can be claimed at most once.

If a result can match multiple still-unmatched truth events, the runner selects deterministically by:

1. smallest distance between representative timestamp and truth-interval center;
2. then stable truth-event ID.

A second result that points to an already-claimed positive event does not create extra recall and is treated as an unmatched result for precision. This prevents duplicate result windows from manufacturing benchmark quality.

Hard-negative attribution is evaluated after positive matching. An unmatched result whose representative lands in an annotated hard-negative interval is labeled with that hard-negative reason; otherwise it is a generic false positive.

## Semantic quality metrics

Metrics are computed per query first, then aggregated. The raw per-query records remain in the result JSON so later analysis never depends only on a single headline number.

### Recall@K

For positive queries:

```text
unique positive truth events matched in top K
---------------------------------------------
        total positive truth events
```

Absent queries have no Recall@K denominator and are excluded from positive-query macro recall.

### Precision@K

For positive queries:

```text
one-to-one positive matches in top K
------------------------------------
                   K
```

If the engine returns fewer than K results, unfilled ranks count as non-relevant for this metric. This prevents a future implementation from improving Precision@K simply by returning almost nothing. Abstention/no-match behavior is reported separately.

### First relevant rank and reciprocal rank

For each positive query:

- `first_relevant_rank`: 1-based rank of the first one-to-one positive match, or null if absent;
- reciprocal rank: `1 / first_relevant_rank`, or zero if no positive is found.

Aggregate MRR is reported across positive queries.

### Success@K

A positive query succeeds at K if at least one positive event is found in top K.

This becomes the basis for interpretable family-specific rates such as:

```text
attribute_binding_accuracy@K
confusable_object_accuracy@K
```

These names mean macro `Success@K` across the corresponding benchmark family; they are not probabilities and are not model confidence values.

### Hard-negative rate@K

For each query/family, report the fraction of inspected top-K ranks attributed to annotated hard negatives, plus counts by hard-negative reason.

This distinguishes failures such as wrong attribute binding from unrelated nearest-neighbor noise.

### Language and synonym consistency

For each `equivalence_group`, preserve the per-query metrics and report pair/group gaps, including:

- difference in Recall@K;
- difference in first relevant rank when both queries retrieve a positive;
- whether one formulation succeeds at K while an equivalent formulation fails.

The benchmark does not average query embeddings or fuse equivalent queries. Consistency reporting observes the current model as-is.

### No-match behavior

For `expect_no_match=true` queries, report:

- returned result count;
- whether the engine abstained completely;
- top raw cosine score when present;
- top-K raw score distribution;
- annotated hard-negative hits.

Aggregate outputs include `no_match_abstention_rate` and absent-query score summaries.

Phase 1G does not decide a threshold from these numbers. They provide calibration evidence for later #32 experiments.

## Temporal localization metrics

Semantic correctness and temporal precision are separate dimensions.

For each semantically matched video result, compare its returned passage to the matched truth interval and report:

- start error in seconds;
- end error in seconds;
- passage duration;
- truth duration;
- temporal intersection-over-union (tIoU).

A semantically correct representative can therefore receive full semantic credit while still exposing an overly broad Phase 1F passage. Conversely, passage overlap alone cannot rescue a semantically wrong representative.

This prevents Phase 2 localization work from being confused with embedding/retrieval quality.

## Performance measurements

Performance measurements are reported independently from semantic metrics.

### Instrumentation boundary

Phase 1G must not add benchmark-specific methods to `VisualIndex` or change search/indexing semantics merely to obtain timings.

Instead, `evaluation.instrumentation` provides a typed delegating `VisualIndex` wrapper used only by the benchmark runner. It implements the same Winston `VisualIndex` protocol and records:

- number of visuals passed through `upsert()` during the measured indexing run;
- duration of ANN `search_visuals()` calls;
- optional total duration spent streaming refinement windows through `iter_visuals()`.

All actual storage behavior remains delegated to the real index implementation.

This keeps Qdrant SDK types inside the existing Qdrant adapter and keeps evaluation provider-agnostic.

### Indexing throughput

A full baseline run uses an empty/dedicated evaluation Qdrant instance/collection and a freshly extracted pinned dataset.

The runner records wall-clock duration around `IndexingPipeline.run()` and independently computes total video duration by probing the fixed dataset.

Report at least:

```text
indexing_wall_seconds
video_footage_seconds
footage_hours_per_wall_hour
upserted_visual_count
vectors_per_footage_hour
indexed_assets
failed_assets
skipped_assets
```

A canonical cold baseline run is invalid if any asset was skipped or failed. This prevents restart/idempotency behavior from being mistaken for fresh indexing throughput.

Still images contribute vectors and indexing work but not video-footage hours. Their counts remain explicit in run provenance.

### Search latency

For every benchmark query, report separately:

```text
end_to_end_search_ms
qdrant_ann_ms
refinement_scroll_ms (when video refinement occurs)
```

Aggregate latency summaries include count, mean, median/p50, and p95. Raw samples are retained in the result JSON.

`qdrant_ann_ms` measures the actual `search_visuals()` call(s), including bounded ANN overfetch when Phase 1F performs it. It does not include text embedding, local exact cosine, or temporal passage selection. End-to-end latency does include the complete current search pipeline.

No concurrent query load generator is added in Phase 1G; measurements are single-query sequential baseline measurements.

## Runner and CLI

The user-facing command is:

```text
winston evaluate semantic <manifest> --dataset-root <path> --output <results.json>
```

The runner has two explicit execution modes:

```text
--mode search
--mode full
```

### `--mode search`

- validates the benchmark manifest and local media inventory/digests;
- opens the existing compatible configured visual index read-only;
- runs every query through the current `SearchPipeline`;
- emits semantic, temporal, and search-latency metrics;
- does not modify/rebuild the index.

This mode is useful for repeated local comparisons against an already prepared index.

### `--mode full`

- validates the manifest and media first;
- runs the existing `IndexingPipeline` against the configured dedicated evaluation collection;
- requires a cold run (`skipped == 0`, `failed == 0`);
- records indexing metrics through the instrumentation wrapper;
- then executes the same search evaluation;
- emits one combined result document.

The CLI never silently deletes or recreates a user's collection. The canonical full baseline GitHub Action supplies a fresh Qdrant service/collection. A local full run is the caller's responsibility to point at a dedicated empty evaluation target; if cold-run invariants are violated, the benchmark fails rather than publishing misleading throughput.

CLI fatal validation/runtime failures go to stderr and return non-zero. Machine-readable benchmark output goes only to the requested JSON path, avoiding progress logs mixed into the artifact.

## Result document and reproducibility

The result JSON has its own schema version and contains at least:

```text
result_schema_version
benchmark_id
manifest_sha256
dataset release/digest provenance
run timestamp UTC
Winston revision when available
embedding identity
search settings snapshot
index/Qdrant settings relevant to semantics/performance
per-query ranked judgments
per-query metrics
aggregate semantic metrics
family metrics
equivalence-group/language metrics
temporal localization metrics
performance samples and summaries
```

Secrets, API keys, raw vectors, and raw media are never written into the result document.

The exact manifest SHA-256 is recorded so two result artifacts can immediately establish whether they used identical benchmark definitions.

Search settings that affect semantics or resource bounds are captured because a comparison between two runs is not meaningful if, for example, candidate limits or temporal context silently changed.

## Initial benchmark content

The first version should deliberately cover the failure modes already observed during Phase 1F manual testing, not only easy positive queries.

Candidate query intents include:

```text
simple object:
  cyclist
  dog
  bag

confusable objects / synonyms:
  stroller
  pram
  pushchair
  bicycle

attribute binding:
  red bicycle
  person wearing a yellow T-shirt
  person wearing a white T-shirt

person/object or clothing:
  man with a cap
  man with a hoodie
  person with a bag

French/English equivalents:
  un cycliste / a cyclist
  une poussette / a stroller
  une personne avec un t-shirt blanc / a person with a white T-shirt

absent queries:
  only concepts that have been manually verified absent from the fixed release
```

The exact v1 list is constrained by verified footage. If a proposed concept cannot be confidently annotated from the pinned release, it is excluded rather than guessed.

Issue #11's original examples (`pink stroller`, `green cap`, `cardboard box`, `red bicycle`, `umbrella`, `blue car`) remain desired benchmark concepts, but they may only enter v1 when the release contains a human-verified positive or when intentionally declared as a verified absent query. The issue text does not itself prove that the current release contains them.

## Annotation workflow

Ground-truth creation is a data-labeling task, not a model-output conversion.

For each candidate intent:

1. inspect the fixed release footage independently enough to identify true positive intervals;
2. record each verified positive event with source path and interval;
3. record particularly misleading but wrong intervals as hard negatives with a reason;
4. verify absent queries across the fixed benchmark scope before labeling them absent;
5. only then freeze the manifest.

Current Phase 1F search can be used to jump to candidate timestamps and reduce review effort, but an ANN hit does not become truth until visually verified. Known manual observations from prior testing can seed where to inspect, but the benchmark must record the verified event, not the model's claim.

A later benchmark version can add new labels/media without rewriting v1 history.

## GitHub Actions

Phase 1G adds two different automation paths.

### Permanent contract workflow

Runs on ordinary PR/push changes affecting evaluation code. It uses deterministic fake embedders/indexes and small repository fixtures only.

It verifies:

- strict manifest validation;
- one-to-one relevance matching;
- Recall/Precision/rank/MRR calculations;
- hard-negative attribution;
- absent-query handling;
- language/equivalence aggregation;
- temporal metrics;
- instrumentation accounting/timing plumbing;
- CLI JSON output and failure behavior;
- source compilation.

No Jina secret, 844 MB download, or real embedding inference is required for this workflow.

### Real baseline workflow

A manual `workflow_dispatch` workflow runs the actual benchmark:

```text
checkout chosen revision
        ↓
download videos.zip from test-dataset-v1
        ↓
verify size + SHA-256
        ↓
extract expected media
        ↓
start fresh Qdrant
        ↓
run full semantic benchmark with configured Jina engine
        ↓
upload results.json as workflow artifact
```

The workflow uses the repository secret for Jina credentials and fails clearly when the secret is unavailable. It does not run on every push because the media download and embedding calls are intentionally expensive.

The artifact is evidence for the baseline; it is not automatically committed back to the repository.

## Error handling and validity rules

A benchmark run fails rather than silently degrading when:

- manifest schema/version is unsupported;
- query IDs or event IDs are duplicated;
- a referenced positive/hard-negative event does not exist;
- a query is both `expect_no_match=true` and has positive events;
- timestamps are invalid/non-finite/out of order;
- an event references a source path outside the pinned media inventory;
- release asset size/digest differs;
- extracted expected paths differ from the manifest;
- the configured index embedding identity is incompatible;
- a full run skips or fails assets;
- evaluation receives malformed/non-finite search results;
- output path cannot be written.

A query returning zero results is a valid measurement, not a runner failure.

Metric denominators, missing values, and absent-query exclusions must be explicit in the JSON rather than represented by NaN or Infinity. JSON output contains only finite numeric values or null where a metric is mathematically not applicable.

## Testing strategy

Development follows RED -> GREEN TDD.

### Manifest/model tests

Cover:

- valid v1 manifest;
- strict unknown-field rejection;
- duplicate IDs;
- missing event references;
- malformed release SHA-256;
- invalid timestamps;
- invalid/duplicate cutoffs;
- contradictory absent-query definitions;
- unsupported family/language/hard-negative reason;
- media inventory mismatch.

### Relevance tests

Cover:

- representative timestamp inside/outside truth interval;
- same timestamp on wrong asset;
- deterministic one-to-one matching;
- duplicate returned windows do not inflate recall/precision;
- hard-negative attribution after positive matching;
- generic false positives;
- still-image source matching.

### Metric tests

Use tiny hand-computable ranked lists to prove exact values for:

- Recall@1/@3/@K;
- Precision@K;
- first relevant rank;
- MRR;
- Success@K;
- family attribute-binding/confusable-object accuracy;
- hard-negative rates/reasons;
- absent-query summaries;
- equivalence-group language gaps;
- temporal start/end errors and tIoU;
- p50/p95 latency summary behavior.

### Instrumentation/runner tests

Use typed fakes to prove:

- wrapped `upsert()` counts visuals without changing delegated behavior;
- ANN latency samples correspond to actual `search_visuals()` calls, including overfetch;
- full mode rejects skipped/failed indexing runs;
- search mode never calls mutating index operations;
- each manifest query is executed exactly once as written;
- settings/embedding/manifest provenance is emitted;
- raw cosine remains raw cosine and is never relabeled as confidence.

### Real integration

The existing Qdrant integration workflow receives focused coverage for evaluation components that interact with the real adapter where useful. The expensive public-release/Jina baseline remains the manual real-baseline workflow.

## Compatibility with issue #32

Phase 1G intentionally makes later experiments replaceable at the boundary around the system under test.

After the baseline is frozen, #32 experiments can run against the same manifest and produce comparable result JSON for:

```text
current Jina CLIP v1
FR -> EN normalization
query expansion/synonyms
alternative crop strategies
Jina CLIP v2
other multimodal embedding models
query decomposition
VLM/reranker verification
calibrated no-match policies
```

Experiments that require a different visual index must reindex but keep the benchmark manifest fixed. Experiments that only alter query handling can reuse a compatible visual index. Every result artifact records embedding/search settings so incompatible runs are visible rather than silently compared as if identical.

## Branch and PR strategy

Phase 1G is implemented in one focused branch/PR:

```text
feat/search-evaluation
```

The branch starts from the validated Phase 1F head because evaluation depends on the new semantic-search contracts. While PR #28 is unmerged, the Phase 1G PR is stacked against Phase 1F. Once Phase 1F lands on `main`, Phase 1G is retargeted/rebased as appropriate without changing its benchmark contract.

The Phase 1G PR description includes:

```text
Closes #11
```

Issue #32 remains open after Phase 1G: Phase 1G supplies its benchmark foundation but does not complete the broader semantic-quality investigation.

## Acceptance criteria

Phase 1G is complete when:

1. a strict versioned benchmark manifest is committed and pins the exact dataset release/digest;
2. the v1 query set contains independently verified positives/hard negatives/absent cases sufficient to exercise several semantic families without predefined classes;
3. normal CI proves manifest, relevance, metrics, instrumentation, runner, and CLI contracts without external providers;
4. the real baseline workflow can reconstruct the pinned dataset, use fresh Qdrant, run the current Jina CLIP v1 indexing/search pipeline, and upload a result artifact;
5. result JSON reports semantic Recall@K, Precision@K, first relevant rank/MRR, family Success@K, hard-negative behavior, language/synonym consistency, and no-match behavior;
6. temporal localization metrics are reported separately from semantic correctness;
7. indexing throughput, vectors per footage hour, end-to-end search latency, and Qdrant ANN latency are reported separately;
8. no metric is called confidence and no uncalibrated threshold is introduced;
9. benchmark ground truth is not derived automatically from current model rankings;
10. the produced baseline artifact is sufficient to compare later #32 experiments without changing the benchmark definition.
