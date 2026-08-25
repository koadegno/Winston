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
- pin benchmark media to immutable archive and extracted-file digests rather than commit footage to Git;
- represent positive events, known hard negatives, query language, semantic family, and cross-language/synonym equivalence explicitly;
- evaluate existing Phase 1F behavior without changing ranking, temporal selection, crops, embedding model, or score semantics;
- compute deterministic per-query and aggregate semantic-quality metrics;
- separate semantic retrieval quality from temporal localization quality;
- measure indexing throughput, video vectors per hour of footage, end-to-end query latency, and Qdrant ANN latency separately;
- preserve enough run provenance to compare later models, region strategies, query processing, rerankers, and no-match policies;
- provide a CLI that emits machine-readable JSON suitable for GitHub Actions artifacts and later comparison;
- keep normal CI fast and provider-free by testing evaluation logic with deterministic fakes/fixtures;
- provide an explicit GitHub Actions path for the real baseline using the pinned release, Qdrant, and configured Jina provider;
- remain class-agnostic: benchmark labels describe what is visible, but they never become detector classes or indexing requirements.

## Non-goals

Phase 1G does not:

- switch from Jina CLIP v1 to v2 or another embedding model;
- translate French queries, expand synonyms, or fuse multiple query embeddings;
- change fixed crops/tiles or add region proposals;
- add a VLM/reranker;
- add a no-match threshold or convert cosine into a confidence percentage;
- tune ANN candidate limits, temporal grouping, local cosine scoring, moving average, z-score, or Kadane to improve benchmark numbers;
- decode additional P-frames or implement Phase 2 temporal refinement;
- introduce MLflow, a database of experiment runs, notebooks as the canonical benchmark, or a dashboard;
- commit the 844 MB media release to the repository;
- infer ground truth from the current search ranking.

A benchmark labeled by accepting the current model's own top results as truth would be circular and is explicitly forbidden.

## Why a dedicated evaluation subsystem

A one-off script could print a few Recall@K values, but issue #32 requires repeated comparisons across language handling, crop strategies, embedding models, compositional scoring, reranking, and no-match behavior. The benchmark therefore needs stable typed boundaries without becoming a full experiment-management platform.

The chosen architecture is:

```text
benchmarks/semantic-search/v1/manifest.json
        │
        ├─ dataset provenance
        ├─ fixed queries
        ├─ neutral truth events
        ├─ positive / hard-negative query-event references
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

The core search/indexing subsystems remain the system under test. Evaluation code consumes Winston-owned contracts and does not move benchmark-specific rules into `src/winston/search` or `src/winston/indexing`.

New Phase 1G production code and tests use precise Python types. `typing.Any` and `object` are not used as escape hatches for benchmark data, results, or instrumentation.

## Fixed dataset provenance

The first benchmark version uses the existing public release:

```text
release tag:   test-dataset-v1
release name:  Winston test dataset v1
asset:         videos.zip
size:          844175165 bytes
sha256:        94fa6b655e4118401fb724736d5e21f5e34d52b677f91562af9297e4a08201b6
```

The manifest records both archive identity and the exact extracted media inventory.

Archive provenance contains:

```text
release_tag
asset_name
asset_size_bytes
archive_sha256
```

Each expected extracted media entry contains:

```text
source_path
size_bytes
sha256
```

`source_path` is a normalized POSIX-style path relative to the extracted dataset root, with no absolute path or `..` traversal.

The manual real-baseline workflow validates the downloaded archive size/digest before extraction. The evaluator validates every extracted media path, size, and SHA-256, so `--mode search` remains reproducible even when the original ZIP is no longer present locally.

This provides stable identities for:

```text
benchmark manifest
        +
release archive
        +
extracted media inventory
```

Changing annotations/queries creates a new benchmark manifest version. Changing media creates a new pinned dataset identity. Historical results therefore remain interpretable.

## Benchmark manifest

Canonical path:

```text
benchmarks/semantic-search/v1/manifest.json
```

The JSON is data, not executable configuration. It is parsed into strict Winston-owned models with unknown fields rejected.

Top-level v1 fields are:

```text
schema_version = 1
benchmark_id = "winston-semantic-v1"
dataset
cutoffs
queries
events
```

The first fixed cutoffs are:

```text
1, 3, 5, 10
```

They are part of the manifest, not CLI tuning flags for the canonical baseline.

### Dataset section

The dataset section contains the release/archive fields above plus `expected_media`, where every extracted file has a path, size, and SHA-256.

The v1 implementation computes these per-file identities from the existing `test-dataset-v1` release before freezing the manifest. Files not present in the pinned release cannot appear in truth annotations.

### Query identity

Each query has:

```text
id
text
language
family
variant_kind
equivalence_group (optional)
expect_no_match
positive_event_ids
hard_negatives
```

`hard_negatives` is a list of query-specific references:

```text
event_id
reason
```

The reason belongs to the query-event relationship, not to the neutral event itself. The same visible interval may be a `wrong_color` negative for one query and a `confusable_object` negative for another.

The exact `text` is sent to Phase 1F unchanged. The runner must not translate, rewrite, expand, or generate alternate prompts. Existing search-pipeline whitespace validation remains the only normal query normalization.

### Semantic query families

The controlled v1 `family` vocabulary is:

```text
simple_object
person_object
object_attribute
attribute_binding
person_clothing
person_object_relation
confusable_object
```

Family describes the semantic challenge. Language and wording variation are separate dimensions.

### Variant kind

`variant_kind` is one of:

```text
canonical
translation
synonym
```

This avoids classifying `pram` only as a synonym when it is also semantically a confusable-object query. A query can therefore be:

```text
family = confusable_object
variant_kind = synonym
```

### Language

Initial languages are explicitly:

```text
en
fr
```

Language is metadata only. It does not alter execution.

### Equivalence groups

Queries expressing the same intended visual condition share `equivalence_group`, for example:

```text
stroller / pram / pushchair
une poussette / a stroller
une personne avec un t-shirt blanc / a person with a white T-shirt
```

The group permits paired language/synonym reporting without merging retrieval runs. Every exact text query is still evaluated independently.

## Ground-truth event model

`events` are neutral, reusable descriptions of real visible intervals in the pinned release. Positivity and hard-negative meaning are assigned by each query.

A truth event contains:

```text
id
source_path
media_type
start_timestamp_seconds
end_timestamp_seconds
optional region annotation
annotation_note
```

For video:

- timestamps are finite and non-negative;
- `end_timestamp_seconds > start_timestamp_seconds`;
- no hidden timing tolerance is added by evaluation.

If annotation uncertainty requires a wider valid interval, that uncertainty must be represented explicitly in the truth interval itself.

For still images, timestamps are absent and source identity is sufficient.

An optional manually verified region can document where the relevant object is visible, but Phase 1G semantic relevance does **not** require region/tile IoU. Current Phase 1F regions are intentionally coarse deterministic crops; exact spatial overlap would conflate crop localization with whether Winston found the correct visual event.

Region-specific evaluation can be introduced in a later benchmark version without rewriting v1.

### Positive events

A query's `positive_event_ids` reference events where that exact query condition is genuinely satisfied.

Examples include the interval in which a cyclist is visible or the interval in which a person is actually wearing the requested clothing color.

### Hard negatives

A query-specific hard negative references a tempting but wrong event and assigns one reason:

```text
wrong_attribute_binding
wrong_color
wrong_object
confusable_object
wrong_relation
other
```

Examples from issue #32:

```text
query: red bicycle
negative: red clothing/person next to a non-red bicycle
reason: wrong_attribute_binding

query: person wearing a yellow T-shirt
negative: person carrying a yellow bag while the shirt is not yellow
reason: wrong_attribute_binding

query: stroller
negative: bicycle/cyclist scene
reason: confusable_object
```

Hard negatives make failures diagnosable instead of treating every wrong result as undifferentiated noise.

### Absent queries

An absent query is represented by:

```text
expect_no_match = true
positive_event_ids = []
```

It still has a real semantic family and may reference known hard negatives. `absent` is therefore not a semantic family.

A query may not simultaneously declare `expect_no_match=true` and positive events. Conversely, a positive benchmark query must contain at least one positive event.

Phase 1F currently returns best-available neighbors rather than abstaining, so poor absent-query behavior is an expected baseline result, not something Phase 1G should hide with a threshold.

## Canonical retrieval depth

Every benchmark query is executed **once** with:

```text
limit = max(manifest.cutoffs)
```

All Recall@K/Precision@K values are prefixes of that single ranked list.

This matters because Phase 1F's ANN overfetch behavior can depend on requested result count. Phase 1G therefore measures one explicitly defined retrieval depth rather than pretending that `Recall@1` is the same experiment as executing the CLI separately with `--limit 1`.

The canonical v1 benchmark evaluates the ranking produced for `limit=10`, then measures prefixes K=1,3,5,10. Future benchmark versions may change this contract explicitly, but a single v1 run never executes one query multiple times merely to compute several cutoffs.

## Semantic relevance rule

Phase 1G evaluates the representative visual selected by Phase 1F rather than allowing a broad returned passage to count as correct merely because it overlaps a truth interval somewhere.

A video result can match a positive event when:

```text
result.source_path == event.source_path
AND
event.start <= result.representative_timestamp <= event.end
```

A still-image result can match when source paths are equal.

This isolates the question:

> Did semantic retrieval select a visual moment that actually satisfies the query?

Returned passage boundaries are evaluated separately.

### Deterministic one-to-one matching

Within the ranked prefix, results are processed in rank order. Each positive event can be claimed at most once.

If one result can match multiple still-unmatched positive events, choose deterministically by:

1. smallest distance between representative timestamp and event-interval center;
2. stable event ID.

A later result pointing to an already-claimed positive event does not create extra recall and is unmatched for precision. Duplicate result windows therefore cannot manufacture benchmark quality.

Hard-negative attribution occurs after positive matching. An unmatched result whose representative falls in a query's hard-negative event is labeled with that query-event reason; otherwise it is a generic false positive.

## Semantic quality metrics

Metrics are computed per query first, then aggregated. Raw per-query judgments remain in result JSON so later analysis never depends only on a headline number.

### Recall@K

For positive queries:

```text
unique positive events matched in top K
---------------------------------------
        total positive events
```

Absent queries have no Recall@K denominator and are excluded from positive-query recall aggregates.

### Precision@K

For positive queries:

```text
one-to-one positive matches in top K
------------------------------------
                   K
```

Unfilled ranks count as non-relevant. This prevents a future implementation from improving Precision@K merely by returning almost nothing. Abstention behavior is reported separately.

### First relevant rank and MRR

For each positive query:

- `first_relevant_rank` is the 1-based first positive-match rank, or null;
- reciprocal rank is `1 / first_relevant_rank`, or zero when no positive is found.

MRR is the macro mean reciprocal rank across positive queries.

### Success@K and family accuracy

A positive query has `success@K = true` when at least one positive event is found in top K.

Family metrics named as accuracies are defined exactly as macro Success@K for that family, for example:

```text
attribute_binding_accuracy@K
confusable_object_accuracy@K
```

They are empirical benchmark rates, not probabilities and not model confidence.

### Aggregate recall/precision

Primary aggregate values are macro means across eligible positive queries:

```text
macro_recall@K
macro_precision@K
```

The result document also preserves numerator/denominator counts so micro summaries can be derived without re-running the benchmark.

### Hard-negative rate@K

Hard-negative diagnostics use only actually returned ranks in the top-K prefix:

```text
annotated hard-negative results in returned top-K
-------------------------------------------------
        number of results actually returned in top-K
```

If no result is returned, the rate is null and counts are zero. Counts by hard-negative reason are always emitted.

This diagnostic is distinct from Precision@K, whose denominator remains K.

### Language and synonym consistency

For each `equivalence_group`, preserve per-query metrics and report:

- Recall@K range/gap;
- first-relevant-rank differences when comparable;
- success/failure disagreement at K;
- language and variant labels for each member.

The benchmark does not average embeddings or fuse equivalent queries. Consistency reporting observes current behavior as-is.

### No-match behavior

For `expect_no_match=true` queries, report:

- returned result count;
- whether the engine abstained completely;
- top raw cosine score when present;
- top-K raw score list/distribution;
- annotated hard-negative hits by reason.

Aggregate outputs include `no_match_abstention_rate` plus absent-query raw-score summaries.

Phase 1G does not derive a rejection threshold from these numbers. They are calibration evidence for later #32 work.

## Temporal localization metrics

Semantic correctness and temporal precision are separate dimensions.

For each semantically matched video result, compare returned passage bounds with the matched event interval and report:

- signed start error in seconds (`result_start - truth_start`);
- signed end error in seconds (`result_end - truth_end`);
- passage duration;
- truth duration;
- temporal intersection-over-union (tIoU).

A semantically correct representative can receive semantic credit while exposing an overly broad Phase 1F passage. Passage overlap alone cannot rescue a semantically wrong representative.

This prevents Phase 2 localization work from being confused with embedding/retrieval quality.

## Performance measurements

Performance metrics are reported independently from semantic metrics.

### Instrumentation boundary

Phase 1G does not add benchmark-specific methods to `VisualIndex` or change search/indexing semantics merely to obtain timings.

`evaluation.instrumentation` instead provides a typed delegating `VisualIndex` wrapper used only by the benchmark runner. It implements the same Winston protocol and records:

- total visuals passed to `upsert()`;
- video visuals passed to `upsert()`;
- still-image visuals passed to `upsert()`;
- duration of each ANN `search_visuals()` call;
- total duration spent streaming refinement windows through `iter_visuals()` when applicable.

All real storage behavior remains delegated to the underlying index.

This keeps Qdrant SDK types inside the existing Qdrant adapter and keeps evaluation provider-agnostic.

### Indexing throughput

A canonical full baseline uses an empty/dedicated evaluation Qdrant instance/collection and freshly extracted pinned media.

The runner records wall-clock duration around `IndexingPipeline.run()` and independently computes total video duration by probing the verified dataset.

Report at least:

```text
indexing_wall_seconds
video_footage_seconds
footage_hours_per_wall_hour
total_upserted_visual_count
video_visual_count
image_visual_count
video_vectors_per_footage_hour
indexed_assets
failed_assets
skipped_assets
```

`video_vectors_per_footage_hour` uses only video visuals in the numerator. Still-image vectors remain explicit but never inflate a per-video-hour rate.

A canonical cold baseline is invalid if any asset is skipped or failed. This prevents restart/idempotency behavior from being mistaken for fresh indexing throughput.

### Search latency

For every benchmark query, report separately:

```text
end_to_end_search_ms
qdrant_ann_ms
refinement_scroll_ms
```

`qdrant_ann_ms` is the sum of actual `search_visuals()` calls for that query, including bounded ANN overfetch. It excludes text embedding, exact local cosine, and temporal selection. End-to-end latency contains the complete current search pipeline.

`refinement_scroll_ms` is zero when no video refinement occurs and otherwise records time spent obtaining indexed visuals through the delegated timeline iterator.

Aggregate latency summaries contain:

```text
count
mean
p50
p95
```

`p50` is the ordinary median. `p95` uses deterministic nearest-rank selection: sort ascending and take rank `ceil(0.95 * n)`, clamped to the available sample count. Raw samples are retained.

Each fixed query executes once, sequentially. Phase 1G is a reproducible baseline, not a concurrency/load benchmark, so these latency numbers are descriptive rather than a claim about production tail latency under load.

## Runner and CLI

User-facing command:

```text
winston evaluate semantic <manifest> --dataset-root <path> --output <results.json> --mode <search|full>
```

`--mode` is mandatory so a caller never accidentally triggers expensive indexing.

### Search mode

`--mode search`:

- validates manifest and every extracted media path/size/SHA-256;
- opens the existing compatible configured visual index read-only;
- executes every exact manifest query once at canonical retrieval depth;
- emits semantic, temporal, and search-latency metrics;
- does not invoke mutating index operations.

### Full mode

`--mode full`:

- validates manifest/media first;
- runs the existing `IndexingPipeline` against the configured dedicated evaluation collection;
- requires a cold run (`skipped == 0`, `failed == 0`);
- records indexing metrics through the instrumentation wrapper;
- executes the same search evaluation afterward;
- emits one combined result document.

The CLI never silently deletes or recreates a user's collection. The canonical GitHub Action supplies fresh Qdrant. A local full run is the caller's responsibility to point at a dedicated empty evaluation target; violated cold-run invariants fail rather than publish misleading throughput.

Fatal manifest/runtime/output failures go to stderr and return non-zero. The benchmark result is written only to the requested JSON file, so progress logs never corrupt machine-readable output.

## Result document and reproducibility

Result JSON has an independent schema version and contains at least:

```text
result_schema_version
benchmark_id
manifest_sha256
dataset archive and extracted-media provenance
run_timestamp_utc
Winston revision when available
embedding identity
semantic/resource search settings snapshot
relevant index/Qdrant settings
per-query ranked judgments
per-query metrics
macro/family metrics
equivalence-group metrics
temporal localization metrics
performance samples and summaries
```

Secrets, API keys, raw vectors, and raw media are never written.

The exact manifest SHA-256 is recorded so two artifacts immediately reveal whether they used identical benchmark definitions. Search settings that affect semantic behavior or resource bounds are captured because candidate limits or temporal context changes make runs materially different.

All numeric JSON values are finite. Mathematically inapplicable metrics are `null`, never NaN or Infinity.

## Initial benchmark content

V1 deliberately covers observed Phase 1F failure modes, not only easy positives.

Candidate intents include:

```text
simple objects:
  cyclist
  dog
  bag

confusable objects:
  stroller
  bicycle

synonym variants:
  stroller / pram / pushchair

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

absent cases:
  only concepts manually verified absent from the entire pinned benchmark scope
```

The exact list is constrained by verified footage. If a proposed concept cannot be confidently annotated from the release, it is excluded rather than guessed.

Issue #11's examples (`pink stroller`, `green cap`, `cardboard box`, `red bicycle`, `umbrella`, `blue car`) remain desired intents, but enter v1 only when the release contains a human-verified positive or when deliberately used as a verified absent query. The issue text itself does not prove that the current release contains them.

## Annotation workflow

Ground-truth creation is data labeling, not conversion of model output.

For each candidate intent:

1. inspect the fixed release footage enough to identify true positive intervals independently;
2. record every verified positive event with source path and interval;
3. record particularly misleading wrong intervals as query-specific hard negatives with reasons;
4. verify absent queries over the entire fixed benchmark scope before marking them absent;
5. review source paths/timestamps against the pinned media;
6. freeze the manifest before treating the first baseline scores as evidence.

Current Phase 1F search may help jump to candidate timestamps, but an ANN hit does not become truth until visually verified. Known manual observations can seed where to inspect; the benchmark records the verified event, not the model's claim.

A later benchmark version can add labels/media without rewriting v1 history.

## GitHub Actions

Phase 1G adds two automation paths.

### Permanent contract workflow

Runs on ordinary PR/push changes affecting evaluation code and uses deterministic fake embedders/indexes plus tiny repository fixtures.

It verifies:

- strict manifest validation;
- media inventory validation;
- one-to-one relevance matching;
- Recall/Precision/rank/MRR calculations;
- hard-negative attribution;
- absent-query handling;
- language/equivalence aggregation;
- temporal metrics;
- instrumentation accounting/timing plumbing;
- CLI JSON output/failure behavior;
- source compilation.

No Jina secret, 844 MB download, or real embedding inference is required.

### Real baseline workflow

A manual `workflow_dispatch` workflow executes:

```text
checkout chosen revision
        ↓
download test-dataset-v1/videos.zip
        ↓
verify 844175165 bytes + archive SHA-256
        ↓
extract media
        ↓
evaluator verifies every extracted file size + SHA-256
        ↓
start fresh Qdrant
        ↓
run full semantic benchmark with configured Jina engine
        ↓
upload results.json as workflow artifact
```

The workflow uses the repository Jina secret and fails clearly when unavailable. It does not run on every push because the large download and provider inference are intentionally expensive.

The artifact is evidence for the baseline and is not automatically committed back to the repository.

## Error handling and validity rules

A run fails rather than silently degrading when:

- manifest schema/version is unsupported;
- benchmark/query/event IDs are invalid or duplicated;
- a referenced positive/hard-negative event does not exist;
- the same event is redundantly referenced within one query role;
- `expect_no_match=true` has positives;
- `expect_no_match=false` has no positives;
- cutoffs are empty, non-positive, duplicated, or unordered;
- language/family/variant/reason is unsupported;
- timestamps are invalid, non-finite, or non-positive-duration for video truth;
- event source paths are non-normalized or outside pinned media inventory;
- extracted media path/size/SHA-256 differs;
- release archive size/digest differs in the real workflow;
- configured search/index embedding identity is incompatible;
- full mode skips or fails assets;
- full mode has zero video footage for a metric that requires video hours;
- evaluation receives malformed/non-finite search results;
- output path cannot be written.

A query returning zero results is a valid measurement, not a runner failure.

No hidden timing tolerance, score threshold, or automatic query fallback may be introduced by the evaluator.

## Testing strategy

Development follows RED -> GREEN TDD.

### Manifest/model tests

Cover:

- valid v1 manifest;
- strict unknown-field rejection;
- duplicate/invalid IDs;
- missing event references;
- malformed archive/per-file SHA-256;
- invalid relative source paths;
- invalid timestamps;
- invalid/duplicate/unordered cutoffs;
- contradictory absent/positive definitions;
- unsupported family/language/variant/reason;
- media inventory mismatch.

### Relevance tests

Cover:

- representative timestamp inside/outside event;
- no implicit temporal tolerance;
- same timestamp on wrong asset;
- deterministic one-to-one matching;
- duplicate returned windows do not inflate metrics;
- query-specific hard-negative attribution after positive matching;
- one event used with different negative reasons by different queries;
- generic false positives;
- still-image source matching.

### Metric tests

Use hand-computable ranked lists to prove exact values for:

- Recall@1/@3/@5/@10;
- Precision@K with unfilled ranks;
- first relevant rank and MRR;
- Success@K;
- macro Recall/Precision;
- attribute-binding/confusable-object family accuracy;
- hard-negative counts/rates and zero-result null rate;
- absent-query summaries;
- equivalence/language gaps;
- temporal signed errors and tIoU;
- deterministic p50/p95 latency summaries.

### Instrumentation/runner tests

Typed fakes prove:

- wrapped upserts count total/video/image visuals without changing delegation;
- ANN latency samples correspond to real delegated `search_visuals()` calls, including overfetch;
- refinement iterator timing does not alter yielded visuals;
- full mode rejects skipped/failed runs;
- search mode never invokes mutating index operations;
- each query executes exactly once at `max(cutoffs)` and is not rewritten;
- settings/embedding/manifest/media provenance is emitted;
- raw cosine remains raw cosine and is never relabeled confidence.

### Real integration

The existing Qdrant workflow receives focused evaluation coverage where useful. The expensive release/Jina baseline remains a manual workflow.

## Compatibility with issue #32

After v1 baseline is frozen, #32 experiments can run against the same manifest and produce comparable result JSON for:

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

Experiments requiring a different visual embedding space must reindex but keep the benchmark manifest fixed. Experiments altering only query handling can reuse a compatible index. Every result artifact records identity/settings so incompatible runs are visible rather than silently compared.

Issue #32 remains open after Phase 1G. Phase 1G supplies its benchmark foundation but does not complete the investigation.

## Branch and PR strategy

Phase 1G is implemented in one focused branch/PR:

```text
feat/search-evaluation
```

The branch starts from the validated Phase 1F head because evaluation depends on semantic-search contracts introduced by PR #28. While #28 is unmerged, Phase 1G remains stacked on that work. After Phase 1F lands on `main`, Phase 1G is retargeted/rebased as appropriate without changing the benchmark contract.

The Phase 1G PR description includes:

```text
Closes #11
```

## Acceptance criteria

Phase 1G is complete when:

1. a strict versioned manifest pins the exact release archive and every extracted media file identity;
2. v1 contains independently verified positives, query-specific hard negatives, language/synonym pairs, and verified absent cases across several semantic families without predefined classes;
3. each query executes exactly once at fixed retrieval depth and all metrics derive from that ranked list;
4. normal CI proves manifest, relevance, metrics, instrumentation, runner, and CLI contracts without external providers;
5. the manual real-baseline workflow reconstructs verified media, uses fresh Qdrant, runs current Jina CLIP v1 indexing/search, and uploads result JSON;
6. result JSON reports Recall@K, Precision@K, first relevant rank/MRR, family Success@K, hard-negative behavior, language/synonym consistency, and no-match behavior;
7. temporal localization is reported separately from semantic correctness;
8. indexing throughput, video vectors per footage hour, end-to-end search latency, and Qdrant ANN latency are separate metrics;
9. no metric is called confidence and no uncalibrated rejection threshold is introduced;
10. ground truth is not derived automatically from current model rankings;
11. the baseline artifact can be reused unchanged to compare later issue #32 experiments.

Phase 1's broader quality exit criterion is evaluated from this artifact rather than silently enforced by tuning Phase 1G. If the frozen baseline shows a semantic weakness, that evidence becomes input to #32 instead of changing annotations or evaluator behavior after seeing the scores.
