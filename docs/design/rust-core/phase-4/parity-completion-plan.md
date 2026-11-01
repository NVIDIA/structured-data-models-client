<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 parity completion plan

**Approved and active: implement known gaps first, then run the real endpoint matrix and fix discrepancies.**
This document is the persistent goal’s implementation and acceptance reference.
The parity checks are permanent product-test infrastructure, retained after the merge into `main`.

**Completed original scope:** [the generated matrix](parity-matrix.md) records all 536 required comparisons passing: 291 Python and 245 Java, including S3.
Its 46 deferred Java Snowflake cells describe the previous scope, not a claim that Snowflake is implemented.
All implementation and reviewed test fixes are integrated; the owner explicitly deferred dedicated CI access until later, so local real-endpoint verification closes this milestone.
The protected workflow and release checks remain implemented; [CI deployment](parity-ci.md) still needs dedicated credentials and a suitable runner.

**Approved follow-up:** implement Java Snowflake for Relational and Tabular, then verify the 46 ordinary-table and semantic-view cases against the existing frozen Python evidence.
Relational should reuse shared Rust warehouse policy through a caller-owned JDBC connection; Tabular should read one table into Arrow and retain its in-memory API.
Review connector/type boundaries before implementation, preserve existing verified evidence, and run focused regressions plus actual Snowflake-to-model journeys after the implementation.

### Java Snowflake boundary plan

```text
Relational: SnowflakeSource → Rust plans catalog/sampling SQL
                            → Java executes JDBC and returns Arrow
                            → Rust builds the model request → predict or fit/predict/close

Tabular:    Readers.readTable/readQuery → Java JDBC → Arrow table
                                     → existing Tabular materialize/predict or fit/predict/close
```

| Boundary | Decision |
|---|---|
| Public API | `SnowflakeSource.of(connection[, database, schema])`, plus the existing `semanticModel(name, trustExpressions)` pattern; the caller owns authentication and the connection |
| Java host | Execute statements, bind values, preserve SQL error details, and convert JDBC results to owned Arrow batches without losing numeric or timestamp meaning |
| Shared Rust | Reuse existing Snowflake catalog, namespace, semantic-view, sampling and session policy; no parallel Java policy or unrelated warehouse SQL changes |
| Type check before coding | Probe the actual driver against Python Arrow for exact numbers, nulls, DATE/TIME, NTZ/LTZ/TZ and metadata statements; review any gap before choosing a conversion |
| Acceptance | Focused type/ownership regressions, independent reviews, then all 46 former Snowflake deferral cells against real sources/models; preserve the preceding 536 results and rerun only affected neighbors |

The JDBC driver is optional for consumers who do not use Snowflake.
No connection or session settings are changed behind the caller's back.

## Acceptance: what “done” means

Use frozen Python `rust-core-reference` (`43e55f4f5`) as the common baseline for **both languages**.
Java lacking an API today does not mean the old client never supported the capability.
Rows are **connector/input × prediction/task × lifecycle**; columns are **Python Relational | Python Tabular | Java Relational | Java Tabular**.
Each completed cell must say **Parity**, **Improvement** (with a documented reason), or **Parity, never supported** (verified against the old client).
An untested, blocked or failing cell stays incomplete; the follow-up must replace Java Snowflake's historical deferral with actual evidence, never relabel it as parity without testing.
Preserve the approved single-table Tabular API: source reads become in-memory rows, not `Schema.from_source` support.

## Implementation order

| Step | Concrete work | Boundary / completion check |
|---|---|---|
| 1. Remove active legacy rules | Replace in-memory PQL shadow validation with Rust-resolved query facts; move frame schema edits into shared Rust | Preserve frame-vs-warehouse provenance explicitly; do not pretend frame observations are catalog facts or add Python callbacks |
| 2. Close Java prediction gaps | Forward warehouse candidates; retain decomposed rank/forecast requests; reuse existing Rust prediction merging; align model defaults | One shared policy for memory and warehouse, including typed candidates, partial answers and ownership |
| 3. Close API gaps | Add direct-request schema/inference/task options and Tabular PQL task override; reconcile result merge/unscored metadata, health/model discovery, read helpers and persistence | Equivalent user capabilities, idiomatic Java types; review public signatures and serialization format before coding them |
| 4. Resolve remaining scope | Verify old-client support for Java-missing Spark/file/S3 input paths, custom-provider time rules, large Arrow values and fallback behavior; implement supported requirements or record an explicit owner deferral | No silent exclusions; JDBC Databricks is not proof of Spark-source parity; Python pickle bytes need not become Java's format |
| 5. End-to-end sweep | Run the matrix below on installed Python and Java artifacts, against real data sources and model endpoints | No mocked model responses, skipped tests or code inspection counted as E2E parity |
| 6. Repair and close | Reproduce mismatches, fix them, rerun affected cells and regression neighbors; reconcile original phase exits | Independent review, evidence-linked matrix, signed commits on `rust-core-phase-4` |

Steps 1 and 2 can run in parallel; agree any shared contract changes first.
Use small regressions for boundary/type decisions, then consolidate endpoint testing after implementation.
Keep warehouse sampling SQL unchanged unless a reproduced correctness gap requires a specific reviewed fix.

### Implementation checkpoint

- Delivered Java warehouse candidate/ranking/forecast handling and model defaults (`f04b23144`).
- Implemented Rust-resolved in-memory queries, frame schema edits and fresh Arrow/pandas key observations, plus Java direct options and Tabular task overrides (`7895f4fd4`, `9aaeb732e`).
- Added standalone schema checks, explicit missing-dtype preservation, Java result merging, request parts and endpoint discovery (`01f460072`, `5baef1dcf`).
  The latest installed Python sweep passed 735 cases; both failures were corrected and all 55 cases in the affected files then passed.
  Java passed 69 cases with two environment skips; Clippy passed across core and JNI.
  These are regression results, not live parity certification.
- Shared Python/Java result-merge policy is implemented (`3b0bddd0d`, `4fe90fa0e`), preserving original Python row/metadata owners.
  All 58 installed Python regression checks pass, including the corrected shared error diagnostics.
  Native merging keeps exact numeric candidates and compares aware datetimes by UTC instant; stable deduplication removes duplicates within each part too.
- Java readers and caller-owned Spark sources landed (`b6196d1fd`): local/S3 CSV and Parquet, compressed files, JDBC reads and optional Spark integration.
  Java regression results: 299 passed, 14 explicitly skipped; both independent source reviews passed.
  Real S3 CSV/ZIP reads and a real Databricks Spark source journey passed with temporary fixtures cleaned up.
  The Spark journey used a local model stub, so it does not count as real-model parity.
- Rust provider fixes landed (`2d4021249`): client date rules, declared fallback/recovery, explicit text-order compatibility and slice-aware byte limits.
  Validation passed 132 Rust tests, 50 installed Python regressions, contract checks and Clippy.
  Actual >2 GiB Python/DuckDB probes passed: binary is omitted through prediction; text describes then refuses materialization by name (local model stub).
  Java large-value transport is covered separately below; Python evidence does not establish that cell.
- Java persistence landed (`50f40e34c`): schema, direct requests and materialized requests have data-only stream snapshots; caller streams stay open and live connections/sessions are excluded.
  Java regression results: 306 passed, 14 explicitly skipped; two source reviews passed.
- Java DuckDB streaming landed (`22abbf6e1`), retaining result/statement ownership until the Arrow stream closes without whole-result byte-array serialization.
  Java regression results: 310 passed, 14 explicitly skipped; both source reviews passed.
  An actual >2 GiB binary column was omitted through materialization and prediction using a local model stub; live-model acceptance remains pending.
- Permanent parity machinery landed (`af727a749`): shared fixtures, Python/Java runners, exact comparisons, legacy-route checks, API coverage and CI/release gates.
  Frozen Python completed all 285 required baseline cells; observations are retained for candidate comparison.
- Shared request inspection landed through `fc5dec307`, with [reviewed boundaries](request-split.md), contract 55 and native decoding for both languages.
  Python integration passed 133 tests; Java production API tests passed 200, with 15 explicit optional/provider skips.
  The comparator passed 30 tests, including rejection of wide-integer precision loss.
- Initial live validation completed using immutable production artifacts from `6b7875900`.
  All 285 Python cells compare successfully against the frozen client; Java completed all 239 applicable journeys, with result comparisons exposing a public result-access gap.
  Exact nullable entity-reference integer preservation is reported as an improvement; lossless timestamp-unit differences require identical values.
  Raw evidence is preserved, and private model bridges remain explicitly labeled.
- The reviewed result-access fix and Java fitted-output option landed in `7f7c19f95`: Rust projects semantic result items for Python and Java, including ranking candidates and mixed merged results.
  Matching production artifacts are built from that commit.
  Python passed 272 API checks with no skips and 144 focused result/direct/session tests; Java passed 208 public API tests with 15 explicit optional/provider skips.
  Final live comparisons are running against 291 frozen Python baseline cases, with 245 applicable Java cases.
  All 29 memory comparisons pass in both languages, including explicit outputs, batching and actual fitted-session holdout payloads and session handles.
  All non-S3 cells compare successfully: 269 Python and 223 Java.
  Java CSV/ZIP fixtures now match the frozen client's timestamp text, guarded against truncating non-midnight values; only the affected 22 journeys were rerun.
- Frozen Python and the final `7f7c19f95` Python candidate both passed the actual-model >2 GiB omitted-binary journey.
  Java also passed the same shop task with a >2 GiB binary value and eight finite real-model predictions; the earlier toy task's model refusal is preserved separately.
- The protected live workflow now requires sequential actual-model large-value tests for all three clients, with exactly one passed test each and no skips.
  Dedicated CI configuration remains pending; this is implemented enforcement, not a claim of an operational scheduled run.

AWS access was refreshed and all 44 final S3 cases passed; completed evidence is retained.

**Still required for the follow-up:** implement and validate Java Snowflake for both models, including semantic-view input where frozen Python supports it.
Dedicated CI access is deferred by the owner; [the scheduled workflow is not yet operational](parity-ci.md).

Access preflight verified Databricks SQL, Lakebase, Snowflake, an existing S3 fixture and readiness of the three model endpoints.
Java Databricks Connect completed its real-model Spark journeys; final result comparisons remain pending with the corrected result-access artifact.

## Final E2E matrix

- **Inputs:** memory, SQLite, DuckDB, PostgreSQL/Lakebase, Databricks SQL/Spark/metric views, Python Snowflake/semantic views, and supported file/S3 ingestion.
- **Tasks:** node regression, binary/multiclass, temporal aggregates/conditions, foreign-key classification, direct temporal links, candidate ranking and forecasts; verify old refusals for model-incompatible tasks.
- **Journeys:** schema/read → materialize → predict; fit → repeated predict → close where supported; requested outputs, batching, holdout/leakage and representative failure/cleanup behavior.
- **Comparison:** same source snapshot, query/options and model version for frozen Python/new Python/Java; compare identities, context/labels, task/options, result schema/meaning and deterministic answers, with justified tolerances for nondeterministic model outputs.
- **Route proof:** trace the whole journey, including setup and worker threads, and fail if legacy validator/schema/sampler/adapter policy executes; retain only intentional host conversion, database I/O, HTTP and presentation.
- **Evidence:** every cell links to its fixture, commands, artifact/model identity and sanitized result/log; source-connection access and model-serving access are separate prerequisites.

Check existing endpoint capabilities and connector access early, without running the full suite early.
A private validation bridge may exercise the model but must be identified; it cannot establish public Databricks transport parity.
Remove or isolate production legacy-selection overrides only after supported paths are covered; keep the frozen reference in a separate test environment.

## Confirmed scope

The frozen Python client already provides Spark-session sources, CSV/Parquet reads from local files/S3, database `read_table`, model listing, readiness checks, result merging, and schema pickle round trips.
These capabilities are in scope for equivalent Java support; exact persistence coverage must be checked per object, not inferred as support for serializing connections or live sessions.
Java equivalents need not read Python pickle bytes or use identical method names.
Java Snowflake was deferred in the original milestone and is now the approved next implementation scope.

Frozen-runtime probes also confirmed pickle round trips for direct requests and materialized Tabular and in-memory Relational requests, not just schemas.
Prediction after restore remains an E2E acceptance case.
Original scope still includes large-value description (the old client could omit a >2GiB binary column and predict), custom-provider whole-column text-time rules, and declared provider fallback.
The latter two are provider capabilities, not failures of the five built-in SQL connectors; ordinary frame sampling does not discharge them.

## Permanent parity machinery: small and enforceable

Use a product-owned `tests/parity/` directory, not migration-only `contract/` or `scripts/rust_core/` infrastructure.

| Piece | Responsibility |
|---|---|
| One capability manifest | Stable case IDs, Python↔Java API mapping, model/connector applicability, expected support/refusal, baseline evidence and explicit deferrals |
| Shared fixtures + two thin runners | Same input data/options through public installed Python/Java APIs; existing pytest/JUnit and connector helpers do the work |
| One result format/comparator | Check semantic results, identities, errors and lifecycle; record artifact, fixture and endpoint versions; documented tolerances/improvements only |
| Generated Markdown matrix | Four language/model columns; statuses derived from evidence, never hand-maintained green cells |

The manifest describes expected behavior; run results describe what was actually verified.
Every capability needs tests or an explicit tracked deferral; additions/removals require updating the manifest and both language mappings.
An absent API, missing case, unexpected fallback, skipped required test or unsupported-but-previously-supported cell fails its gate.
Different model capabilities are intentional where verified; unsupported combinations need refusal tests and old-client evidence, not a model call that should never happen.

**PR CI:** API coverage/mapping checks, local behavioral contracts, refusal cases, legacy-policy tripwires and matrix-generation consistency.
These checks catch regressions cheaply but do not certify live parity.
**Protected scheduled/on-demand CI and release gate:** actual connector→model endpoint journeys for every supported matrix case, using installed artifacts.
Missing credentials or unavailable endpoints produce incomplete coverage, never a passing cell; retain logs and cleanup created fixtures/sessions.
Final Phase 4 acceptance and releases require complete applicable live coverage for the candidate, not an old green report.

Keep frozen Python isolated for the initial E2E comparison and reproducible historical baselines.
Retain versioned fixtures, expected contracts and approved improvement decisions for ongoing regression checks; do not make the shipped client depend on the old package.
Record model versions so endpoint changes cannot silently redefine parity.
No custom scheduler, generic testing framework or duplicate hand-written matrices are needed.
No new deployment or access request is needed for planning; verify available remote-source credentials and endpoint support at implementation kickoff, then ask only for concrete missing access.

Findings behind this plan: [coverage audit](coverage-audit.md).
