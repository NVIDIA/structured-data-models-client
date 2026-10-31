<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 parity completion plan

**Approved and active: implement known gaps first, then run the real endpoint matrix and fix discrepancies.**
This document is the persistent goal’s implementation and acceptance reference.
The parity checks are permanent product-test infrastructure, retained after the merge into `main`.

## Acceptance: what “done” means

Use frozen Python `rust-core-reference` (`43e55f4f5`) as the common baseline for **both languages**.
Java lacking an API today does not mean the old client never supported the capability.
Rows are **connector/input × prediction/task × lifecycle**; columns are **Python Relational | Python Tabular | Java Relational | Java Tabular**.
Each completed cell must say **Parity**, **Improvement** (with a documented reason), or **Parity, never supported** (verified against the old client).
An untested, blocked or failing cell stays incomplete; Java Snowflake stays visibly deferred, never relabeled as parity.
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
- Next: shared policy for Python's public result merge, Java readers/persistence, remaining provider obligations and permanent parity machinery.
  The real endpoint matrix and remaining scope reconciliation are still pending.

Access preflight verified Databricks SQL, Lakebase, Snowflake, an existing S3 fixture and readiness of the three model endpoints.
Spark runtime access remains unverified.

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
Only Java Snowflake remains explicitly deferred.

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
