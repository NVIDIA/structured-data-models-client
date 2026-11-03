<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First four restorations

Each feature has its own branch/worktree, public Python/Java contract review, focused tests, implementation and independent code review.
The coordinator merges reviewed commits into `rust-core-phase-6` and reconciles shared API inventories after integration.
Use at most two heavy jobs, Cargo jobs=2 and the existing heavy guard; do not install competing native builds into a shared environment.

| Feature | Branch | Status |
|---|---|---|
| Visualization | `rust-core-phase-6-visualization` | Reviewed and merged; 13 Python / 3 Java tests passed; wheel/JAR assets verified |
| Sanitization | `rust-core-phase-6-sanitization` | Reviewed and merged; 321 Rust / 74 Python tests and Java relational tests passed |
| RelBench | `rust-core-phase-6-relbench` | Reviewed and merged; 58 Python / 4 Java tests passed; real dataset matches across hosts |
| Reports/explanations | `rust-core-phase-6-reports` | Reviewed and merged; shared fixtures and contract 59 verified |

Three workers can run alongside the coordinator; the fourth starts when a slot frees.
No feature is complete on implementation alone: contracts, affected tests and independent review must pass.
Remaining audit capabilities stay in the Phase 6 scope and are not implicitly completed by these four merges.

Sanitization introduced contract 58.
Reports adds contract 59 for lazy attribution aggregation.
Final integration passed 157 Python tests against the installed wheel and 33 Java tests.
Wheel/JAR assets and packaged native loading passed.
See [completion evidence](completion.md) for scope, reviews and remaining audit work.


## Next three restorations and stronger coverage

| Work | Branch | Status |
|---|---|---|
| Inference and safe edits | `rust-core-phase-6-inference` | Reviewed and merged; 95 affected Python / 12 Java tests and live three-row prediction passed |
| Time-series preparation | `rust-core-phase-6-timeseries` | Reviewed and merged; Python/Java fixtures and live four-row forecast passed |
| Labels-only extraction | `rust-core-phase-6-labels` | Reviewed and merged; 21 Python / 3 Java tests and live Databricks/Snowflake labels passed |
| Thorough testing of the first four | `rust-core-phase-6-coverage` | Reviewed and merged; composed workflows, corruption/ownership/report edge cases; 22 Python / 15 Java tests passed |

The three designs under `phase-5/api-removal-audit/{inference-edits,timeseries,labels-only}/design.md` define the scope and testing expectations.
Use existing test infrastructure, focused old-client comparisons and independently calculated expected results; do not rebuild migration instrumentation.
The coverage worker audits actual tests for the first four features, closes meaningful missing public-workflow/security/ownership/error coverage, reproduces and fixes real bugs, and records remaining limits honestly.
Prioritize real execution over mocks where it exercises the feature; use model endpoints only for workflows that actually predict.
All changes receive independent review and merge into Phase 6; no publication or merge into `rust-core`/`main`.
The coordinator serializes contract-version allocation and warm-target ownership to avoid conflicting native builds.

All three contracts are integrated through contract 62; the warm native target is released.
Inference/labels may develop isolated logic and tests, but must integrate the preceding contract before generating their own version.

See [restoration completion](restoration-completion.md) for combined production-artifact checks and remaining audit scope.

Contract lock policy retains merged entries and only the newest unmerged version via `scripts/rust_core/contract_lock.py --prune-working`; schema history files remain unchanged.

[Original-feature coverage audit](coverage-audit.md) records the focused additions and their limits.


## Remaining scoped restorations

| Work | Branch | Status |
|---|---|---|
| End-time roles and future-value masking | `rust-core-phase-6-end-time` | Reviewed and merged (`b4292b660`); local suites, old-client comparison and live Databricks/Snowflake predictions passed |
| SQL columns and custom relational examples | `rust-core-phase-6-examples` | Reviewed and merged (`3a413d0ab`, `1f7c191a9`); SQL and supplied examples passed local, real warehouse/model and final combined artifact checks |
| Deeper original-four coverage | `rust-core-phase-6-coverage2` | Stopped under the replacement goal; unmerged work preserved |

The authoritative designs are `phase-5/api-removal-audit/end-time/design.md` and `expressions-tasks/design.md`.
Use independent expected answers and focused old-client comparisons; do not preserve demonstrated bugs or introduce an exhaustive differential framework.
Contracts 63, 64 and 65 are merged; the warm target is released.
Native supplied forecasting was verified against the existing forecast endpoint: one prediction example per batch, with the horizon no larger than the context count.
The client will preserve caller row/forecast-step alignment while batching; no deployment change or regression decomposition is needed.
The replacement goal assigns implementation and testing to the two feature agents only.
The coordinator reviews boundaries and integration; the feature agents also review each other’s changes.
The separate original-four coverage branch is preserved but is outside this goal.
All implementations require independent review, affected checks and integration into Phase 6 only.

Supplied-example contract 65 is complete and merged; final production artifacts passed 145 local Python checks, six live model cases and 22 Java tests.
Its shared Python/Java expectations live in `tests/fixtures/custom-examples/`; the fixture documents the old timeless-anchor overflow that the new implementation must avoid.
Independent review covers explicit task metadata without fabricated PQL, authoritative labels, unchanged sampling, checked forecast response alignment and input ownership.

See [final restoration evidence](final-restorations.md) for the completed scope, validation and limitations.

## Connection and execution follow-up

The owner approved the remaining mappings in [the short audit and plan](remaining-api-mappings.md).
A new goal covers these three efforts; prior restoration evidence remains valid for its recorded scope.

| Effort | Implementation branch | Status |
|---|---|---|
| Connection rebinding | `rust-core-phase-6-rebinding` | Reviewed and integrated; 23 local Python / 3 local Java tests, five live Python warehouse/model journeys and one live Java journey passed |
| Batch controls, retries and automatic session reuse | `rust-core-phase-6-batching` | Contract approved: supplied batch-size control, identical-context session reuse, bounded recovery and centralized retries; implementing |
| Typed settings and optional summary generation | `rust-core-phase-6-settings` | Reviewed and integrated; 22 Python / 6 Java checks passed; final combined inventory and validation pending |

Each agent submits its precise API/boundary proposal before substantial implementation.
The coordinator reviews proposals, serializes native contract versions/builds and arranges reciprocal implementation review.
Tests cover independently expected public behavior, focused old-client comparisons, ownership/errors and real provider/model journeys where applicable.
Controlled HTTP endpoints exercise failures and lifecycle behavior that live services cannot reliably reproduce.
Summary live validation uses only an existing authorized endpoint and synthetic data; any external validation limit is recorded explicitly.

Native contract allocation: rebinding 66 first, batching 67 after integration; settings/summaries are host adapters.
Rebinding released the shared native target after production 66; batching owns it for production 67 and final lifecycle validation.
No authorized live summarizer endpoint was found in the existing model configuration; controlled HTTP tests are required and real-provider validation remains explicitly unclaimed.
