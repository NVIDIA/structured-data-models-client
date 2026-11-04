<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Main parity: remaining designs

**Inventory: four product fixes, two CI improvements, and one evidence follow-up.**
Temporal-link batching, wheel provenance and Java analysis are now implemented and merged; see [completion evidence](implementation.md).
The remaining three designs now record the approved DuckDB version policy, explanation ownership boundaries and PostgreSQL correction.
Implementation has not started; PR #63’s final explanation API remains pending upstream review.
The latest investigation also records main `847a1e9e3` (PR#62, `EXPLAIN PREDICT`) in the explanation design; the original audit remains pinned to its earlier snapshot.
Based on the [commit audit](../main-commit-audit.md): main `1c5e2e71d` versus Phase 6 `2e29a56a9`.
These are missing behaviors, not a proposal to restore the old Python engine.

| Design | What the user loses today | Owner |
|---|---|---|
| [Gradient explanations](gradient-explanations/design.md) | Explanations for every prediction, including related rows | Rust interprets results; hosts present them; Python runs optional local models |
| [Temporal-link batching](temporal-link-batching/design.md) | Safe automatic batching when predicting future links for more than 200 entities | Rust request planning |
| [PostgreSQL ordering](postgres-ordering/design.md) | Reliable sampling from keyless tables containing JSON/XML | Rust sampling and SQL dialect |
| [DuckDB 2](duckdb-2/design.md) | Installable DuckDB 2 support without unnecessary result copies | Python dependency/provider adapter |
| [Wheel provenance](wheel-provenance/design.md) | Strong evidence that CI tested the actual candidate wheels | CI only |
| [Java analysis](java-analysis/design.md) | Static security analysis for the new Java code | CI only; not an old-client feature |
| [Evidence follow-up](evidence-followup/design.md) | Not an established product gap: resolve three regression checks and old engine logging | Tests/documentation first |

## Models, connectors and sampling scope

| Item | Models | Connector scope | Push-down impact | Language |
|---|---|---|---|---|
| Explanations | Both, for supported explanation/task combinations | All supported input origins | Consumes already sampled results; no SQL/selection change | Python + Java; local model runtime Python-only |
| Temporal-link batching | Relational temporal links only | All routes that already support this task | Changes per-call batch size for push-down and memory paths; not SQL policy | Python + Java |
| PostgreSQL ordering | Relational neighbor sampling | PostgreSQL + Lakebase only | **Direct SQL ordering fix** | Python + Java |
| DuckDB 2 | Relational directly; Tabular via loaded rows | Python DuckDB only | Provider reads/results, not Rust SQL policy | Python |
| Wheel provenance | Both indirectly | All Python packages indirectly | None | Python distributions, including native Rust module |
| Java analysis | Both indirectly | All shipped Java adapters | None | Java |

“All input origins” means memory, SQLite, DuckDB, PostgreSQL/Lakebase, Databricks SQL/Spark, Snowflake, and files/S3 after loading, within each route's existing capabilities.
It does not certify every endpoint/task/semantic-view combination.
Tabular uses in-memory rows: a warehouse reader can supply them, but it does not perform Relational-style push-down sampling or accept a source-backed schema through this work.
Lakebase uses the PostgreSQL path, not the Databricks SQL dialect.
Each linked design contains the narrower applicability and regression boundaries.
This is source-path coverage, not a new cross-connector test run.

## Boundary rule

Rust owns model/task rules, sampling decisions and interpretation of model responses.
Python and Java own connections, transport, input/output adapters and presentation.
A utility does not belong in Rust merely because both languages could use it.
None of these designs replaces push-down sampling; PostgreSQL ordering is a narrow correction inside that existing path.

## Pending upstream change

[PR #63](https://github.com/NVIDIA/structured-data-models-client/pull/63) is **open**, latest inspection at head `f240fe448` on November 4.
It proposes `result.explanations.gradients` plus client-computed `.columns`, changing the explanation API and fit/predict lifecycle.
Do not freeze the new public explanation shape or claim final main parity until its reviewed outcome is reconciled.
The other designs can proceed independently.

## What counts as complete

Reproduce each confirmed defect through the public API before changing it, then add small maintained regressions.
Use Rust tests for engine policy, Python/Java public tests for adapters, and targeted real database/model checks where required.
Test expected outcomes directly; compare selected old-client cases only where useful.
Do not preserve old bugs, add a second legacy runtime, or build another comparison framework.
Update the existing capability ledger and regenerate public/wire snapshots only when their contracts change.
Implementation status and fresh test results are recorded separately in [completion evidence](implementation.md) and each completed design.
