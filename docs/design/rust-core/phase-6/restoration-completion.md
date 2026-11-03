<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Three more API restorations

**Complete: all three features and stronger original-feature coverage are merged into `rust-core-phase-6`.**
All work targets `rust-core-phase-6`, with no publication or merge into `rust-core`/`main`.

| Capability | What users get | Evidence |
|---|---|---|
| Inference and safe edits | Python/Java infer missing metadata or links, remove tables/links and inspect link origins without mutating the original schema | Shared fixtures, persistence, closed-provider edits, DuckDB materialization; live edited-schema prediction returned 3 rows |
| Time-series preparation | Python/Java turn array histories into an ordinary schema and forecasting query | Shared timestamp/history fixtures, invalid inputs and Arrow ownership; live forecast returned 4 rows |
| Labels without a model | Python/Java calculate historical target values for scalar, temporal, forecast and link queries | Independently calculated fixtures and focused old-client comparisons; memory, SQLite, DuckDB, Databricks and Snowflake |
| Original four features | Stronger visualization, report, sanitization and RelBench coverage | Composed workflows, persisted drop counts, malformed reports, ownership and corrupted downloads; [coverage audit](coverage-audit.md) |

## Boundaries and improvements

```text
Python / Java: public API, source access, output ownership
                         ↓
Rust: inference decisions, typed history preparation, target evaluation
                         ↓
Existing Arrow data and provider operations
```

Existing prediction sampling algorithms and warehouse SQL renderers are unchanged.
Labels use the same target calculations as prediction, without gathering neighborhood features or contacting a model.
Explicit forecast entities use existing primary-key lookup.

Improvements include atomic schema edits, preserving explicit disabled inference choices, honest unknown provenance for older records, and rejecting mismatched per-entity histories that the old helper could misalign.
Independent review also caught inconsistent empty-label typing; empty outputs now use the actual engine conversion.
Decimal classification remains value-dependent: colliding numeric class values require string labels, so empty results cannot predict that dtype without reading rows.

## Final validation

- Installed production wheel: contract 62, test-only operations disabled.
- Combined Python feature/session checks: 142 passed, 2 optional browser checks skipped.
- Combined Java feature/persistence/relational checks: 49 passed.
- Labels: 21 Python tests and 3 Java tests passed; both hosts exercise the shared cases and local providers.
- Live labels: Databricks and Snowflake passed scalar, temporal count and link checks against independent expected answers, without model calls.

These counts are separate runs, not a claim of exhaustive old/new equivalence.
Earlier browser and original-feature endpoint evidence remains in [completion.md](completion.md); it was not repeated here.
Native libraries were built locally for Linux x86-64; this is not new cross-platform release qualification.
Rust tests, Clippy, Python types/lint, repository hygiene, generated API snapshots, the API ledger and contract-lock policy passed.
The final ledger records 217 Python and 457 Java public symbols.

The broader removal audit still contains end-time configuration and expression/task-editing work outside this goal.
