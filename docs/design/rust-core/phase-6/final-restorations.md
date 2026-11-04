<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# End-time and supplied-example restorations

The two scoped efforts are complete, independently reviewed and merged into `rust-core-phase-6`.
Nothing is published or merged into `rust-core` or `main`.

| Restored workflow | Public API and responsibility |
|---|---|
| End-time columns | Python table declarations / Java schema edits; Rust checks eligibility and hides future end dates in model-input copies |
| Calculated SQL columns | `with_sql_column` / `withSqlColumn`; providers execute their SQL dialect, Rust admits the resulting schema |
| Supplied relational examples | `RelationalTask` + `materialize_examples` / `materializeExamples`; Rust validates caller labels/IDs/times and uses existing graph gathering |

Hosts adapt DataFrame/Arrow inputs, own connections and transport, and present results.
There is no synthetic PQL query, Python sorting callback or replacement sampling algorithm.

## What was verified

- End-time: 16 Python cases, shared Java memory/SQLite/DuckDB cases, Rust validity checks, focused old-client comparison and live Databricks/Snowflake predictions.
- SQL columns: Python/Java local providers and five real provider-to-model journeys, including metric/semantic views; focused old DuckDB comparison.
- Supplied examples: seven shared task fixtures; 15 local Python cases, four Java tests covering memory/SQLite/DuckDB, persistence, ownership, invalid inputs and forecast alignment.
- Live supplied examples: regression, binary/multiclass classification, temporal links, and forecasts with one/three steps; PostgreSQL, Databricks and Snowflake preparation-to-prediction.
- Regressions/checks: 56 adjacent Python cases, 132 Rust engine/request tests, seven contract checks, four wire checks, clippy, Python lint/types and repository hygiene.
- API ledger: 226 Python and 495 Java symbols; 53 ledger/contract tests passed with two environment skips.

Independent reviews covered both engine/host boundaries and final implementation.
Maintained tests and the fixture rationale are linked from the [end-time design](../phase-5/api-removal-audit/end-time/design.md), [examples design](../phase-5/api-removal-audit/expressions-tasks/design.md) and `tests/fixtures/custom-examples/README.md`.

## Final integrated artifacts

Release-mode Python wheel and Java native library were rebuilt from merge `1f7c191a9` with production features.
The wheel reports `TEST_OPS=False` and matches the recorded public API.
The combined wheel run passed 145 local checks; 12 skips cover opt-in live tests and unavailable/inapplicable test environments.
All six model cases then passed against that rebuilt wheel.
The rebuilt Java library passed 22 tests across end-time, SQL columns, supplied examples, labels and warehouse sessions, with no skips.
The final API ledger and contract-lock checks passed.

## Decisions and limits

Caller-supplied labels remain authoritative, including overlaps with prediction examples.
Native forecasting batches one prediction example per call and checks every expected row/step response.
The forecast horizon cannot exceed its context count.
The correct timeless timestamp replaces a reproduced old-client overflow; incomplete forecast responses are rejected explicitly.

SQL expressions remain source-backed, new-name projections over physical columns; they are not an in-memory expression language or mutable task editor.
Live-source persistence still refuses open connections, as before; memory-backed requests support persistence.
The separate deeper original-four coverage branch remains preserved and outside this goal.
Other partial mappings in the broader API audit are not implicitly declared resolved by these two efforts.
