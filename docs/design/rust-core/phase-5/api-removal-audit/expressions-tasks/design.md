<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Calculated columns and custom relational examples

**Proposal: restore two narrow workflows, rather than a general mutable task editor.**
Approved for implementation in Python and Java, with contract review and independent code review.

## What the two gaps mean

| Workflow | Intended use and old support | What is missing today |
|---|---|---|
| Calculated SQL column | `Table.add_column(s)` with `ColumnSpec(expr=...)`; for example, `line_total = unit_price_usd * quantity` | A public way to add that column to a source-backed schema without creating a warehouse view |
| Custom relational examples | `TaskTable` plus `predict_task()` accepted caller-supplied labels, entity IDs, timestamps and task type | Predict using externally supplied relational labels, rather than requiring labels derived from a PQL target |

Old tests demonstrate both examples above, including custom multiclass/link labels without PQL.
Both had source docstrings and tests; tracked Markdown/notebooks contained a security mention of expressions, but no usage tutorial.

Current semantic-view loading already supports some SQL expressions internally.
Current direct `Request(context, predict, ...)` is **Tabular-only**: Python explicitly refuses `related_tables`, and Java documents the same restriction.
Automatic batching already exists, but does not replace external relational labels.

## A. Add a calculated SQL column

Proposed shape in both languages: `schema.with_sql_column(table, name, expression)` / `withSqlColumn(...)`, returning a new schema.
Expressions are caller-supplied SQL in the source dialect.
Start with new names and expressions over existing physical columns; reject overwrites.
Providers validate against physical source columns; composing calculated aliases is initially unsupported.

```text
Python / Java: name + SQL expression
                    ↓
Rust: validate schema structure and retain a derived relation
                    ↓
Provider: describe/read SELECT unit_price_usd * quantity AS line_total
                    ↓
Existing Rust schema admission and prediction preparation
```

Rust owns column identity, type admission and key/link consistency.
Providers own SQL syntax and execution; Rust keeps SQL text opaque, as it does today.
Describe/preview the projection before admission; failures leave the original schema unchanged.
This creates a query projection, not a stored warehouse column.

Old support covered DuckDB, SQLite, PostgreSQL, Databricks and Snowflake.
Today only Databricks/Snowflake advertise derived-relation support, so the other three need provider projection work and tests.
This extends connector rendering, not the sampling algorithm.
In-memory expressions were explicitly unsupported by the old client: users should still calculate columns with pandas or Arrow before schema admission.
Destructive column editing remains outside this proposal; the old removal method could leave dangling links.

## B. Prepare a request from supplied examples

Proposed shape: `schema.materialize_examples(task, context, predict, ...)` / `materializeExamples(...)`, returning the existing `Request`.
Use an explicit task descriptor built from existing task/target concepts: task kind, entity table/key, target type and link destination where needed.
Review exact signatures before implementation; no PQL is required.

| Input | Columns and meaning |
|---|---|
| Context examples | `ENTITY`, `ANCHOR_TIMESTAMP`, `TARGET`: known labels for graph entities at specified times |
| Prediction examples | `ENTITY`, `ANCHOR_TIMESTAMP`: entities/times to score, without supplied answers |

These names reuse the new labels output layout.
Python adapts DataFrame/Arrow inputs; Java uses Arrow.
Users slice their own tables before admission, avoiding a mutable Rust `TaskTable`.

```text
Python / Java: explicit task + supplied example tables
                          ↓
Rust: validate IDs, anchors, label shapes and row alignment
                          ↓
Existing graph gathering → ordinary Request → existing predict()
```

Rust preserves supplied labels, row order and repeated entities at different anchors, validates references/types, and gathers related data.
Include binary/multiclass classification, regression and temporal link labels.
The old constructor also accepted forecasting with `step_size` and `num_forecasts`; define and verify the supplied-forecast row layout during contract review before implementation.
Query-derived forecasting does not replace supplied forecasting examples; retain this gap until verified or explicitly deferred.

## Checks before completion

Expressions: shared values/nulls/types, quoted names, invalid SQL, unchanged original schemas and persistence across all five SQL connectors; verify existing sampling results remain unchanged.
Custom examples: independent expected labels/IDs, scalar/link targets and the agreed supplied-forecast contract, row slicing/alignment, duplicate entities at different anchors, ownership, invalid inputs and persistence in both hosts.
Run memory/local SQL and real warehouse preparation, then a live relational prediction from custom examples.
Use focused old-client cases and independent answers; correct demonstrated old bugs.

## Verified evidence

Old source at `12955d397`: `rfm/base/table.py:225,321`, `base/sql_sampler.py:100`, `backend/local/table.py:152`, and `inference/task.py:20,125` under the retired engine's `rfm/` directory.
Old tests: `tests/backend/databricks/test_databricks_table.py:165`, `tests/test_task_table.py:48`, and `tests/test_rfm.py:1395,1435`.
Current boundaries: `sdm-contract/src/data.rs:243` (opaque derived SQL), `sdm-core/src/view.rs` (projection admission), Python `request.py:377` and Java `Request.java:39` (direct Tabular restriction).

## Test standard: parity or a verified improvement

Use maintained public-API tests with small shared fixtures and independently calculated answers.
Run focused old-client comparisons to establish semantics, not an exhaustive differential framework.
Investigate each discrepancy: preserve useful behavior, fix demonstrated bugs, and record the changed expectation with a regression test.
Verify real provider execution and relevant model predictions; mocks alone do not establish end-to-end support.
Finish with cross-language contract checks, independent implementation review and combined production-artifact validation.
