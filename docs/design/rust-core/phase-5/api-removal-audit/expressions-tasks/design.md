<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Calculated columns and custom relational examples

**Proposal: restore two narrow workflows, rather than a general mutable task editor.**
Both have approved Python and Java boundaries; implementation is in progress.

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
Tables carry typed calculated-column declarations separately from their logical source identity.
On refresh Rust resolves the current physical source/view, preserves its filter and validates its original columns, then reapplies only the recorded caller additions.
Hosts retain this metadata mechanically; a resolved SQL projection is not a second editable definition.
SQLite refuses a calculated alias named `rowid`, which would shadow its existing sampling identity.
Existing live-source schemas and their prepared requests cannot be persisted because they retain open connections (verified unchanged on contract 63); this work does not serialize credentials or connections.

Old support covered DuckDB, SQLite, PostgreSQL, Databricks and Snowflake.
Today only Databricks/Snowflake advertise derived-relation support, so the other three need provider projection work and tests.
This extends connector rendering, not the sampling algorithm.
In-memory expressions were explicitly unsupported by the old client: users should still calculate columns with pandas or Arrow before schema admission.
Destructive column editing remains outside this proposal; the old removal method could leave dangling links.

## B. Prepare a request from supplied examples

Proposed shape: `schema.materialize_examples(task, context, predict, ...)` / `materializeExamples(...)`, returning the existing `Request`.
Use an explicit task descriptor built from existing task/target concepts: task kind, entity table/key, target type and link destination where needed.
No PQL is required.
Use immutable `RelationalTask(kind, entity_table, link_target_table=None, step_size_ns=None, num_forecasts=None)` and `materialize_examples(task, context, predict, *, num_neighbors=None, random_seed=42, inference=None)`.
Rust owns task meaning; target physical types come from Arrow, never guessed task kinds from values.

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
Supplied forecasting uses numeric context labels, valid anchors, positive int64 `step_size_ns` and positive `num_forecasts`, checked for range/size.
The old client converted offsets to nanoseconds and sent wire `step_size`; keep that wire meaning and verify forecast-step response alignment against the endpoint.
Static tasks may omit anchors: Rust uses the graph’s latest start time, or the existing timeless sentinel when no start-time bounds exist; never wall-clock time.
Temporal/forecast tasks require valid anchors.
Supplied examples retain their labels and order even when context and prediction IDs overlap; the PQL-specific answer-removal rule does not apply.
Reject invalid IDs/labels rather than silently dropping rows.

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

## SQL implementation checkpoint (contract 64)

Python and Java local SQLite/DuckDB checks pass, including values/nulls, repeated refresh, original-schema ownership, invalid expressions, quoted names and source changes.
A filtered projection with a renamed UNIQUE-index key verifies alias-preserving discovery on both local SQL providers.
PostgreSQL, Databricks SQL, Snowflake, Databricks metric views and Snowflake semantic views each produced the independently expected 24 labels and two live predictions.
The old DuckDB calculated-column path produced the same 29 expected non-null fixture values and one null.
The independent boundary review found no blocker; custom examples remain the next checkpoint.
