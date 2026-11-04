<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sample PostgreSQL JSON/XML rows reliably

**Gap:** keyless neighbor tables can generate an `ORDER BY` that PostgreSQL cannot execute.
The affected columns include JSON, XML and domains wrapping those types.

## Approved correction, in plain terms

A customer's two notes have the same timestamp and no unique note ID, but different JSON bodies.
To keep only one neighbor consistently, sampling needs a tie-break after the timestamp.
PostgreSQL cannot sort its raw `json` or `xml` values, so the current generated query can fail.
The old fix effectively uses `ORDER BY timestamp DESC, CAST(body AS TEXT)` for that string-like tie term.
This changes how tied rows are selected; it does not modify stored JSON, flatten tables or move sampling into Python.
Ordinary keyed tables, other database dialects and Tabular prediction do not need this correction.

## Applicability (source-audited)

| Dimension | Scope |
|---|---|
| Model/task | Kumo Relational whenever neighbor gathering encounters a keyless table with affected columns, across supported node, temporal-link and forecast/decomposed tasks. Trigger is the sampled table, not the prediction task name. |
| Connectors | **PostgreSQL and PostgreSQL-backed Lakebase**, through the PostgreSQL dialect. Not Databricks SQL/Spark, Snowflake, DuckDB or SQLite. Lakebase is distinct from a Databricks SQL warehouse. |
| Sampling | Directly affects **push-down neighbor ordering**. Does not change in-memory sorting or plain `read_table()` SQL. Loading the database table into memory first avoids this particular SQL path. |
| Languages/models excluded | Python and Java share the fix, including Java using a PostgreSQL connection to Lakebase. Tabular has no relational neighbor sampling and is unaffected. |

The typed tie-term contract is shared with other dialects, so those need no-regression tests even though only PostgreSQL should render the new cast.
Source evidence: `sdm-engine::sample::tie_break`, PostgreSQL `order_term` use, and host PostgreSQL provider/source routing.

## Why users need it

A customer's notes table may have no primary key and contain a timestamp plus JSON/XML content.
When several notes share a timestamp, sampling needs a stable secondary order to choose the bounded set of neighbors.
Skipping the content columns makes that choice ambiguous; ordering raw JSON/XML can fail.

| Question | Finding |
|---|---|
| Did main support it? | Yes. `6aa25db22` orders string-typed tie-break columns by `CAST(column AS text)`, including domains. It corrected an earlier fix that simply omitted JSON/XML. |
| Was it documented? | The historical changelog describes the failure and fix; PostgreSQL engine tests include JSON, XML and a JSON domain. No dedicated end-user notebook/example was found for this edge case. |
| What is missing here? | Rust tie-break terms lose string-type information and the PostgreSQL SQL path orders the original value. Output-transfer casts do not fix the `ORDER BY`. |

Reproduced against PostgreSQL through public `Schema.from_source` → PQL materialization before the fix: PostgreSQL rejected `ORDER BY` on a JSON column.

## Small design

```text
Rust sampling: choose deterministic tie-break columns with their types
    → Rust PostgreSQL dialect: render string terms as CAST(column AS text)
    → Python / Java: execute SQL and return Arrow rows
```

Carry the minimum semantic type needed from `sdm-engine` through the existing tie-term contract to `sdm-data`.
PostgreSQL owns the dialect-specific cast; the shared planner must not embed PostgreSQL syntax.
If that internal contract changes, update its serialization/fixtures together rather than guessing types in a host adapter.
Keep timestamp direction, null placement, quoting and other connector behavior unchanged.
Do not cast numeric columns to text, drop troublesome columns, or fetch the whole table into Python to sort it.
Identical rows may still tie; do not promise a stable hidden physical-row identity or identical collation across databases.

## Tests and acceptance

First reproduce via public `Schema.from_source` → PQL materialization against PostgreSQL, using a keyless related table with equal timestamps and JSON/XML/domain values.
Then check rendered SQL and actual selected rows through both Python and Java, including nulls, quoted identifiers and a numeric control column.
Repeat bounded selection and compare expected textual ordering under the database's collation.
Confirm other dialect SQL and ordinary keyed-table selection remain unchanged.
One read → materialize → predict smoke test verifies these rows reach the model without test-specific cleanup.
Inspect the query plan on a representative fixture; the cast is needed for correctness, but no speedup is assumed.

Evidence: main `6aa25db22`; current `sdm-engine/src/sample.rs:410`, `sdm-contract/src/data.rs` (`TieTerm`) and `sdm-data/src/sql.rs:332`.

## Implementation and verification

Rust now carries a `string_like` fact on keyless tie terms (internal contract 68).
Only the PostgreSQL renderer uses it; Python and Java still execute the generated SQL without sorting rows themselves.

| Check | Result |
|---|---|
| Rust | 45 data unit tests, 2 dialect-matrix tests, 3 PostgreSQL golden tests and the engine type-boundary regression passed; Clippy passed. |
| Python public API | JSON, XML and JSON-domain fixtures passed against PostgreSQL, including numeric 2 before 10, equal/older/null timestamps, null bodies, repeated bounded selection and a quoted column name. |
| Java public API | The same JSON/XML/domain ordering fixture and live prediction passed, plus all 7 existing PostgreSQL read controls (9 tests total). |
| Real model | PostgreSQL read → materialize → predict passed for both requested IDs in Python and Java against the same deployed relational model. |
| Existing PostgreSQL controls | 15 source/key tests passed with the production wheel. |
| CI | New Python/Java regressions are included in existing PostgreSQL-service jobs; 54 workflow checks passed. |

The actual generated query's `EXPLAIN ANALYZE` retained per-seed limits and database sorting; no host-side whole-table fetch was added.
The small fixture is a correctness check, not a performance benchmark.
Regenerating the dialect matrix corrected three pre-existing capability rows (`derived_relations`); existing SQL snapshots did not change.
The same capability mismatch was verified against the parent implementation, independently of the new string-type field.
Production wheel: commit `347de10fc`, default production features.
Independent review found no material boundary or correctness issue.
