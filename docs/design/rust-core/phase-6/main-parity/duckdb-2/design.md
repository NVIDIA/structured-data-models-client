<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Complete DuckDB 2 support

**Gap:** installation still prohibits DuckDB 2, and the native client's Python provider uses a result-reading path that makes extra copies there.

## Approved version policy and SQL checks

Fetched main `847a1e9e3` declares `duckdb>=1.5.0`: it allows both 1.5 and 2, rather than requiring 2.
The cached DuckDB 2 wheel declares Python >=3.11; our client/connectors still declare Python >=3.10.
Use DuckDB 2.x as the primary target and retain DuckDB 1.5 compatibility for Python 3.10.
Do not silently drop Python 3.10 support or introduce separate sampling implementations for the two versions.
Java already tests DuckDB JDBC 2.0.0.0.

This is more than an Arrow-read patch: audit generated SQL and provider lifecycle on both drivers before changing support claims.

| Area | Concrete check |
|---|---|
| Expressions | Regex matching changes; old lambda arrows are disabled; integer modulo/division by zero now errors. Audit calculated expressions and filters, not just sampling SQL. |
| Catalog/sampling | Exercise keys/FKs, quoted names, null/type discovery, reservoir sampling, seeded keys and window/neighbor queries. |
| Ownership | Test failed transactions, concurrent cursors, pending results and Arrow batch boundaries; never silently change a borrowed connection's compatibility settings. |
| Files | New 2.0 files are not readable by 1.x by default; test existing-file access without upgrading user storage implicitly. |

Local probes confirmed regex/modulo/lambda differences between 1.5.6 and 2.0.0; simple catalog column indices and seeded reservoir results matched.
That is narrow evidence, not a complete SQL compatibility verdict.
[DuckDB SQL migration](https://duckdb.org/docs/2.0/operations_manual/migration) and [Python migration](https://duckdb.org/docs/2.0/clients/python/migration) are the upstream checklist.
If shared Rust SQL needs correction, test it through Python and Java; Python-only driver changes remain host-owned.

## Applicability (source-audited)

| Dimension | Scope |
|---|---|
| Connector/language | Dependency/provider changes apply to Python DuckDB, including aggregate extras. Shared SQL compatibility checks apply to Python and Java; Java JDBC does not use the Python version cap. |
| Relational | DuckDB source description and push-down materialization use `DuckDBProvider`; the final-result read change applies here. SQL planning/neighbor policy remains Rust-owned; change it only for a demonstrated compatibility defect. |
| Tabular | Can consume a DuckDB table after `read_table()` loads it into memory. That reader already uses relations; removing the dependency cap enables installation, but the provider optimization does not change Tabular execution. No Tabular `Schema.from_source()` support is added. |
| Other input paths | Other databases and native file/S3 readers are unaffected. If a user explicitly exposes file/S3 data through a DuckDB connection/view, its DuckDB query result does use this provider. In-memory prediction itself is unaffected. |
| Tasks | Provider result reading is task-independent, for tasks already supported by the selected model/path; it adds no prediction task. |

Source evidence: connector `reader.py` dispatches DuckDB separately; file reads use their existing pandas/Arrow paths, and Tabular's `_rows` rejects source-backed tables rather than doing push-down sampling.

## Why users need it

A user with DuckDB 2 should be able to install the connector and predict from an existing DuckDB table.
This should not require downgrading their database library or repeatedly copying query results unnecessarily.

| Question | Finding |
|---|---|
| Did main support it? | Yes. `09ed6af4b` removed the DuckDB<2 hold and switched result reads to DuckDB relations. |
| Was it documented? | Yes, the changelog describes DuckDB 2/result-copy compatibility. General source/reader examples cover usage; no separate version-2 API was introduced. |
| What is missing here? | Connector dependencies still require `<2`. `read_table()` already uses relations, but `DuckDBProvider` uses cursor Arrow output. |

Audit probes with cached DuckDB 2 and pandas 3.1 successfully materialized both memory and DuckDB inputs, while DuckDB reported extra copies.
That proves a narrow working path, not complete version-2 support or a measured memory improvement.
The old pandas zero-offset failure is already avoided by Rust time arithmetic; do not port that Python helper back.

## Small design

```text
Rust: produce SQL and Arrow key attachments
    → Python provider: register keys, execute setup, read final query as a relation
    → Arrow result back to Rust
```

This is a **Python driver adapter fix**, not Rust sampling policy.
Keep an isolated cursor, registered Arrow key relations, statement order, full results and empty-result schema.
Use a relation for the final result-producing query without accidentally executing setup twice or dropping earlier statements.
Keep the borrowed connection open and clean up temporary cursor resources on failure.
Then remove the dependency cap and regenerate the lock using the normal resolver.
Java already targets JDBC2; run shared SQL compatibility cases there too, while keeping Python driver adaptation separate.

## Tests and acceptance

Test supported DuckDB 1.5 and 2.x through public `read_table()` and `Schema.from_source()` → materialize, with Arrow key attachments, empty tables, multiple result batches, timestamps and failure cleanup.
Exercise catalog/type discovery and actual multi-statement plans before lifting the cap.
Assert equal rows/schema and caller connection usability; warning wording is not a product contract.
Use a bounded large-result check to compare memory/time before and after, without promising zero-copy operation.
Finish with one source read → predict endpoint check and dependency-resolution checks for the connector extras.

Evidence: main `09ed6af4b`; connectors `pyproject.toml:67`, `reader.py:230`, and client `_rust/providers/duckdb.py:60`.

## Implemented and verified

Python executes setup statements once, then consumes the final relation before its isolated cursor closes.
No Rust SQL, public API, connection setting or storage format was changed.
Connector extras now select DuckDB 1.5 on Python 3.10 and DuckDB 2 on Python 3.11+; this deliberately narrows main's unrestricted `>=1.5` policy.
Existing Python CI version jobs exercise both supported combinations; Python 3.12 contract jobs now pin 2.0 instead of conflicting with the new requirement.

| Evidence | Result |
|---|---|
| Public source/reader, calculated columns, temporal links, end times, provider ownership and SQL suites | 252 passed, 4 optional skips on **each** real DuckDB 1.5.6 and 2.0.0. |
| New driver regressions | Four tests cover Arrow attachment lifetime, exact unsigned/decimal values, empty results, multiple batches, pending parent results, transaction failure cleanup and concurrent cursors. |
| Minimum runtime | Actual Python 3.10.22 connector-extra installation selected 1.5.6; all four new tests passed. |
| Java JDBC 2 | Nine large-value, streaming, calculated-column and temporal-link tests passed. |
| Live model prediction | Maintained shop `node.regression` journey passed through both Relational source sampling and Tabular `read_table()` on both DuckDB versions, with legacy routing prohibited. Existing model endpoints were reused. |
| Existing storage | A database created by 1.5 remained readable by 1.5 after 2.0 read and inserted a row. No compatibility setting was changed. |
| Bounded performance check | Fresh-process 250,000-row Arrow read: old DB-API 0.134 s / 143,284 KiB peak RSS; relation 0.092 s / 120,232 KiB. One local observation, not a general speed or zero-copy guarantee. |

The Python/Java checks used the existing native build `e74b32178699`; this patch changes only the Python driver adapter and dependency selection.
DuckDB's changed regex, lambda and division/modulo semantics still apply to user-written SQL expressions.
We neither rewrite those expressions nor emulate 1.5 settings on borrowed connections.
No incompatibility requiring a shared Rust SQL change was found in these checks.
