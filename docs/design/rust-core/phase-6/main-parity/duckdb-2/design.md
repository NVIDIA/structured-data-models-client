<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Complete DuckDB 2 support

**Gap:** installation still prohibits DuckDB 2, and the native client's Python provider uses a result-reading path that makes extra copies there.

## Applicability (source-audited)

| Dimension | Scope |
|---|---|
| Connector/language | **Python DuckDB only**, including `duckdb`/aggregate extras that pull that dependency. Java DuckDB uses JDBC, not the Python driver or this version cap. |
| Relational | DuckDB source description and push-down materialization use `DuckDBProvider`; the final-result read change applies here. SQL planning/neighbor policy remains Rust-owned and unchanged. |
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
Do not change the Java JDBC dependency solely because the Python package moved to version 2; Java needs only a no-regression check for shared contracts.

## Tests and acceptance

Test supported DuckDB 1.5 and 2.x through public `read_table()` and `Schema.from_source()` → materialize, with Arrow key attachments, empty tables, multiple result batches, timestamps and failure cleanup.
Exercise catalog/type discovery and actual multi-statement plans before lifting the cap.
Assert equal rows/schema and caller connection usability; warning wording is not a product contract.
Use a bounded large-result check to compare memory/time before and after, without promising zero-copy operation.
Finish with one source read → predict endpoint check and dependency-resolution checks for the connector extras.

Evidence: main `09ed6af4b`; connectors `pyproject.toml:67`, `reader.py:230`, and client `_rust/providers/duckdb.py:60`.
