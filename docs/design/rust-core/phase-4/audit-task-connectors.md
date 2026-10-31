<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Task, connector and original Phase 4 scope audit

Audit baseline: `cb64d1cca`, source inspection only; no builds or test execution by this reviewer.
Paths below are relative to `/home/awadaskar/code/sdmc-p4-core`.

## Verdict

The delivered common paths are materially broader than the original phase-three client, but “nothing remains in the original Phase 4” is not established.
The original phase table requires slice/switch gates and three named refusals removed; one remains for a non-built-in provider mode, and separate Phase 4 backlog items remain visible in code.
These are separate from ordinary warehouse string timestamps, which built-in providers support.
Do not turn the old task labels into new product requirements without reconciling accepted scope decisions.

## Prediction semantics matrix

“Implemented” below means reachable shared code exists, not that every connector ran successfully.

| Task | Kumo Relational | Kumo Tabular | Important distinction |
|---|---|---|---|
| Static node regression | Implemented, numerical entity column | Implemented, one-table target | A node is the entity being scored, not a separate task kind |
| Static node multiclass | Implemented, categorical/ID entity column | Implemented classification | Binary-vs-multiclass output follows task/target validation; do not infer both from a regression test |
| Binary condition | Implemented comparisons/IN/boolean conditions | Derived PQL target prohibited | Direct Tabular classification is supported; derived PQL conditions are a different capability |
| Temporal SUM/COUNT/AVG/MIN/MAX | Implemented over bounded related-row reads | Prohibited by model profile | Temporal comparisons produce binary tasks |
| Direct temporal link | Implemented LIST_DISTINCT foreign-key target | Prohibited | New warehouse fix retains item identity after sampling |
| Candidate ranking | Implemented through per-candidate classification | Foreign-key ranking prohibited | This is not the direct temporal-link endpoint path |
| Multi-step forecast | Implemented via per-step decomposition | Prohibited | Materialize helper's `problem_type` refusal is not proof that public forecasting is absent |
| “Static link prediction” | Needs precise definition before marking supported | No link task | Static FK target is classified as multiclass by `asked`; explicit link payload admission is the windowed LIST_DISTINCT path |
| EXPLAIN / EVALUATE | Prefixes accepted by materialization; output/server support requires separate evidence | Inspect model output capabilities separately | Acceptance is not proof of a supported endpoint output |

Source anchors: `rust/crates/sdm-core/src/materialize.rs:545-609` (target typing and prefix meaning), `:683-727` (list target), `:2739-2840` (public decomposition), `rust/crates/sdm-core/src/profiles.rs:97-106,250-295` (Tabular capability restrictions).
Scalar LIST_DISTINCT refusal at materialize.rs:647 still exists intentionally for scalar operands; it is not the now-enabled bare link target refusal.
Link statistics and link lags retain explicit refusals at materialize.rs:3304 and :4938.

## Connector / input interpretation

| Input/source | Python Relational | Java Relational | Tabular interpretation |
|---|---|---|---|
| In-memory | Shared Rust frame graph/materialization | Shared Rust Arrow graph/materialization | Supported ordinary in-memory rows, Python DataFrame/numeric inputs vs Java Arrow Table |
| SQLite / DuckDB | Built-in SQL providers | JDBC providers | Read rows first, then use in-memory Tabular; not source-backed Tabular materialization |
| PostgreSQL | Built-in SQL provider | JDBC provider | Same read-then-in-memory distinction |
| Databricks SQL | Built-in SQL provider | Databricks source | Same distinction; actual endpoint session support independent of connector |
| Databricks Spark | Python Spark bridge/source | No separate Spark source class found | Do not claim Java Spark merely because Databricks SQL works |
| Lakebase | Python PostgreSQL-family route | Requires checking Java source alias/API, not a separate source class | Do not count alias as independent execution evidence |
| Snowflake | Built-in SQL/Snowpark source | Explicit user-approved follow-up | Python can read rows, not source-backed Tabular |
| Databricks metric / Snowflake semantic views | Semantic metadata expands to physical-table graph | Check Java semantic options separately | Not a Tabular schema API |
| Files / S3 | Reader returns in-memory rows; then frame path | No inference from Arrow support to a shipped Java reader | Not a push-down connector |

Source anchors: connectors `source.py:254-264,292-495,495-579`; `reader.py:67-91`; `docs/design/rust-core/phase-4/tabular-source-support.md:3-5,49-71` records explicit decision against Tabular `Schema.from_source`.
Java concrete sources found: `DuckDbSource`, `SqliteSource`, `PostgresSource`, `DatabricksSource`.
No shipped `S3Source` class was found; matrix should list file ingestion as a path to frames rather than warehouse sampling.

## What existing tests actually establish

| Evidence inspected | Scope it can support | Cannot establish |
|---|---|---|
| Python `test_frame_materialize_core.py:44-233` | Static/temporal materialization, links/ranking, snapshots, composite IDs, float link precision, old/core comparisons | All classification families or all connectors |
| Python `test_warehouse_links_core.py` | DuckDB/SQLite direct temporal links, exact IDs, temporal labels/leakage, HTTP rankings/session, candidate-ranking regression | Postgres/Databricks/Snowflake live temporal links |
| Python `test_warehouse_sessions_core.py` | DuckDB/SQLite fit/score/close, holdout/temporal/API failures | Every connector and task session combination |
| Java `RelationalTest.java:144-340` | In-memory regression, temporal/link/rank/forecast, lifecycle/concurrency/holdout/errors, local endpoint stub | Live Java endpoint results; full in-memory binary/multiclass matrix |
| Java `WarehouseSessionTest.java:63-151` | Ten cases parameterized only over DuckDB/SQLite | Remote warehouse sessions |
| Java `TabularTest.java:101-260,288` | Direct/PQL/session/identity/Arrow ownership; live test gated by environment | Every classification dtype/connector |
| Python `test_tabular_pql_core.py:41-43,214-262` | Index/dtype families and explicit no-legacy Relational route guard | Whole-client no-legacy proof |
| Python `test_live_relational.py:35-44` | Environment-gated live tests exist | Their presence is not proof they ran; whole module still requires old engine extra |
| `contract/cases` inventory | S0=24, S1=291, S2=79 files | No S3 case directory found; independent Phase4 tests may cover S3 but original slice gate closure needs explicit evidence |

The latest paired warehouse live evidence uses DuckDB with real model endpoints.
It does not validate Databricks/Snowflake source reads merely because the model itself is hosted in Databricks.

## Original Phase 4 obligations still requiring reconciliation

1. **Named `text_time_rules` refusal remains.**
   `migration-plan.md:346,554` explicitly requires it gone; `materialize.rs:1192,1308` still refuses `TextTimes::ClientRules`.
   Built-ins use `ValueByValue`: `sdm-data/src/duckdb.rs:306`, `sqlite.rs:185`, `postgres.rs:147`, `databricks.rs:134`, `snowflake.rs:120`.
   Therefore ordinary built-in warehouse text timestamps are not missing, but the literal old exit needs implementation or an explicit scope decision.
2. **Large single-value Arrow handling remains constrained.**
   Original FX-3 at migration-plan.md:398,453 deferred >2GiB VARCHAR/BLOB handling to Phase4.
   `sdm-host/src/admission.rs:80-81,586-599` still narrows large variable-size arrays using int32 offset capacity and refuses overflow.
   This is a real surviving boundary limitation, not a requested performance benchmark; avoid allocating multi-GiB test data without resource planning.
3. **Fallback-provider backlog remains.**
   migration-plan.md:450-452 names fallback optional-operation behavior, slice-memory accounting and text collation.
   `sdm-core/src/fallback.rs:183-188` still accounts with `get_array_memory_size`.
   All five built-in provider declarations set `fallback: None`, so this is not a demonstrated ordinary connector bug.
   Decide whether the fallback provider ships before representing these as user-visible failures.
4. **Original slice/switch gate closure is not proved by the latest delivery summary.**
   The summary explicitly disclaims broader platform/performance acceptance.
   Record which approved restart gates replaced the original ones, or identify their missing execution evidence.
5. **Candidate type/inter-language boundary review still warranted.**
   Original P2-WR-5/P2-T2.14b-4 at migration-plan.md:420,431 concern spelled date/time candidates and equality.
   `sdm-request/src/ranking.rs:263-267` still maps direct typed binary/date/time scalars to `Candidate::Other`; spelling/typed wrappers exist.
   Java added exact numeric candidate tests, but that is not proof of Python/Java date/time candidate parity.

These findings do not justify removing Python host conversion or SQL-provider code: the audit must distinguish host work from retained legacy sampling/model policy.
