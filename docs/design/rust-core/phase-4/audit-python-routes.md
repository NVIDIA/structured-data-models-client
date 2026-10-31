<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Python route audit

Audited source: `/home/awadaskar/code/sdmc-p4-core`, baseline `cb64d1cca` plus documentation-only audit work.
This is code-trace evidence; no runtime suite was executed by this reviewer.
Paths below are relative to `packages/structured-data-models-client/src/sdm_client/` unless qualified.

## Verdict

Default sampling, prediction and fitted-session dispatch selects Rust for the shipped warehouse source kinds and in-memory inputs.
There is no catch-and-retry-on-old-engine fallback in the inspected Rust materialize/predict/session paths.
However, “the old client is not actively used” is too broad: source-less PQL still executes the old Python validator, and in-memory schema edits still execute Python schema policy.
Python also retains explicit route overrides, legacy routes for unknown source kinds, and some pre-dispatch validation/presentation policy.

## Default route matrix

| Entry point/input | Actual dispatch | Caveat |
|---|---|---|
| `Schema.from_data` | `schema.py:1105` → `_rust.frame_schema.from_data` | Native dataframe observations precede Rust assembly; later edits are Python, below |
| `Schema.from_source`, supported warehouse | `schema.py:1274` → `_rust.schema.from_source` | Unknown source kind takes `EngineTables` at1293 |
| Relational materialize, supported warehouse | `query.py:248` → wrapper `RelationalMaterializer.materialize` → `_rust.materialize.gather` at `materializers/relational.py:303` | Wrapper name is not evidence of legacy sampling |
| Relational materialize, in-memory | Same wrapper → `_rust.frame_materialize.gather` at `materializers/relational.py:296` | Old PQL shadow validation still executes first |
| Tabular PQL, in-memory | `query.py:293` → `_rust.tabular_materialize.materialize` | Old PQL shadow validation; frame observations and native row rebuilding remain Python |
| Tabular from-source schema | Refused at `query.py:215` before materialization | Reading a source into a dataframe first is a different supported input route |
| Direct `Request(context,predict,...)` | `request.py:374` → `_rust.direct.construct` | Public Python object/conversion layer remains |
| Predict, direct dataframe request | `client.py:910` → `_rust.direct` | Explicit old-route override bypasses this |
| Predict, materialized supported-source/in-memory request | `client.py:920` → `_rust.predict` | Core prediction can also consume manually prepared payloads |
| Ranking/forecast predict | `client.py:516` chooses core from first part and sends whole decomposition to `_rust.predict` | Non-core source/explicit old route uses Python `_ranked`/`_by_step` |
| Fit, Tabular | `client.py:750` → `_rust.sessions.fit` | Rewritten ranking/forecast requests are refused before dispatch at747 |
| Fit, Relational frames/warehouse | `client.py:757` → `_rust.frame_sessions.fit` | Unknown source kind/forced old route uses adapter session policy at769 |
| Fitted predict/close | `client.py:196` / `client.py:246` inspect `_relational_session` or `_tabular_session` | Existing old fitted objects retain old behavior; environment changes do not turn a new fitted object into an old one |

Supported warehouse route-map kinds are DuckDB, SQLite, PostgreSQL, Databricks SQL, Databricks Spark, Databricks metric view, Snowflake and Snowflake semantic view (`_routing.py:382–420`).
Lakebase is a PostgreSQL connection wrapped by `PostgresSource`, whose `kind` is `postgres`; absence of a separate `lakebase` key in materialize/predict maps does not itself indicate a fallback (`sdm_connectors/source.py:248,428`; `sdm_connectors/databricks.py:178`).
S3/file `read_table` is ingestion into host data, not an extra relational SQL-provider kind.

## Confirmed active Python policy

### 1. Source-less PQL invokes the old validator

`query.py:719` defines `gathered_on_the_core` using only `_routing.on_the_core(schema.source, MATERIALIZE)`.
For `source=None`, this is false even when in-memory sampling will run in Rust.
After Rust validation succeeds, `query.py:760` calls `_rust.s0.shadow`.
`_rust/s0.py:120–124` imports and calls `predictive_query.validate` to create the old resolved-query object.
If Rust accepts and the old validator rejects, it raises `INTERNAL_ERROR` rather than using Rust's accepted query.
This occurs for both in-memory Relational PQL and Tabular PQL; supported warehouse Relational returns `CoreQuery` instead at `query.py:738`.
It is an active old-language dependency, not merely import overhead or HTTP presentation.

### 2. In-memory schema edits use Python rules

`_rust/frame_schema.py:128` clears every rebound column's `_source` facts.
`schema.py:1980–1998` recognizes core-editable schemas only by those facts.
Consequently, normal frame schemas use Python branches for semantic-type, primary/time-key and relationship edits (`schema.py:196–209,414,948,2812`).
For example, setting `schema['users']['target'].stype` reaches `_check_stype`, `_check_supports_dtype`, and key/time reconciliation in Python.
Warehouse schema edits with retained source facts route to `_rust.edits`.
Manually constructed schema objects and schemas whose source facts are discarded also follow Python edit policy.
This is a policy-boundary gap relative to a blanket “all schema rules live in Rust” claim; whether every edit was required by the accepted Phase 4 slice needs reconciliation with the original plan.

### 3. Native host work and remaining pre-dispatch rules

DataFrame equality/hash/dtype/key-quality observations, frame reconstruction, warning/error presentation, request-ID generation and transport execution are still Python.
Some are intentional boundary responsibilities, not legacy sampling.
`_rust/tabular_materialize.py:22` imports `_rows`, `_asking`, `_is_whole` from the old materializer module; these are row access, query recording and numeric observation helpers, not `TabularMaterializer.indices/table` execution.
Its ordinary row selection/sampling/order invokes Rust (`select_tabular_rows`, `sample_tabular_rows`, `order_sampled_tabular_rows`), while exceptional native masking remains Python at179–204 and525–548.
Custom sorting methods are explicitly refused at270–273; there is no old-sorting fallback there.
`client.fit` still refuses rewritten questions and unsupported transport sessions before Rust dispatch (`client.py:738–747`).
Adapter `explain_error`, `_keys_the_caller_set`, `_request_id` and transport close can execute on new routes; these alone do not prove old prediction/session policy ran.

## Explicit remaining old routes

`_routing.py:552–569` treats an explicitly empty `SDM_CORE_ROUTES` as all-old.
A nonempty override sends only selected paths to core and all other paths to old; split operation groups raise `RouteError`.
With the environment absent, `CORE_ROUTED` defaults are used (`_routing.py:472–518`).
Unknown source kinds return false from `on_the_core` (`_routing.py:584–595`) and use the legacy engine/adapter branches.
Do not call these implicit retry fallbacks: they are dispatch selections, before execution.
Schema construct/inspect/pickle paths listed as old in `ROUTES` are not by themselves proof that each accessor bypasses Rust; trace their actual implementations and source-fact inheritance.

## Runtime tripwires

Use ordinary default environment, removing `SDM_CORE_ROUTES` before each case.
Trap these to fail immediately if legacy computational policy is reached:

| Trap | Expected result |
|---|---|
| `predictive_query.validate` | Fails source-less PQL today; should not run for normal warehouse Relational PQL |
| `RelationalMaterializer._gather`, `_get_model`, `_build_graph` | Should never run on supported default materialization paths |
| `TabularMaterializer.indices`, `.table` | Should never run on ordinary default Tabular materialization |
| `RelationalAdapter` / `TabularAdapter`: `prediction_plan`, `session_plan`, `session_prediction_plan`, `normalize` | Should never run on new default predict/session paths |
| `client._held_back`, `_ranked`, `_by_step` | Should never implement holdout/decomposition policy on new default paths |
| `Column._check_supports_dtype` after frame schema construction | Expected to run for frame stype edit; pair with a trace of `_rust.edits.column` to expose the distinction |

Do not trap the `RelationalMaterializer.materialize` dispatcher, adapter constructors/diagnostics, or public result conversion indiscriminately.
Suggested public journeys: in-memory and DuckDB/SQLite scalar PQL; Tabular direct/PQL; frame and warehouse ranking/forecast; temporal links; fit/holdout/two scores/close; frame and warehouse schema edits.
Record native opcodes and the trap log alongside results, rather than merely asserting `_core` metadata exists.
