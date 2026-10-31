<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4: support, parity and remaining work

**There is still work left; the earlier completion claim covered delivered features, not full migration or language parity.**
Audit baseline: `cb64d1cca`, followed by audit-only changes.
The model/task and connector tables below are combined: a combination works only if both tables permit it.
“Code” means reachable implementation, not a passed end-to-end test on every connector.

## 1. Prediction matrix

A node is the row/entity being scored; regression and classification describe its target.

| Prediction | Python Relational | Java Relational, memory | Java Relational, warehouse | Tabular, Python / Java |
|---|---|---|---|---|
| Node regression | Code + local tests | Tested | Tested SQLite/DuckDB | Supported / supported |
| Node binary classification | Code + existing tests | Code + predicate tests | Existing local tests | Supported / audit tested |
| Node multiclass | Code + existing tests | Audit runtime passed | Existing local tests | Supported / audit tested |
| Temporal SUM/COUNT/etc. | Code + local tests | Tested | Tested SQLite/DuckDB | Not supported by model profile |
| Temporal comparison → binary | Code | Code | Tested | Not a Tabular PQL target |
| Direct temporal links (`LIST_DISTINCT`) | Tested local + live model | Tested | Audit runtime passed SQLite/DuckDB | Not supported |
| Rank explicit candidates | Code + local tests | Tested | **Gap: explicit candidates are not forwarded** | Not supported |
| Multi-step forecast | Code + existing tests | Tested | **Gap: decomposed requests refused** | Not supported |
| Static foreign-key target | Multiclass path | Shared multiclass path | Shared multiclass path | Ordinary single-column classification only |
| Separate “static link” endpoint task | Not demonstrated; do not confuse with FK classification | Same | Same | Not supported |

Java warehouse ranking currently reports `candidates_missing` even when the caller supplied candidates.
Forecasting reports `in_parts`; both were reproduced on SQLite and DuckDB.
EXPLAIN/EVALUATE syntax and requested output fields are separate from prediction task support; endpoint output availability is not established by parsing success.

## 2. Connector × model × language

| Data input | Python Relational | Java Relational | Python Tabular | Java Tabular |
|---|---|---|---|---|
| In-memory | DataFrame graph | Arrow Table graph | DataFrame / numeric direct input | Arrow Table |
| DuckDB | SQL provider; local tests | JDBC; local tests | Read rows first | Application supplies Arrow rows |
| SQLite | SQL provider; local tests | JDBC; local tests | Read rows first | Application supplies Arrow rows |
| PostgreSQL | SQL provider; code | JDBC; code, gated DB tests | Read rows first | Application supplies Arrow rows |
| Lakebase | PostgreSQL route + auth helper | Caller supplies PostgreSQL JDBC connection; unverified as Lakebase | Read rows first | Application supplies Arrow rows |
| Databricks SQL | SQL provider; code | JDBC provider; code | Read rows first | Application supplies Arrow rows |
| Databricks Spark | Python source bridge; code | No Spark-session source | Read rows first | Application supplies Arrow rows |
| Databricks metric view | Semantic metadata → physical graph | Semantic source options; code | Read rows first, not semantic-schema support | Application supplies Arrow rows |
| Snowflake | SQL/Snowpark source; code | **Explicit follow-up** | Read rows first | Application supplies Arrow rows |
| Snowflake semantic view | Semantic metadata → physical graph | **Explicit follow-up** | Read rows first, not semantic-schema support | Application supplies Arrow rows |
| Files / S3 | Reader → in-memory path | No shipped reader; application Arrow input | Reader → DataFrame path | No shipped reader; application Arrow input |

Tabular deliberately does **not** accept source-backed `Schema.from_source`/`fromSource` materialization.
Reading one table into memory first is supported input preparation, not warehouse push-down Tabular sampling.
For Spark, Snowpark and semantic sources this means application/platform conversion, not a universal `read_table()` helper; that helper expects a cursor-style connection.
Remote provider code and past gated tests are not fresh validation of this whole task matrix.
The latest live warehouse comparisons used DuckDB data plus a real model hosted in Databricks; they did not test Databricks source reads.

## 3. Lifecycle and API parity

| Capability | Python | Java | Assessment |
|---|---|---|---|
| Schema → materialize → predict | Available | Available | Task/connector restrictions above apply |
| Fit → repeated predict → close | Tabular + Relational, memory + warehouse | Same paths subject to task/connector restrictions | Both use shared Rust rules; rewritten rank/forecast fit is refused |
| Model default on multiple tables/relationships | Selects Relational | Requires explicit Relational for memory | **Behavior differs** |
| Direct-input configuration | Schema/entity/inference/task options | Target/task only in public constructor | **Java surface is smaller** |
| Tabular PQL task override | `task=` | No equivalent option | **Parity gap** |
| Schema editing | Mutable properties | Immutable `with*` methods | Appropriate host API difference; Python frame rules remain legacy |
| Partial prediction failure | Retains completed result | `answered(allocator)` | Both supported |
| Result presentation | Pandas/items | ArrowReader | Appropriate language difference |
| Result merge / unscored candidates | Public helpers/access | No equivalent public helpers found | Java gap; decide required scope |
| Health/model listing, source read helper | Available | Smaller capability API; no read helper | Java convenience gap |
| Request/schema persistence | Python pickle behavior | No public serialization API | Not equivalent |
| Async API | None found | None found | No demonstrated asymmetry |

The data connector is independent of the model transport.
Generic HTTP sessions require a session-capable server; built-in Databricks Model Serving transport does not expose sessions in either language.
The validation bridge does not change that public API limitation.

## 4. Is old Python still running?

**Yes, although the tested sampling/prediction/session computations use Rust.**

```text
In-memory PQL → Rust validation → OLD Python validator → Rust materialization
Warehouse PQL → Rust validation ───────────────────────→ Rust materialization
Frame schema edit → Python schema rules
Prediction / fitted sessions → Rust policy → host HTTP → Rust answers
```

| Route evidence | Result |
|---|---|
| Default-route runtime tripwires | 52 exercised test bodies plus nine routing checks passed without trapped legacy computation |
| Actual Rust host execution | Recorded in 52 of those tests; others include early refusals/routing checks |
| Legacy PQL validation | Recorded in 15 passing tests; separate public probes reproduced it for both in-memory models |
| Trap positive controls | 11 comparison cases deliberately forced the old route and were caught |
| Frame schema edits | Runtime stype edit: Python rules for frames, Rust edit for DuckDB; source trace also identifies key/time/relationship rules |
| Explicit override | Empty `SDM_CORE_ROUTES` selects old execution; unknown source kinds also select legacy paths |
| Automatic retry fallback | None found in inspected Rust materialize/predict/session refusal handling |
| Java | JNI/shared Rust execution; no Java-to-Python fallback found |

Python input observations, DataFrame construction, warnings, HTTP and result presentation are expected host work.
A method located in an old adapter module is not automatically old computation; the audit traps specific computational entrypoints.
Tests that change routing internally are classified separately from default-route proof.
The runner traces the main-thread test body, not fixture setup/teardown or worker-thread activity; these counts are bounded evidence, not whole-process proof.

## 5. What remains

| Priority | Finding | Proposed next action |
|---|---|---|
| 1 | Old PQL shadow validation on ordinary memory paths | Return Rust-resolved query facts; remove dependency with public acceptance/error regressions |
| 1 | Java warehouse ranking/forecast gaps | Pass candidates and retain decomposed requests using existing shared Rust operations |
| 2 | Python frame schema-edit policy remains | Reconcile Rust edit contract and source facts; migrate rules without adding callbacks |
| 2 | Java model-default and configuration differences | Agree which are intentional; implement promised parity with small API tests |
| 2 | Remote connector/task coverage is incomplete | Run a targeted PostgreSQL/Databricks/Snowflake matrix with actual source connections; report credentials/environment limits |
| Scope decision | Original `ClientRules` time mode, >2GiB Arrow value limit, fallback-provider backlog | Explicitly close or defer original obligations; ordinary built-in string timestamps already work |
| Acceptance evidence | Original slice/switch gates are not established by latest summary | Map replacement restart checks to original exits or run missing gates |
| Accepted follow-up | Java Snowflake | Keep outside current Java delivery |

Do not remove legacy packages merely because common routes run in Rust.
No product API or sampling SQL has been changed by this audit.

## Evidence and audit status

Python tripwire runner: [`phase4_route_audit.py`](../../../../scripts/rust_core/phase4_route_audit.py).
Java runtime cases: [`Phase4CoverageAuditTest.java`](../../../../java/src/test/java/com/nvidia/sdm/Phase4CoverageAuditTest.java).
Per-case results and artifact identities: [runtime evidence](audit-runtime-results.json).
Source/coverage details: [Python routes](audit-python-routes.md), [Java API parity](audit-java-api.md), [task/connector and original scope](audit-task-connectors.md).
Audit test logs and function traces are retained under the local orchestration `phase-4/coverage-audit` evidence directory.

Java audit: nine cases passed, comprising five supported journeys and four assertions of known refusals.
Two independent source/evidence reviewers checked the synthesis; their scope and trace-limit corrections are incorporated.
The audit identifies remaining work; it does not claim the implementation gaps have been fixed.
