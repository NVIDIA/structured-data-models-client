<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Java/Python API and coverage audit

**Historical audit at `cb64d1cca`; not current open-work status.**
See the [reconciled findings](coverage-audit.md) and [acceptance map](acceptance-map.md).

Source baseline: `cb64d1cca`, inspected in `sdmc-p4-core`.
This is source analysis plus inspection of existing test reports; no new tests or builds ran for this audit.
Paths below are relative to that checkout.

## Main findings

1. **Java warehouse ranking/forecast is incomplete.**
   `PredictiveQuery.java:181` rejects `state.inParts()` with `UNSUPPORTED_REQUEST` even though the in-memory Relational branch retains `FrameRequest` parts.
   The warehouse branch also never forwards `MaterializeOptions.candidates()`; only `RelationalMaterializer.java:62` does.
   Do not transfer Java in-memory ranking/forecast success to JDBC sources.
2. **Model defaults differ.**
   Python selects Relational automatically for multiple tables or relationships (`query.py:145`), while Java requires explicit Relational selection or a `RelationalMaterializer`; otherwise multiple tables fail as Tabular (`PredictiveQuery.java:119-130`).
   A one-table self-relationship also needs checking because the Java default branch does not select Relational from relationships.
3. **Java exposes a smaller direct-input API.**
   Python `Request` accepts schema/entity-name/inference/task-options overrides (`request.py:332`); Java's only public constructor takes two Arrow Tables, target and task (`Request.java:30`).
   Java Tabular PQL has no task override and rejects every generic materialization option (`PredictiveQuery.java:134`); Python has `task=` and `**options` (`query.py:126`).
4. **Java lacks several convenience/introspection APIs.**
   No public `models()`, `health_ready()`, generic `read_table()`, result merge, request-parts inspection, or serialized request/schema API was found.
   Python provides these capabilities or Python pickle support.
   These are parity differences, not automatically Phase 4 blockers.
5. **Java does not invoke legacy Python.**
   All inspected public preparation/prediction/session paths call `Host.run`, then `Run.start`, then JNI `NativeCore.start` (`internal/Host.java:311`, `internal/Run.java:141-146`).
   There is no Java-to-Python fallback path in these entrypoints.
   JDBC and HTTP conversion remain Java host work.

## Connector/model matrix

`Code` means reachable implementation, not a successfully executed test for every task.
Warehouse rows must first be read into an Arrow Table by application code to use Java Tabular; `Schema.fromSource` is Relational-only.

| Input/source | Java Relational | Java Tabular | Evidence/limit |
|---|---|---|---|
| Arrow in-memory | Code; explicit model or retained sampler | Code; direct request and single-table PQL | `Schema.java:63`, `RelationalMaterializer.java:31`, `PredictiveQuery.java:77,119` |
| DuckDB JDBC | Code; scalar/temporal tasks, sessions; parts refused | No source-backed Tabular route | `Source.java:18`, `PredictiveQuery.java:124,181`; local JDBC tests |
| SQLite JDBC | Same | Same | Same; local JDBC tests |
| PostgreSQL JDBC | Same code path | Same | `internal/Providers.java:49`; database tests gated by `SDM_POSTGRES_TEST_DSN` (`T312Shop.java:116-122`) |
| Databricks SQL JDBC | Same code path | Same | `internal/Providers.java:51`; provider/conversion tests exist, not proof of live task matrix |
| Databricks metric view | Source conversion code | No source-backed Tabular route | `DatabricksSource.semanticModel`, `internal/Providers.java:74`; conversion-note tests |
| Lakebase | Can supply PostgreSQL JDBC connection; no dedicated authentication helper | Application Arrow input only | `PostgresSource` accepts caller connection; no Lakebase-specific Java API or test claim |
| Databricks Spark | No Spark-session source | Application Arrow input only | Sealed `Source` permits four JDBC source types only |
| Snowflake / semantic view | Absent, accepted follow-up | Application Arrow input only | No Java source type; do not call this an accidental scope omission |
| S3/files | No Java reader/source API | Application Arrow input only | Python connector reader is a separate capability, not warehouse push-down |

## Prediction support and actual evidence

| Task | In-memory Java Relational | JDBC Java Relational | Java Tabular |
|---|---|---|---|
| Node regression | Code + local HTTP test | DuckDB/SQLite test; PostgreSQL gated test | Direct/PQL regression tested |
| Binary classification | Code; equality-query session test | Binary temporal predicate tested | Core task input exists; no positive classification journey found in `TabularTest` |
| Multiclass classification | Shared core path; not a dedicated in-memory test found | Multiclass tested through `T312PredictTest` | Core task input exists; no positive classification journey found in `TabularTest` |
| Temporal aggregate regression/classification | Materialize/predict tested | DuckDB/SQLite tests; PostgreSQL gated | Not a general Tabular task |
| Temporal link | In-memory materialize/predict tested | New Rust admission reachable, but no Java JDBC link test found | Not a Tabular task |
| Ranking | In-memory tested with explicit candidates | Gap: candidates not forwarded and parts refused | Not a Tabular task |
| Multi-step forecast | In-memory tested | Gap: parts refused | Not a Tabular task |
| Static link | Requires separate Rust task-admission audit; no Java-specific test found | Same | Not a Tabular task |
| Fitted sessions | Scalar/holdout/retry/partial-answer tests | 10 DuckDB/SQLite scalar/temporal lifecycle tests | Direct/PQL regression session tests |

In-memory rank/forecast fit is explicitly refused, as a session pins one question (`RelationalSessions.java:22`; `RelationalTest.java:201-202`).
Databricks Model Serving sessions are explicitly unsupported (`SdmClient.java:251,258`), independent of which database supplied the data.
The live validation bridge used by prior Python validation does not change that public transport limit.

## Public API parity

| Capability | Python | Java | Verdict |
|---|---|---|---|
| Named schema inputs | DataFrame mapping; source tables/options | Arrow Table mapping; JDBC source/options | Equivalent concept, differing supported input types/connectors |
| Schema edits | Mutable table/column properties | Immutable `with*`/`without*` schema methods | Intentional language/ownership difference |
| Default model selection | Multi-table/relationship selects Relational | In-memory defaults Tabular; explicit Relational | Behavioral difference |
| Query parsing | Query bound to schema; materialize model/task args | Parse separately, pass schema/options | Similar workflow; task override gap |
| Direct inputs | DataFrame; schema/inference/task options | Arrow Table; target/task only | Smaller Java surface |
| Source read helper | `sdm_connectors.read_table` | No public equivalent | Convenience/input gap |
| Predict and requested outputs | Supported | Supported | Shared Rust interpretation |
| Multi-batch prediction | Supported | Supported ordinary requests | Warehouse decomposition gap distinct from batching |
| Fit/predict/close | Supported; Python context manager | Supported; AutoCloseable | Equivalent core policy; Java synchronized owner |
| Retained graph | Python schema/materializer ownership | Explicit `RelationalMaterializer` | Explicit Java lifetime needed for repeated RNG/graph reuse |
| Result rows | Pandas + items/iteration | ArrowReader | Intentional language presentation |
| Result extras | explanation, absent, unscored_candidates, merge | explanation, absent; no unscored-candidates or merge method | Java convenience/metadata access gaps |
| Partial failure | Exception retains answered result | `StructuredDataException.answered(allocator)` | Both retain completed rows; different container |
| Health/model listing | `models()`, `health_ready()` | `capabilities(model)` only | Java API gap |
| Serialization | Python schema/request pickle behavior | No public schema/request serialization API | Gap if persistence parity required |
| Async APIs | None found | None found | No asymmetry established |
| HTTP configuration | API key, timeout, retries | Same basic options | Similar; built-in Databricks serving differs from generic HTTP |
| Warnings | Python warnings | SLF4J WARN | Host presentation, not legacy execution |

## Inspected test evidence

Existing reports in `sdmc-p4-java-relational/java/target/surefire-reports` show:

| Report | Tests | Failures/errors | Skips |
|---|---:|---:|---:|
| RelationalTest | 13 | 0 | 0 |
| TabularTest | 13 | 0 | 1 |
| WarehouseSessionTest | 10 | 0 | 0 |
| T312PredictTest | 15 | 0 | 1 |

The reports are existing evidence, not newly executed audit coverage.
`RelationalTest.java:178` explicitly exercises temporal regression, temporal links, ranking and forecasts on Arrow in-memory data.
`T312PredictTest.java:68-99` exercises regression/binary/multiclass on DuckDB/SQLite/PostgreSQL with a local HTTP endpoint; PostgreSQL may skip.
`WarehouseSessionTest.java:64-170` exercises only DuckDB and SQLite sessions.
`TabularTest.java:286` has an environment-gated live test; the recorded skip must not count as live success.

## Documentation drift

`PredictiveQuery.java:102` still describes ranking/forecast as unsupported broadly; that now applies to its warehouse branch, not in-memory Relational.
The end of `java/README.md` says `Schema.fromData` accepts one table, while its earlier section and implementation accept multi-table Relational inputs.
Clarify that one-table constraint is Tabular-specific.

## Suggested next verification

Reproduce warehouse ranking/forecast through public Java API with a local DuckDB fixture and record exact refusal paths.
Add positive Tabular binary/multiclass cases, in-memory Relational multiclass, and Java JDBC temporal-link prediction/session cases.
Audit whether defaults and configuration gaps are accepted API differences or promised parity gaps before changing public signatures.
Do not mark remote Databricks/PostgreSQL/Lakebase cells tested solely from JDBC conversion tests or shared Rust tests.


## New focused runtime audit

`Phase4CoverageAuditTest` now passes 9 cases with the rebuilt current JNI artifact: 5 positive functionality cases and 4 explicitly asserted current gaps.
This is not nine supported feature combinations.

| Cases | Observed result |
|---|---|
| JDBC ranking with explicit candidates, DuckDB/SQLite | Refused `INVALID_REQUEST`, `rule=candidates_missing`, `field=candidates`; confirms candidate forwarding gap |
| JDBC forecast, DuckDB/SQLite | Refused `UNSUPPORTED_REQUEST`, `rule=in_parts`; confirms Java host decomposition gap |
| JDBC temporal links, DuckDB/SQLite | Direct prediction and fit/repeated prediction/close succeed after closing JDBC connection |
| Tabular classification with two and three classes | Positive HTTP prediction succeeds; task spelling is `classification` for both, unlike Relational binary/multiclass task names |
| In-memory Relational multiclass | Materialization creates multiclass task and HTTP prediction succeeds |

Evidence: `java-audit-result.json`, `java-audit-r4.log`, `TEST-com.nvidia.sdm.Phase4CoverageAuditTest.xml`, and `java-native-identity.json` in this directory.
The essential JNI rebuild used current Rust tree `35c508dc5c63948a6dc82264eef74c17ddea7809`, warm r8 cache, jobs=2 and the heavy guard.
No production code changed.
Earlier failed audit attempts were fixture issues: wrong Tabular task spelling, attempting to mutate an immutable decoded JSON map, and using a Relational-specific scripted response for Tabular.
These are preserved in `java-audit-r1.log` through `java-audit-r3.log`, not counted as product failures.
