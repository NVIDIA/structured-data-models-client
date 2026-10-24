# Phase 3 tracker: T3.13

The design branch owns `tracker.md`; this file holds T3.13's section until it folds in.

## Test author: the S2 case input files (before H0)

Written by T3.13's test author from the reference at 43e55f4f, on `rust-core-phase-3` at 3ae4d1af, so that C-1 records every case Java's Databricks leg needs.
The files have no `core` section yet; it follows in a later `T3.13 tests:` commit, once T3.2's loaders and T3.4's v28 exist.
Until T3.2 lands, the case loaders refuse the files' `warehouses` field, the one field they do not know yet.

Java runs the same case files as Python (design choice 10, "Tests"), so these two files hold only what the Databricks cases of T3.4 and T3.5 do not.

| Case | Kind | What it shows | Exit |
|---|---|---|---|
| `java.databricks-sql.multiclass` | `java-databricks-sql` (dryrun-plan.yaml) | A multiclass question over a Databricks SQL warehouse, the source opened explicitly and described as the J2 questions describe theirs: tables named, keys, a time column and a relationship given | 1, and E-7 |
| `java.databricks-sql.types` | The brief's input axes (JDBC types) | Every scalar Databricks type of C-1's recorded tables as a feature, each with a null, a NaN, a `BIGINT` past 2**53 and a `DECIMAL(38,6)` past a double's precision, a zoned `TIMESTAMP` beside a `TIMESTAMP_NTZ`, and a `BINARY` and an `ARRAY<INT>` column the old client leaves out | 1, and E-7 (B7) |

Both were run on the reference at 43e55f4f on DuckDB with the same inputs (the S2 loader does not exist yet): both answer through predict, and every feature column but the two left out reaches the payload with its nulls.
Their Databricks reference lines come from C-1.

## Java's 18 S2 paths

| Path | Claimed by |
|---|---|
| `source.open/explicit/databricks-sql` | Both cases above; T3.4's cases |
| `schema.from-source/tables/databricks-sql` | Both cases above; T3.4's AdventureWorks cases |
| `schema.from-source/catalog-keys/databricks-sql` | T3.4's cases (its brief claims every Databricks mode) |
| `schema.from-source/all-tables/databricks-sql` | T3.4's leg cases |
| `schema.from-source/semantic-model/databricks-metric-view` | T3.5's metric-view cases |
| `query.materialize/indices-binary/databricks-sql`, `client.predict/binary/databricks-sql` | `java.databricks-sql.types`; T3.4's `inactivity_90d` case |
| `query.materialize/indices-multiclass/databricks-sql`, `client.predict/multiclass/databricks-sql` | `java.databricks-sql.multiclass`; T3.4's `product_color` case |
| `query.materialize/indices-regression/databricks-sql`, `client.predict/regression/databricks-sql` | T3.4's `demand_30d` and `demand_30d_historical` cases |
| `query.materialize/indices-{binary,multiclass,regression}/databricks-metric-view`, `client.predict/{binary,multiclass,regression}/databricks-metric-view` | T3.5's metric-view cases, which name `databricks-sql` (its brief claims all five kinds) |
| `client.connect/databricks-serving/any` | Not a case: the serving transport's named Java test against a scripted endpoint, in this package's later test round |

A metric-view case runs on Java's `databricks-sql` leg through the same connection (the brief's open question 3).

## Phase 0's J2 questions

Each is a case on AdventureWorks naming `databricks-sql`, written by T3.4's or T3.5's test author; matched by question text (the brief's open question 2), so none is written twice.
The case IDs are filled in when their files land.

| J2 question | Question (the deleted notebook's) | Case |
|---|---|---|
| `demand_30d` | `PREDICT SUM(sales_order_details.OrderQty, 0, 30, days) FOR EACH products.ProductID`, the top product | T3.4's |
| `demand_30d_historical` | The same, with `anchor_time` 180 days before the last order | T3.4's |
| `inactivity_90d` | `PREDICT COUNT(sales_order_headers.*, 0, 90, days)=0 FOR EACH customers.CustomerID`, the 25 most frequent customers | T3.4's |
| `product_color` | `PREDICT products.Color FOR EACH products.ProductID`, the top product | T3.4's |
| `view_demand_30d` | `demand_30d` through the owner's AdventureWorks metric view | T3.5's |
| `view_inactivity_90d` | `inactivity_90d` through the same view | T3.5's |

## Current implementation and verification boundary

The Java source, JDBC conversion table and serving transport are implemented.
Metric views use `DatabricksSource.semanticModel(name, trustExpressions)` and the shared core's `view` choice.
`Schema.conversionNotes()` renders the core's typed notes after caller overrides; grouped loss warnings remain run warnings.
No view parsing, SQL planning or diagnostic policy lives in Java.

The four base J2 shapes have equivalent shop-data tests through Java's core calls and both direct and serving transports: regression, the same historical regression 180 days earlier, 90-day inactivity and multiclass classification.
These tests use the existing DuckDB fixture to exercise the shared semantics; they do not claim live Databricks or AdventureWorks evidence.
JDBC conversion tests separately express the required non-UTC JVM and spring DST parity.
The fake now follows released JDBC 3.4.2's unsupported typed getter and initial JVM-zone timestamp conversion, rather than inventing `LocalDateTime` support.
A narrow probe with that released driver's actual result-set getters confirms ordinary NTZ reads are repaired, but DST-gap wall fields still normalize before reaching the host.
The gap regression retains the desired original wall fields and needs a further fix; it is not passing acceptance evidence.
A live non-UTC warehouse check and the actual E-7 comparison remain phase-end work.

The two known nightly Java probe defects were reproduced with the package artifact: the relative classpath failed after changing to scratch, and the load source failed to compile against the current API.
The corrected classpath loaded the packaged native, and the corrected source compiled on JDK 17.
This targeted diagnosis ran no package suite or full CI.
Canonical binding vectors were regenerated with `contract/tools/t311_java_record.py binding` using lane a's existing test-ops kit; its output exactly reproduced the preserved vector changes.

Package tests and lint wait for two approvals on the final candidate.
The final metric-view integration waits for T3.5's actual phase landing.
The real E-7 JDBC/live runner and model comparison remain T3.10's phase-end evidence.
P3S-2 has the independent broad reviewer's explicit approval for the better malformed-success error.
P3S-1 and P3S-3 through P3S-6 remain proposed; no approval is inferred from this implementation.
P3S-8 records the focused reviewer's nonblocking large-diagnostic control-limit finding for follow-up.
The owner subsequently requires the official Databricks Java SDK and all authentication methods it supports.
The serving client now accepts `WorkspaceClient` or default SDK authentication, forwards refreshed headers while preserving the core's bytes, and includes SDK credential refresh in the serving deadline.
P3S-7 is retired rather than accepted; requests identify the client without mutating the SDK's global product setting.
JV-4's startup-session timezone difference remains proposed and is handed to T3.9 for policy resolution; the existing same-zone offset helper does not fix it.

## Round 1 follow-up

The serving transport now shares the existing HTTP transport's cancellable whole-body exchange helper.
Each serving attempt receives only the remaining overall call budget, including retries and response-body reads.
New tests were committed before the fix and hold response bodies open after headers, both on an initial attempt and after a retry.
A bounded local-server probe returned success after 1,477 ms with the original 150 ms timeout, then raised the intended endpoint failure after 167 ms with the fix.
That probe compiled only the two affected Java sources in private scratch; it was not a package build or suite.
The NTZ fix uses the driver's supported calendar getter instead of the unsupported typed getter and preserves nulls, microseconds and pre-epoch values.
The remaining DST-gap issue requires a lossless transport decision before this package's final checks.
