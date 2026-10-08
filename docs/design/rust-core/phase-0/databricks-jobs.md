# Databricks job matrix to public paths

Task T0.5 of [Phase 0](design.md), part 3.
This maps each cell of the first attempt's 20-cell Databricks job manifest, through today's notebooks in `examples/databricks/`, to the public paths in [scope-python.yaml](scope-python.yaml) it exercises.
The cell list is the historical one from the first attempt's `investigations/P95-ACCEPTANCE-PREPLAN.md` (section 4), which its decision DV55 adopted as the minimum; the old `matrix*.sh` scripts point at notebooks that no longer exist and are not used.
The notebooks are `adventureworks_relational.ipynb` (Unity Catalog, over a SQL warehouse or over the notebook's Spark) and `lakebase_relational.ipynb` (Lakebase through `PostgresSource`).

## The 20 cells

Python versions are as observed in the first attempt's job outputs (`~/.local/share/sdm-dbx/result-review-fix.json` and `result-typed.json`); see [support-matrix.md](support-matrix.md) for the rest of each environment.

| Cell | Route | Runtime | Architecture | Python | Notebook | Data backend |
|---|---|---|---|---|---|---|
| J01 | Unity Catalog via SQL warehouse | DBR 14.3 LTS | x86_64 | 3.10.12 | adventureworks_relational, `warehouse_id` set | `databricks-sql` |
| J02 | Unity Catalog via SQL warehouse | DBR 14.3 LTS | aarch64 | 3.10.12 | same | `databricks-sql` |
| J03 | Unity Catalog via SQL warehouse | DBR 15.4 LTS | x86_64 | 3.11.11 | same | `databricks-sql` |
| J04 | Unity Catalog via SQL warehouse | DBR 15.4 LTS | aarch64 | 3.11.11 | same | `databricks-sql` |
| J05 | Unity Catalog via SQL warehouse | DBR 16.4 LTS | x86_64 | 3.12.3 | same | `databricks-sql` |
| J06 | Unity Catalog via SQL warehouse | DBR 16.4 LTS | aarch64 | 3.12.3 | same | `databricks-sql` |
| J07 | Unity Catalog via SQL warehouse | Serverless env 1 | not proven (x86_64 observed) | 3.10.12 | same | `databricks-sql` |
| J08 | Unity Catalog via SQL warehouse | Serverless env 2 | not proven | 3.11.10 | same | `databricks-sql` |
| J09 | Unity Catalog via SQL warehouse | Serverless env 3 | not proven | 3.12.3 | same | `databricks-sql` |
| J10 | Unity Catalog via Spark | DBR 17.3 LTS | x86_64 | 3.12.3 | adventureworks_relational, `warehouse_id` empty | `databricks-spark` |
| J11 | Unity Catalog via Spark | DBR 17.3 LTS | aarch64 | 3.12.3 | same | `databricks-spark` |
| J12 | Unity Catalog via Spark | Serverless env 4 | not proven | 3.12.3 | same | `databricks-spark` |
| J13 | Unity Catalog via Spark | Serverless env 5 | not proven | 3.12.3 | same | `databricks-spark` |
| J14 | Unity Catalog via Spark | Serverless env 6 | not proven | 3.12.3 | same | `databricks-spark` |
| J15 | Lakebase | Serverless env 1 | not proven | 3.10.12 | lakebase_relational | `lakebase` |
| J16 | Lakebase | Serverless env 2 | not proven | 3.11.10 | same | `lakebase` |
| J17 | Lakebase | Serverless env 3 | not proven | 3.12.3 | same | `lakebase` |
| J18 | Lakebase | Serverless env 4 | not proven | 3.12.3 | same | `lakebase` |
| J19 | Lakebase | Serverless env 5 | not proven | 3.12.3 | same | `lakebase` |
| J20 | Lakebase | Serverless env 6 | not proven | 3.12.3 | same | `lakebase` |

Serverless architecture is "not proven" because a serverless job does not choose it; the preplan requires each cell to record the observed architecture.

## Paths every cell exercises

`B` is the cell's data backend from the table above.

| Notebook step | Path | Slice |
|---|---|---|
| `%pip install --only-binary=:all:` with the runtime's NumPy pinned | Packaging, not a path: a wheel must exist for the cell's Python and architecture (L8, [support-matrix.md](support-matrix.md)) | Not a slice |
| `warehouse(warehouse_id, workspace=...)` (J01 to J09) | `source.connect-databricks/warehouse/databricks-sql` | S2 |
| `lakebase(endpoint, database=..., workspace=...)` (J15 to J20) | `source.connect-databricks/lakebase/lakebase` | S2 |
| `DatabricksSource(connection or spark, catalog=, schema=)`, or `PostgresSource(connection, schema=)` | `source.open/explicit/B` | S2 |
| Unity Catalog: `Schema.from_source` with `primary_keys`, `time_columns` and `relationships` declared | `schema.from-source/tables/databricks-sql` or `.../databricks-spark` | S2 |
| Lakebase: `Schema.from_source` with keys and links read from the catalog | `schema.from-source/catalog-keys/lakebase` | S2 |
| `schema['products'].time_column = None` | `schema.edit/set-time-column/any` | S1 |
| `del table['rowguid']` on each `SourceTable` | `schema.edit/drop-column/any` | S1 |
| `'rowguid' in table`, `print(table)`, `print(schema.relationships)` | `schema.inspect/attributes/any` | S1 |
| `SDMClient.for_databricks_serving(endpoint)` | `client.connect/databricks-serving/any` | S2 |
| `PredictiveQuery(text, schema)` for every question | `query.validate/accept/any` | S0 |
| `answer.to_pandas()`, then `display` and `to_dict('records')` | `result.read/<kind>/any` (per question below) | per question |

A refused question is caught as `StructuredDataError` and reported; that is a case outcome under the question's paths, not a separate path.

## Paths per question

Every Unity Catalog cell asks the six base questions and, when its `metric_view` widget names one, the two view questions; every Lakebase cell asks the six base questions.
That is 8 questions in each of 14 cells and 6 in each of 6, 148 question runs, as the preplan counts.

| Question | PQL kind and options | Paths (data backend `B`) | Slice |
|---|---|---|---|
| `demand_30d` | Regression; `indices`, `inference_config` | `query.materialize/indices-regression/B`, `client.predict/regression/B`, `result.read/regression/any` | S2 |
| `demand_30d_historical` | Regression; adds `anchor_time` | Same as `demand_30d` | S2 |
| `demand_3_timeframes` | `FORECAST 3 TIMEFRAMES`; `anchor_time`, and `inference_config` on Unity Catalog only | `query.materialize/indices-forecast/B`, `client.predict/forecast/B`, `result.read/forecast/any` | **S3** |
| `inactivity_90d` | Binary (`COUNT(...)=0`); 25 `indices` | `query.materialize/indices-binary/B`, `client.predict/binary/B`, `result.read/binary/any` | S2 |
| `recommend_top5` | `RANK TOP 5` over `LIST_DISTINCT`; `candidates`, `anchor_time`, `num_context_samples` | `query.materialize/indices-ranking/B`, `client.predict/ranking/B`, `result.read/ranking/any` | **S3** |
| `product_color` | Multiclass (a static column of the entity) | `query.materialize/indices-multiclass/B`, `client.predict/multiclass/B`, `result.read/multiclass/any` | S2 |
| `view_demand_30d` (J01 to J14) | Regression through a metric view | `schema.from-source/semantic-model/databricks-metric-view`, `query.materialize/indices-regression/databricks-metric-view`, `client.predict/regression/databricks-metric-view`, `result.read/regression/any` | S2 |
| `view_inactivity_90d` (J01 to J14) | Binary through a metric view | `schema.from-source/semantic-model/databricks-metric-view`, `query.materialize/indices-binary/databricks-metric-view`, `client.predict/binary/databricks-metric-view`, `result.read/binary/any` | S2 |

The metric view questions run over the warehouse connection in J01 to J09 and over Spark in J10 to J14; the manifest has one `databricks-metric-view` backend for both, so cases must cover both connections.

## S3 paths the S2 job matrix needs

Two of the six base questions fall in S3, on every one of the 20 cells:

- `demand_3_timeframes` needs `query.materialize/indices-forecast/B`, `client.predict/forecast/B` and `result.read/forecast/any`.
- `recommend_top5` needs `query.materialize/indices-ranking/B`, `client.predict/ranking/B` and `result.read/ranking/any`.

Here `B` is `databricks-sql`, `databricks-spark` and `lakebase`.
These paths carry a note in [scope-python.yaml](scope-python.yaml).
The plan requires the job matrix at the S2 gate and "the Databricks job matrix equal to the baseline" at the Python switch for S0, S1 and S2 paths, so the conflict has to be settled before Phase 3.

## Decisions for the owner

**J1. Where ranking and forecasting on the Databricks and Lakebase backends belong.**
Option A: move the `indices-ranking` and `indices-forecast` modes of `query.materialize`, `client.predict` and `result.read` on the SQL backends into S1 (offline, DuckDB, SQLite, Postgres) and S2 (live), which pulls the decomposition and result merging in `sdm-request` forward into Phase 2 and Phase 3.
Option B: keep them in S3 and, at the S2 switch, run those two questions on the old path; a `Schema` built by the switched `from_source` then feeds an unswitched `materialize` in the same session, which is the cross-call mix the plan's switch-together rule avoids, and it needs its own compatibility check.
Option C: drop the two questions from the S2 job matrix and run them in an S3 job matrix, which narrows the S2 gate to four of six base questions.
Recommendation: Option A, since the job matrix is the main live proof and both questions are what Databricks users run; frames, sessions, link prediction and tabular requests stay in S3.

**J2. What "Python and Java agree on the same Databricks data" covers in S2.**
Java's manifest has only the warehouse route (`databricks-sql`) and no ranking, forecasting, Spark or Lakebase paths, so Java can match Python on `demand_30d`, `demand_30d_historical`, `inactivity_90d`, `product_color` and, if confirmed, the two metric view questions.
Recommendation: state that set as Java's S2 agreement, and keep ranking and forecasting out of Java until the owner adds scope.

**J3. Routes the historical cells never cross.**
The warehouse route runs only on DBR 14.3 to 16.4 and serverless env 1 to 3, and the Spark route only on DBR 17.3 and serverless env 4 to 6, although the warehouse route also works on the newer runtimes.
Recommendation: keep the 20 cells as the minimum (DV55), and have the acceptance author add one warehouse-route cell on DBR 17.3 or a newer serverless environment, since that is a supported user route with no cell.

## Notes for the acceptance author

- The Unity Catalog notebook asks the view questions only when `metric_view` is set; the job runner sets it on every Unity Catalog cell, so a required view question is never a silent skip.
- The two notebooks differ in one option: `demand_3_timeframes` passes `inference_config` on Unity Catalog and not on Lakebase; a runner that reproduces the notebooks keeps that difference or records the change.
- The Unity Catalog notebook leaves `sales_order_headers` without a primary key on purpose; the cases under `schema.from-source/tables/...` should include a table without one.
- J01, J02, J07 and J15 run Python 3.10 and start with pyarrow 8 before the install upgrades it, which is where the pyarrow floor decision in [support-matrix.md](support-matrix.md) bites first.
