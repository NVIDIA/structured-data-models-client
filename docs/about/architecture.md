# structured-data-models-client architecture plan

**What it is:** one Python client for NVIDIA's structured-data foundation
model NIMs.
**Status:** built. All seven phases are implemented and tested end to end
against live NIMs.

> This plan records the design rationale and later corrections. For interfaces,
> the shipped implementation is authoritative. See the
> [quickstart](../get-started/quickstart.md) and
> [prediction output](../reference/prediction-output.md).

---

## 0. One inconsistency in the contract, which nothing shipping depends on

**The contract will not let a new model exist.**

`PredictionRequest.model` and `SessionCreateRequest.model` are a **closed
enum** containing only `kumo-tabular` and `kumo-relational`. The source
document's `model="some-future-model"` escape hatch therefore cannot be sent.
The response's `model` field *is* an open string (`maxLength: 1024`), so the
contract contradicts itself.

Verified on `structured-data-api@main` as of 2026-09-30.

**This does not affect anything shipping.** `kumo-tabular` and
`kumo-relational` are both in the enum, so both work today and neither needs a
contract change. What the enum forecloses is the source doc's
`model="some-future-model"` escape hatch, which is deferred along with the
models it would have served (§11). An unrecognised id is refused client-side
with a message naming the contract change it would need, rather than being sent
and rejected.

If a third model is ever wanted, a small MR on `structured-data-api` opens the
field. Until then this is a note, not a task.

> **Model ids are fine.** An earlier draft claimed the NIMs still served
> `nemotron-*`. That was a stale local checkout. The serving side declares
> `ModelId = Literal["kumo-tabular", "kumo-relational"]`, and the tabular NIM's
> adapter registration, `/v1/models` entry and `model_manifest.yaml` all say
> `kumo-tabular`. All three places agree. **Re-clone before copying anything
> (§8).**

---

## 1. The idea in one page

Today you pick a model by calling a different method. Tomorrow the model is an
argument.

```python
from sdm_client import (
    SDMClient,
    Schema,
    PredictiveQuery,
    KumoRelational,
    KumoTabular,
)

schema = Schema.from_data({'users': users_df, 'orders': orders_df})
query = PredictiveQuery(
    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
)
request = query.materialize(indices=[1, 2, 3])

with SDMClient(url='http://localhost:8000') as client:
    result = client.predict(request, model=KumoRelational())
```

A one-table question can use either model. The models read requests
differently, so each request is **gathered** for one of them. Specify the model
when it is not the default:

```python
schema = Schema.from_data({'users': users_df})
question = PredictiveQuery('PREDICT users.churn FOR EACH users.user_id', schema)

client.predict(question.materialize(indices=[1, 2]), model=KumoTabular())  # ✅ the default
client.predict(
    question.materialize(indices=[1, 2], model=KumoRelational()),  # ✅ gathered as a graph
    model=KumoRelational(),
)
```

Gather and send for the same model. A request built for one is refused by the
other rather than reshaped, because the relational gather samples a
neighbourhood the tabular envelope has nowhere to put.

A relational aggregation is relational-only. The `TABULAR` profile rejects
related tables before anything is sent. `TABULAR` also refuses a target with a
time window, a `WHERE` filter on the entity, an `ASSUMING` clause, and a target
that is not a column of the table.

Two ways to build the one `Request`:

```
PredictiveQuery(text, schema).materialize(...)  ─┐
                                                 ├─►  Request  ─►  client.predict(request, model=…)
Request(context=…, predict=…, target=…)         ─┘
```

Four rules the whole design follows:

| Rule | Consequence |
|---|---|
| The model is an argument, never a method | `client.predict(req, model=…)`. No `client.relational()`. |
| One `Request` for every model | Tabular is the one-table case, not a second system. |
| One return type | `predict()` always returns a `PredictionResult`. Never a DataFrame-or-Explanation union. |
| A model declares itself as data | Adding a model is a declaration plus a small adapter. |

---

## 2. Packages

Five distributions in one repo, released together on one version.

| Distribution | Import | Kind | Holds |
|---|---|---|---|
| `structured-data-models-client` | `sdm_client` | pure python | The public API: client, `Schema`, `Request`, `PredictiveQuery`, model handles, sessions and results. Adapters and transport live here too, as implementation details rather than public API. |
| `structured-data-models-api` | `sdm_api` | pure python | The wire only: OpenAPI-generated bindings and the types the NIM speaks. |
| `structured-data-models-pql` | `predictive_query` | pure python | The language: grammar, parser, validator, AST, model profiles, and the schema types PQL speaks. |
| `structured-data-models-connectors` | `sdm_connectors` | pure python | Reading source tables: SQLite, DuckDB, PostgreSQL, Snowflake, Databricks, S3. |
| `structured-data-models-relational-engine` | `sdm_relational_engine` | **native** | The relational model's heavy client-side work: graph building, samplers, the C++ neighbor sampler. |

```
structured-data-models-pql  ◄── structured-data-models-api  ◄── structured-data-models-client ──► structured-data-models-connectors
   ▲                          │
   └────── structured-data-models-relational-engine ◄┘   (only via the [relational] extra)
```

**Only `structured-data-models-relational-engine` is native.** Everything else installs anywhere,
so `kumo-tabular` works on a Mac; `kumo-relational` needs Linux (x86-64 or
aarch64).

```bash
pip install structured-data-models-client                 # tabular, PQL, transport everywhere
pip install structured-data-models-client[relational]     # adds the native relational engine
pip install structured-data-models-client[snowflake]      # one backend (also duckdb/sqlite/postgres/databricks/s3)
pip install structured-data-models-client[all]            # relational + every backend
```

`[explain]` and `[relbench]` stay outside `[all]` and must be requested by
name. `[explain]` posts row data to a third-party LLM, so installing it is
deliberate.

### Why `structured-data-models-pql` is separate

The language is a distribution of its own because two packages need it and
neither needs the other. `structured-data-models-client` resolves a query against a schema;
`structured-data-models-relational-engine` reads the AST it produces. Putting the parser in either
one would make the other depend on a driver or a client it has no use for.

It carries the parser, the AST, the validators, **and** the schema types
(`Dtype`, `Stype`, `SchemaDefinition`, `TableDefinition`), so those types have
exactly one owner and nothing has to translate between two spellings of the
same enum.

The tree is first-party source of this repository, maintained here under
Apache-2.0. It was briefly carried as a vendored copy of a separate internal
repository; that repository is being retired, so there is no upstream to track,
no manifest to re-record, and no dependency swap pending. It imports as
`predictive_query`, which is a name inherited from that period rather than one
this layout would choose today.

None of it is public API. Users write PQL through `sdm_client.PredictiveQuery`,
and [the stability page](../reference/stability.md) lists `predictive_query`
with the internals, so any release may change it. The same holds for
`ModelSpec`, the `RuntimeContract` it carries, and `ModelAdapter` below: they
are how the client is built, not names it exports.

### Who owns which types

| Type | Owner | Why |
|---|---|---|
| `Dtype`, `Stype`, `TaskType`, `ProblemType` | `structured-data-models-pql` | PQL speaks them and will be an external dependency. One owner, no translation layer. |
| `SchemaDefinition`, `TableDefinition`, `Column`, AST | `structured-data-models-pql` | Same. The relational samplers *read* an AST, they never parse one. |
| `Profile`, `Capabilities` | `structured-data-models-pql` | What a model may be asked, as data. §6. |
| The time column rule (`predictive_query.time_columns`) | `structured-data-models-pql` | Which timestamp inference takes as a table's time column. `Schema.from_data` and the engine behind `Schema.from_source` both call its `choose_time_column`, and neither imports the other. |
| Wire request/response, `InferenceSpec`, `OutputSpec` | `structured-data-models-api` | The Universal TFM envelope. PQL knows nothing about HTTP. |
| The DataFrame→wire encoder (`wire/`) | `structured-data-models-client` | **Corrected during phase 2.** §8 first put this in `structured-data-models-api`, but `wire/dtypes.py` imports numpy and pandas. That would cost `structured-data-models-api` its zero-dependency property, which is why the split exists. `structured-data-models-api` keeps the vocabulary; encoding a frame is client-side work. |
| `Schema`, `Request`, `PredictionResult` | `structured-data-models-client` | The public surface. |

---

## 3. `Schema`: the rename

`Graph` becomes **`Schema`**. Reviewers were right that calling a one-table
dataset a "graph" is confusing, and `Schema` is the word the wire contract
already uses.

```python
schema = Schema.from_data({'users': df})  # one table
schema = Schema.from_data({'users': u, 'orders': o})  # many

schema['users'].primary_key = 'user_id'
schema['users']['churn'].stype = Stype.categorical
```

Everything is inferred: dtypes, semantic types, primary keys, time columns and
relationships. Override only incorrect inferences. **You never write a schema
by hand.**

`PredictiveQuery` also accepts raw tables, so a one-table user need not meet
`Schema` at all:

```python
PredictiveQuery('PREDICT users.churn FOR EACH users.user_id', {'users': df})
```

**Composite keys.** A warehouse row identified by `(customer, region)` is
folded into one derived column, because the wire carries a single `instance_id`
per instance. `Schema` does the fold. Float columns are refused, because a join
must agree exactly on both sides. Precisely: *entity identity* becomes scalar
after folding. The wire's `TableSchema.primary_key` remains an array where a
related table's occurrence identity needs one, for example
`[instance_id, order_id]`.

### Schemas over tables that stay where they are

A schema can describe tables that stay in the warehouse.

`Schema.from_data` needs every table in memory, which caps the client at what one process can hold.
The engine has always sampled Databricks, Snowflake, DuckDB and SQLite tables with SQL, fetching only the rows a prediction needs, and the predecessor client exposed that through `Graph.from_databricks` and its siblings.
Nothing on `Schema` reached it, so every warehouse table had to be read whole into pandas first.
`Schema.from_source` restores it.

```python
from sdm_client import Schema
from sdm_connectors import DatabricksSource

adventureworks = DatabricksSource(spark, catalog='main', schema='sales')  # in a Databricks notebook
schema = Schema.from_source(
    adventureworks,
    tables={'customers': 'adventureworks_customers', 'orders': 'adventureworks_orders'},
    time_columns={'orders': 'order_date'},
)
request = PredictiveQuery(query, schema).materialize(indices=[...])
```

`tables` maps the names a query uses to the names in the source.
Left out, it is every table the source describes, each under its own name.
`primary_keys`, `time_columns` and `relationships` mean what they mean on `from_data`.
`time_columns={'orders': None}` says a table has no time column, on either.

A source is where the tables are and how to reach them, and it is the only thing that differs between backends.
Each warehouse has a class of its own, named for it, so what only one warehouse has is a method only its class offers:

| Source | Built from | Names | Notes |
|---|---|---|---|
| `DatabricksSource` | a `databricks.sql` connection, `sdm_connectors.databricks.warehouse(warehouse_id, workspace=w)` for one opened as your `WorkspaceClient`, or the notebook's `spark` | `catalog`, `schema` | A SQL warehouse on any runtime; `spark` needs no warehouse and uses the notebook's own compute and identity, on DBR 17.3+ or serverless environment 4+, which have `DataFrame.toArrow()`. Offers `semantic_model()` |
| `SnowflakeSource` | a `snowflake.connector` connection, or a Snowpark `session` | `database`, `schema` | A Snowpark session is reached through its own connector connection. Offers `semantic_model()` |
| `PostgresSource` | a `psycopg` connection, or `sdm_connectors.databricks.lakebase(endpoint, database=..., workspace=w)` for Lakebase | `schema` | Lakebase authenticates with a short-lived OAuth token as the password, which `lakebase()` mints with your `WorkspaceClient`; the connection outlives the token once it is open |
| `DuckDBSource`, `SQLiteSource` | a `duckdb` or ADBC SQLite connection | none | What lets CI run the whole path, SQL sampling included, with no credentials |

`Source(connection, ...)` still works, and returns the class that fits the connection; the classes are the spelling to reach for, because they are what a type checker and an editor can see.

`from_data` does not change: a mapping of frames is still read into memory and sampled there.

| | `from_data` | `from_source` |
|---|---|---|
| Table type | `Table`, holding `data: pd.DataFrame` | `SourceTable`: a source name and column metadata, no rows |
| Keys and relationships | given, else inferred from values and names | given, else the catalog's constraints, else inferred from names |
| Keys checked against | every row, in memory | every row, counted in the source unless it enforces the key |
| Semantic types | inferred from the rows | inferred from a 1,000-row preview, the engine's rule |
| Sampling at `materialize()` | the engine's `LocalSampler`, in memory | the engine's SQL sampler for that backend |
| What leaves the source | every row | the sampled rows |
| Request on the wire | sampled rows, embedded | the same |

**One source per schema, by construction.**
Sampling is a conversation, not a query.
The SQL sampler fetches the entities, then each target's history inside its window, then walks the graph a hop at a time: for every table reached, one query per relationship, each sending the ids it has so far and joining them to exactly one table.
The ids come back to the client between every query, so routing each table's queries to a different source is mechanically possible, and deliberately not done.
Ids and timestamps would have to compare identically across engines, time bounds would have to be merged across them, seeded sampling would stop being reproducible, and tables read from sources at different moments can leak or drop labels while the result looks entirely ordinary.
The in-memory sampler is a different algorithm again, a C++ neighbour sampler over the whole graph at once, so frames and warehouse tables cannot share one either.
Where tables really do live in two systems, the platforms already join them: Unity Catalog registers a Lakebase database as a catalog and federates Snowflake and PostgreSQL, so one Databricks source sees all of them.
Taking the source once, as an argument, makes a mixed schema impossible to write rather than an error to explain.
The small in-memory data that does belong beside a warehouse, which entities to score and which candidates to rank, is already an argument to `materialize()`, not a table.

**A second table type, not an optional `data`.**
`Table.data` stays a DataFrame, so no existing caller meets a `None`.
Code that needs rows takes a `Table`, and a `SourceTable` reaching it is refused with a message naming the table, rather than failing on an attribute.

**The caller owns the connection.**
The source holds it and never closes it.
A warehouse connection is the driver's own object, opened with the driver's own `connect`.
It is the type the engine already accepts, so there is no second connection type to convert between.
On Databricks, `sdm_connectors.databricks.warehouse()` and `lakebase()` open one as the caller's `WorkspaceClient` and hand back that same native connection, for the caller to close.
Every connection and API request structured-data-models-client opens there names it as `nvidia_structured-data-models-client/<version>`, as Databricks requires of a partner product, and a warehouse session it opens tags its statements `structured_data_models_client_purpose: sampling`.
Each package keeps a private copy of the identifier, so the connectors never import the client and the serving extra brings no connectors; a test holds the two copies to the same value.
A connection the caller opens and passes in is used as it is.
A notebook's `spark` is wrapped in the few calls the samplers make: a cursor that runs `spark.sql` with positional parameters and returns Arrow.
A schema over a source does not pickle.

**Keys are taken as stated.**
Unity Catalog and Snowflake constraints are informational: nothing stops a declared key from repeating.
`from_data` checks a key for nulls and duplicates over the rows it holds; against a source that is a whole-table aggregate per key, and it is not run yet.
Where it matters, check the key in the source before relying on it.

**Whole-table checks move.**
Each check `from_data` runs over every row either reads metadata for a source table (dtypes, constraints, row counts) or runs on the sampled rows (value casts, and whether scored rows mean what the schema says).
The two that would need a whole-table aggregate, key quality and whether a relationship joins anything, are skipped.
The heuristic that searches column combinations for a composite key is not run: against a warehouse it scans the table once per candidate column.

**Metadata is read once per schema, except the time range.**
Describing a table in place costs queries: its columns, its keys, a preview of up to 1,000 rows, and on Databricks a row count.
The schema keeps the engine's table objects from its first read, and each question is given fresh copies of them, so a question writes its keys and semantic types onto its own copy and never onto another's.
The minimum and maximum of each time column are still read for every question, because they decide the anchor a question defaults to, and a schema that outlives new rows has to see them.
On clustered or partitioned tables that aggregate is cheap; elsewhere it is the first cost a user notices.

**What it costs, measured.**
Against the four AdventureWorks tables in Unity Catalog (121,317, 31,465, 19,820 and 504 rows, 173,106 in all), through a 2X-Small SQL warehouse, one demand question:

| | Rows fetched | Schema | First question | Each later question | Peak memory |
|---|---|---|---|---|---|
| In place, metadata kept | 245,031 | 14.9 s | 10.6 s | 4.6 s | 254 MB |
| In place, metadata read per question | 252,181 | 13.4 s | 17.9 s | 14.7 s | 259 MB |
| Read into memory, `from_data` | 173,106 | 24.0 s | 0.4 s | 0.3 s | 328 MB |

At this size sampling in place is the slower way to ask.
The tables are smaller than the neighbourhoods a thousand context entities reach, so the sampler fetches more rows than the tables hold, and each hop is a round trip to the warehouse.
It wins where a table does not fit in memory, or where reading it whole takes longer than sampling it: the cost of reading into memory grows with the tables, and the cost of sampling with the question.

**Postgres needs the engine first.**
The engine was carried across before the predecessor gained a Postgres backend, so the Postgres source brings that backend over too: the table, the `JOIN LATERAL` sampler, and key discovery from `pg_catalog`.

**A semantic model is a source, not an argument to `Schema`.**
A Databricks metric view and a Snowflake semantic view each already say which tables a question is about and how they join, and the engine reads both into the same thing: tables, the links between them, and row-level columns, with their aggregations (measures, metrics) left out.
The industry's own name for that is a semantic model: the Open Semantic Interchange, which Databricks and Snowflake both back, and dbt use it.
So `source.semantic_model('sales_metrics')` is a source of its own, and `Schema.from_source` reads it like any other; `tables`, if given, selects and renames among the model's tables.
Only `DatabricksSource` and `SnowflakeSource` offer it, because only those warehouses keep one, and `Schema` names no warehouse at all.
An earlier draft took `metric_view=` on `from_source`, which put one vendor's word on the one method meant to be the same for every warehouse.
The engine turns the view into its graph: the source becomes a table named as it is in the catalog, each join a table named after the join, each dimension a column of the table it reads, and each join a relationship to the key it names.
An expression dimension stays an expression, computed in the warehouse when the table is sampled, so `materialize()` rebuilds the tables from the view and then keeps only what the schema kept.
Measures are aggregations rather than row values and are left out; a predictive query states the aggregation it asks about.
The columns a measure over one table aggregates are added to that table, and a conversion note names the aggregation matching each simple measure, so `revenue` can be asked as `SUM(order_lines.LineTotal, 0, 30, days)`.
A wildcard dimension such as `source.* EXCEPT (rowguid)` becomes the columns it stands for, as Databricks expands it.
A dimension truncating or casting a date stands for the column it truncates when the time column is chosen, so windows count days, not months.
A join whose table cannot be read, or whose key repeats, is left out with the tables and dimensions hanging off it; only a source table that cannot be read fails the conversion.
A model is written by someone else, and its expressions run under the caller's credentials each time a request is gathered, so by default they may call only pure, deterministic built-in functions, from an allowlist per dialect.
A column calling a user-defined function, `secret()`, `http_request()` or `SYSTEM$WAIT()` is left out with a message, and `trust_expressions=True` keeps it; sub-queries, comments and statement separators are refused either way.
Allowed is not the same as cheap or fixed.
Some allowed built-ins can build large values from small inputs and so cost the warehouse time and memory: `repeat`, `space`, `lpad`, `rpad`, `array_repeat` and `sequence` on Databricks, and `repeat`, `space` and `array_generate_range` on Snowflake.
Others read the session's time zone, so their results can differ between two sessions reading the same rows: `from_unixtime`, `date_format`, `to_timestamp`, `convert_timezone` with two arguments, and the `*_ltz` functions.
Both kinds stay allowed, since views use them for ordinary columns, so review a model's expressions for them when its author is not your own team.
A metric view's `filter` decides which rows it holds, so it is applied to the source table on every read rather than dropped.
A filter on a joined table cannot be, and the view is refused: create a regular view with that `WHERE` clause and point the metric view's source at it.
A parameter the filter or a dimension references is read as its default, as a query of the view without arguments reads it; a filter on one with no default refuses the view.
Besides the unit tests, metric views are converted live by an opt-in test that creates eight variants over AdventureWorks in the schema `SDM_MV_TEST_SCHEMA` names, and drops them again.
A Snowflake semantic view goes through the same path, read by the engine's own semantic-view support; it is tested against a fake connection, not yet against a Snowflake account.

---

## 4. One `Request` type

`Request` is the wire envelope minus the two things that are `predict()`
arguments. It is not a new abstraction; the Universal TFM contract already
unified tabular and relational.

| Field | Meaning |
|---|---|
| `task` | what is being asked: kind, target and entity |
| `schema` | the tables and how they relate |
| `context` | rows where the answer is known |
| `predict` | rows being asked about |
| `inference` | gathering knobs (§7) |

Tabular is the case where `schema` has one table and no relationships. That is
the whole difference.

`Request` is **singular**. The adapter splits a run that exceeds the server's
per-call cap into several round trips. Batching belongs to transport, not a
user-facing type.

**The direct builder is the single-table one.** A relational request is
gathered, not assembled from frames: the neighbourhood around each entity is
sampled and folded onto the instance table, which is what `materialize()` does
and what the relational model is sent. So `related_tables=` is refused there
rather than building a request no path can send.

```python
Request(context=ctx_df, predict=rows_df, target='churn', task='classification')

PredictiveQuery(
    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
).materialize(indices=[1, 2, 3], model=KumoRelational())
```

### The task kind is stated, never guessed

The source doc's last blocker: *"a 2 class target becomes multiclass, not
binary."* Inferring the kind from a column's cardinality is a guess, and it is
wrong exactly when a binary target looks categorical. What the two models
actually accept settles it, and they differ:

| | Task kinds the NIM accepts |
|---|---|
| `kumo-tabular` | `classification`, `regression` |
| `kumo-relational` | `classification`, `binary_classification`, `multiclass_classification`, `regression`, `forecasting`, `temporal_link_prediction` |

So there are two different answers, and neither is "guess from cardinality":

- **Tabular has nothing to decide *between* classifications.** The NIM takes
  only the coarse `classification`, and the model resolves binary vs multiclass
  itself. The client sends `classification` and never narrows it. The blocker
  does not exist on this path.
- **It does have to decide classification against regression, and it got that
  wrong.** A query names a target; it does not say what is being asked of it.
  The derivation read the target's semantic type, integers are numerical, and
  so `PREDICT users.churn` over a 0/1 column was gathered as a *regression* and
  answered with continuous values near 0.5 that look like probabilities but are
  not. Zero-and-one is the most common way to write a binary label, so that
  exact case now derives `classification` and says so. The values decide, not
  their storage. A 0/1 column with one missing value arrives as `float64` and
  is still a label. Only that case: three whole numbers may be a count, and a
  fractional pair is not a label, so both are left as regressions.
- **The caller has the last word where there is a decision to make.** Use
  `task=` on the direct builder or `materialize(task=…)` on the single-table
  PQL path,
  is authoritative and overrides everything above it. Stating it also silences
  the warning, which is the point of the warning. A relational query has no
  such decision. The engine resolves the query and states the kind in the
  payload, so passing `task=` with one is refused rather than quietly dropped.

PQL's own `TaskType` already has `BINARY_CLASSIFICATION` and
`MULTICLASS_CLASSIFICATION`, but a query resolves only to the coarse
`ProblemType` (`CLASSIFY`/`RANK`/`FORECAST`). The narrowing happens downstream
from the target's cardinality, which is exactly where the source doc's bug
lives. **The caller's explicit `task=` wins over that derivation, and when the
derivation is the only signal and the target has exactly two classes the client
warns**, naming both kinds rather than silently picking multiclass.

### Which rows get scored

A query says *what* to ask. It never says *which rows* to ask about. That is
always an explicit argument.

| Builder | Rows to score | Available |
|---|---|---|
| `Request(context=…, predict=…)` | `predict`: any rows, including rows the model has never seen | v1 |
| `query.materialize(indices=[…])` | existing rows, looked up by entity key | v1, tabular and relational |
| `query.materialize(predict=rows_df)` | supplied rows, which need not exist in the tables | phase 7, tabular |

`materialize()` with neither argument raises, naming both. **There is no
all-rows default** because it would silently score a whole table.

**The model a request is gathered for is an argument, not a consequence of the
schema's shape.** `materialize(model=…)` takes the same handle or id that
`predict()` does. Left unsaid, the shape decides, and the shape that counts is
the one `Request.is_relational` reports: a second table or a relationship sends
it relationally, so a lone table that points at itself is gathered as the graph
it is. Everything else is tabular, and `model='kumo-relational'` gathers it as
a graph of one node. Without that argument the two models could disagree with
no way to reconcile them: a one-table request was always built tabular, so the
relational model was refused it and nothing the caller passed could say
otherwise. Gathering and sending disagree loudly now: each adapter refuses a
request gathered for the other before it is sent.

**Tabular PQL has no key to name, and that is decided, not open.** `FOR EACH`
today must name a primary key, and a tabular frame has none. In phase 7
`Schema` synthesizes a request-local row id for a keyless table. This real,
positional column is never sent as data or exposed in the result, which carries
`row_index`. `FOR EACH users.<synthetic>` then validates unchanged, so the PQL
validator is **not** relaxed and no query that is illegal today becomes legal.
This is the concrete form of what review asked for here.

On the question review raised here: "unseen" means a row that is not in the
tables the schema was built from. `FOR EACH` enumerates rows that *are* there,
so PQL text alone can never produce them. The rows are an input, not a
derivation. This is why the direct builder supports tabular in v1.

---

## 5. `predict`, `fit`, and one result type

### fit is a session

The contract already has it: `POST /v1/sessions` pins context and runs no
inference; `POST /v1/sessions/{id}/predictions` scores against it. That is
`fit` and `predict`.

```python
fitted = client.fit(request, model=KumoRelational())  # POST /v1/sessions
fitted.predict(more_rows)  # reuses the pinned context
fitted.close()  # DELETE /v1/sessions/{id}
```

`client.predict()` works statelessly in one call and never opens a session.
A NIM that does not serve sessions is reported by `fit()`, with the code
`SESSIONS_UNSUPPORTED`.

### One return type

The old design returned a DataFrame, or an `Explanation` if you asked to
explain. That union was undocumented. Now:

```python
result = client.predict(
    request, model=KumoRelational(), outputs=['prediction', 'explanation']
)

result.to_pandas()  # always a DataFrame
result.explanation  # None unless you asked for it
```

Explanations are an **output field**, not a method. Ask for one with
`outputs=[..., "explanation"]`; there is no `explain=` argument. That is the
shape of the call — whether a deployment answers it is a separate question,
and the shipping relational build does not. See [What a deployment actually
serves](#what-a-deployment-actually-serves).

The DataFrame's shape depends on the **task**, not the model. Verified against
a live endpoint: classification and multiclass both give **one row per
entity**, with multiclass carrying a probability per class as columns rather
than a row per class. Ranking also gives one row per entity, holding its `k`
candidates as a list, because expanding them would leave `row_index` no longer
identifying a row; forecasting gives one row per entity per step. An earlier
draft said multiclass gave a row per class, and another said a ranking gave `k`
rows; neither does.

**Every row carries `row_index`. It also carries the entity key when the model
has one.** Tabular requests may have no key. Join on the key when present,
otherwise on `row_index`.
**Never join on row order.**

### What each model can produce

The contract defines seven output fields. **This table is what the client
declares** — what `capabilities()` reports and what it will put on the wire.
It is not what any one deployment answers; see below.

| Output | `kumo-relational` | `kumo-tabular` |
|---|---|---|
| `prediction` | ✅ | ✅ |
| `probabilities` | ✅ | ✅ |
| `quantiles` | ✅ | ✅ |
| `scores` | ✅ | ❌ |
| `rankings` | ✅ | ❌ |
| `embeddings` | ✅ | ❌ |
| `explanation` | ✅ | ❌ |

Asking a model for a field it does not declare fails client-side, naming what
it does declare.

### What a deployment actually serves

**The client declares the broader of the relational builds, and a deployment
may serve less.** The contract has no capabilities route — `/v1/models`
returns an id and nothing else — so the client cannot ask, and refusing
locally what one deployment can answer would be worse than letting another say
no. The narrowing is per *task*, and the endpoint says so with
`422 UNSUPPORTED_OUTPUT`.

Measured against the shipping relational deployment, which is the narrower
build:

| Task | Served |
|---|---|
| `regression` | `prediction`, `quantiles` |
| `binary_classification` | `prediction`, `probabilities` |
| `multiclass_classification` | `prediction`, `probabilities` |

So `scores`, `rankings`, `embeddings` and `explanation` are declared by the
client and refused by that deployment on every task. `rankings` is the one
that still works end to end: `RANK TOP k` is decomposed into a classification
per candidate, so the ranking is assembled here rather than asked for.

`forecasting` and `temporal_link_prediction` are declared as tasks and are not
served either — a forecast is sent as a regression per step, and a bare
`LIST_DISTINCT` is refused outright.

`skills/relational/SKILL.md` carries the same matrix for an agent, and
`tests/test_skills.py` asks the endpoint for every one of these on every task,
so a deployment that gains or loses an output fails the suite rather than
leaving the document wrong.

---

## 6. Adding a model

Three pieces. Two are cheap; one is a contract change we do not own.

**1. The contract must accommodate the model, which requires more than an id.**
`PredictionRequest.model` is a closed enum, so at minimum a new model needs an
MR on `structured-data-api` and regenerated `structured-data-models-api`. Until §0 is fixed,
an unrecognised id fails **client-side with a message naming that MR** rather
than being silently sent and rejected.

But a genuinely new *kind* of model needs more than a name. Anomaly detection
is the honest test:

| What it needs | Does the contract have it? |
|---|---|
| a task kind `anomaly_detection` | ❌ `TaskSpec.kind` has no such value |
| a request with **no label target** | ❌ `TaskSpec.target` is required |
| score direction, threshold, calibration | ❌ nothing standardises them |
| a place to put the score | ✅ `OutputSpec.fields` already has `scores` |

So **"add a model" is a contract change plus a profile plus an adapter, and for
a new task shape the contract change is the large part.** Anything else would
push the meaning into open `metadata` dicts, and a unified client whose
semantics live in untyped dicts is a facade over per-model behaviour, not a
common interface. Timeseries is cheaper because `forecasting` already exists as
a task kind. It is therefore the sensible third model, not anomaly.

This is the seam's real cost, stated up front so nobody discovers it at model
three.

**2. A profile** says what the model may be asked. `structured-data-models-pql` already ships
`RELATIONAL`, `TABULAR`, `TIMESERIES`, `ANOMALY`. A capability may only
*reject*: the language accepts a fixed set of queries and each model accepts a
subset, so a new model can never widen the language.

```python
TABULAR = Profile(
    mode=INFERENCE,
    capabilities=Capabilities(
        model='a tabular model',
        relationships=False,
        multiple_tables=False,
        temporal=False,
        entity_filters=False,
        counterfactuals=False,
        derived_targets=False,
        link_prediction=False,
        forecasting=False,
    ),
)
```

**3. An adapter** does what a profile cannot describe: shape the wire request,
normalize the response, choose a session strategy, release resources.

```python
class ModelAdapter(Protocol):
    spec: ModelSpec  # model id + profile + runtime facts

    def default_output(self, request) -> OutputSpec: ...
    def prediction_plan(
        self, request, output
    ) -> Iterable[PredictionRequest]: ...
    def session_plan(self, request) -> SessionCreateRequest: ...
    def normalize(self, response) -> PredictionResult: ...
    def close(self) -> None: ...
```

The profile, not the adapter, owns **query-shape** validation. Everything else
is validated by shared client code combining the profile with
`ModelSpec.runtime`: which output fields exist, whether sessions are served,
and the resource limits in §7. A profile cannot know whether *this deployment*
has a regressor checkpoint; that is a runtime fact and lives beside it, never
duplicated into it.

> **Cost, stated plainly.** The code this came from deleted its registry on
> purpose. Re-opening it buys back an adapter protocol to keep stable and
> conformance tests to maintain. That is the price of serving more
> than one model, and it is the price the doc asks for.

A model with no heavy client-side work (tabular) is an adapter alone. A model
with heavy work (relational) also gets its own package behind an extra, so base
installs stay light.

### What the endpoint says vs what the client believes

The old SDK reported its own registry, so it claimed a model the NIM did not
serve.

| Call | Answer |
|---|---|
| `client.models()` | Read from the endpoint's `/v1/models`. Endpoint truth. |
| `client.capabilities(model)` | Local: what this client can build and send. Tagged `source="sdk"`. |

`/v1/models` is a NIM runtime route and is **not** one of the contract's four
paths, so a NIM without it degrades to a clear error rather than a wrong
answer.

---

## 7. Knobs: `run_mode` is gone

`run_mode` was a preset over two numbers, and it leaked a server detail into
the user's hands.

We checked the server. It distinguishes only two context budgets, the smaller
one reached by `fast`; the client's four-tier `debug/fast/normal/best` ladder
was a client-side fiction. Both numbers are in
[limits](../reference/limits.md).

So the user sets the real numbers, and the client derives the wire value:

```python
query.materialize(indices=[1, 2, 3], num_context_samples=500, num_neighbors=[8, 8])
```

**And the number is honoured, not just used to pick a bucket.** The endpoint's
budget is a ceiling, not a request; gathering 1,000 because that is the bucket
that admits 500 would make the number as decorative as the two preset values
that did nothing, which is the defect this replaced. The count narrows the
gathered examples where they are built, before anything is encoded.

The client **pre-flights the server's limits before sending**, so you get a
readable error instead of a 500, naming the limit and how to get under it.

Every value and where each is checked lives in
[limits](../reference/limits.md), and only there. This page had its own copy of
those tables and they went stale twice: a limit that moved was corrected in the
reference and not here, so the two pages disagreed about what the client does.

---

## 8. Where this code came from

Most of this client is not new. It consolidates implementations that already
existed at NVIDIA, were in production, and had been hardened and tested
against live endpoints. Consolidating them was preferred to rewriting: a
rewrite would have traded working code for new code and kept none of what
those implementations had learned.

What changed on the way, for anyone reading code that looks familiar:

| Area | Change |
|---|---|
| Connectors | Moved into `structured-data-models-connectors` under new names |
| Relational engine, including the C++ neighbour sampler | Moved into `structured-data-models-relational-engine`. `Graph` became `Schema`, and the PQL types now come from `predictive_query` |
| Relational adapter | Returns the unified `Request` and `PredictionResult` |
| Tabular adapter | Model id renamed to `kumo-tabular` |
| Wire types | Live in `structured-data-models-client`, not `structured-data-models-api`, which stays dependency-free (see §2) |
| Predictive Query Language | Maintained here from this point on, in `structured-data-models-pql` |

Two behaviours were carried over deliberately, because each cost something to
learn: every redirect hop is closed rather than drained, and credentials in a
URL are stripped everywhere the URL is rendered.

## 9. Phases

Each phase is independently shippable and tested.

| # | Phase | Ships | Done when |
|---|---|---|---|
| 1 | **Foundation** | Repo, CI, `structured-data-models-api` + `structured-data-models-pql`, SPDX headers, provenance records | `predictive_query` imports clean; generated bindings match the contract hash |
| 2 | **Client + tabular** | `SDMClient`, `Schema`, `Request`, `PredictionResult`, transport, sessions, tabular adapter | a direct `Request` predicts against the tabular NIM |
| 3 | **Connectors** | `structured-data-models-connectors` | every backend reads a table |
| 4 | **Relational** | `structured-data-models-relational-engine`, relational adapter, PQL materialization | `PredictiveQuery(...).materialize(indices=…)` predicts against the relational NIM |
| 5 | **fit + explain** | Session handle, explanation outputs, pre-flight limits | `fit()`/`predict()` reuse one context; every §7 limit fails locally |
| 6 | **Docs + E2E** | Docs, notebook, live-NIM harness for both models | both models pass live E2E |
| 7 | **PQL for tabular** | `materialize(predict=…)` emits a one-table request | a tabular question runs from PQL text |

Phases 2 and 3 are independent. Phase 4 needs phases 1 through 3. Nothing is
blocked on the contract: both shipping models are already in its enum (§0).

---

## 10. This gets open-sourced

Public release was the destination from the first commit, not a maybe. Making
a codebase publishable after the fact means auditing it after the fact. Every
rule below is cheap on day one and expensive later.

**Written OSS-clean from commit 1, and now carrying the rest:**

| Held from commit 1 | Added for the release |
|---|---|
| SPDX header on every source file | `LICENSE` and `NOTICE` at the root, and in every distribution's metadata |
| A provenance record for every vendored tree, CI-enforced | `GOVERNANCE.md`, `MAINTAINERS.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md` |
| Provenance for the generated OpenAPI bindings: spec path, ref, SHA-256, generator version, and a test that they match | Public issue policy, and the repository URL every package publishes to, checked by the hygiene script |
| An automated scan for internal URLs, ticket ids, usernames, engineer TODOs and codenames in shipped packages and docs | None |

Both have since landed. The release workflow publishes through Trusted
Publishing, with no upload token, and attaches PEP 740 attestations. A
generated **SBOM** is produced and validated on the release, covering every
distribution and every connector extra, the native engine included. The
build-time pipeline generates one too, without the native extra, because no
toolchain is present there. A hand-maintained dependency table would not have
counted as either.

The repository is released under **Apache-2.0**, which matches the code this
consolidates, so nothing carried across needs relicensing. The models these
packages talk to are released under OpenMDW, a licence NVIDIA drafted for
models. The two sit side by side and stay distinct: this is software.

### Vendored code is the risk, and we already know how it breaks

We vendor one tree verbatim and rename aggressively across the repo
(`Graph`→`Schema`, `kumo_*`→`sdm_*`). This exact combination already caused
a licensing defect in this family of repos: a repo-wide rename silently rewrote
two identifiers **inside** the minified `mermaid.min.js`, turning a vendored
artifact into a modified one while the LICENSE still described it as upstream's
bytes. Six bytes in a 3.3 MB file. No test caught it, and it invalidated the
component enumeration the licence review had already accepted.

So, as a rule, not a suggestion:

1. Every vendored tree has a **digest pinned beside it and asserted in CI**.
   For `mermaid.min.js`, this is one SHA-256 over the bundle.
2. A test asserts the rename **did not inject target-repo identifiers** into a
   vendored tree. No `sdm_`, `Schema`, or `structured-data-models-client` strings appearing where
   upstream has none. The assertion is "unchanged from upstream", not "contains
   no forbidden word".

> This rule applied to `predictive_query` while that tree was a verbatim copy
> of a separate repository. That repository is being retired, so the tree is
> first-party source of this one now: there is no upstream to diverge from, and
> a manifest pinned to a baseline nobody can fetch would assert nothing. It is
> linted and header-checked like any other first-party tree instead. `mermaid`
> is the only tree the rule still governs.
3. The rename tool **excludes vendored paths and `assets/` by configuration,
   before it runs**. The hash test then verifies the result. Exclusion prevents
   the churn; the hash only detects it.
4. Bumping a vendored bundle requires **re-enumerating its components from the
   new release's manifest**. A version bump silently moves sub-component
   versions too.

This test came across with the rest.

### What ships that we did not write

| Component | License | Where | Note |
|---|---|---|---|
| ANTLR-generated parser | Apache-2.0 | `structured-data-models-pql` | Generated by ANTLR from `PQLGrammar.g4` in the same tree; the grammar and everything around it are first-party |
| ANTLR Python runtime | BSD-3-Clause | `structured-data-models-pql` dependency | Required by the PQL parser |
| `mermaid.min.js` (vendored) | MIT bundle; its components are enumerated in `LICENSE` | `structured-data-models-relational-engine` assets | **Kept.** `visualize()` draws offline. Snowflake and Databricks notebooks block external scripts, so a CDN-fetched diagram would not draw. Upstream bytes, SHA-256 pinned, digest asserted by a test. `to_mermaid()` returns the source for external rendering. |
| pandas, numpy, requests, pydantic | BSD / Apache-2.0 | dependencies | Resolved at install, not vendored |

Any licence review needs the provenance records and the inventory above.
Writing them from the first commit makes that a form-filling exercise rather
than an audit.

---

## 11. Deferred, on purpose

| Thing | Why | What would change it |
|---|---|---|
| Timeseries, anomaly, Tesseract models | Not asked for yet | A NIM serves one; profiles already exist, plus a contract MR (§6) |
| The model reading the warehouse itself | The wire requires embedded data | The contract gains source references |
| Arrow IPC | The wire is JSON-arrays-only everywhere | The contract activates it |

Sampling inside the warehouse is not on this list.
The deferred row is the model reading tables by reference; the client sampling them with SQL and embedding what it fetched fits the wire as it is (§3, schemas over tables that stay where they are).

---

## 12. Open questions for the human

None outstanding. Two earlier ones are settled:

- **The client is `SDMClient`.**
- **The contract's closed `model` enum is not a task.** Both shipping models
  are in it. It only forecloses the escape hatch for a third, which is deferred
  along with the models it would serve (§11). If one is ever wanted, a small
  `structured-data-api` MR opens the field.
