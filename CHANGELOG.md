# Changelog

Kept in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) shape, and
versioned by [semantic versioning](https://semver.org) as
[`docs/reference/stability.md`](docs/reference/stability.md) defines it for
these packages. The three Python distributions share one version.

## Unreleased

The release candidate version is `2.0.0rc1`, spelled `2.0.0-rc.1` in Rust's `Cargo.toml`.
The Java library is built and tested in CI but is not released.
The public API snapshot records this version update; publication remains pending.

### Added

- Python `Schema.from_relbench` and Java `Schema.fromRelbench` load the same pinned
  RelBench datasets with verified caches, offline mode and declared schema metadata.
- Python and Java requests and fitted handles expose immutable sanitization reports for in-memory relational preparation, with per-table row exclusions and explicit unavailable/not-applicable states.

- Every `sdm_connectors` source says its kind and its namespace in one way:
  `Source.kind` (`'databricks-sql'` for a SQL warehouse connection,
  `'databricks-spark'` for a notebook's Spark session,
  `'databricks-metric-view'` or `'snowflake-semantic-view'` for a semantic
  model, else the backend) and `Source.namespace_parts()` (the namespace's
  parts, outermost first, `None` where the connection's current one is
  meant).
- `SDMClient.in_process()` runs `kumo-tabular` and `kumo-relational` in this
  Python process with structured-data-models, through the same calls as a NIM:
  predictions, fitted sessions, rankings, forecasts and explanations. It loads each model
  from its published weights on first use, or uses one passed in. The request
  is prepared as it is for a NIM and the model run on it directly, the context
  fitted as the NIM fits it. A request the model cannot answer is refused with
  a `StructuredDataError` whose code says why. Install it with the
  `in-process` extra.
- A Snowflake session the client opens names itself to Snowflake as
  `nvidia/kumo-sdm`, so Snowflake can attribute its queries. An application
  named any other way is kept: an `application` argument, one in the
  `connections.toml` profile, `SF_PARTNER`, or the connector's own detection
  of Streamlit and notebooks. A connection or Snowpark session you open
  yourself is used as it is.
- The client now contains the Rust core instead of depending on separate
  relational-engine and Python PQL packages.
  Python wheels target Linux x86-64/ARM64, macOS x86-64/ARM64 and Windows
  x86-64; the Java library, built from source, supports the same platforms.
  Native Windows ARM64 is deferred pending upstream Arrow dependencies.

- Both models support gradient explanation collections when the model runtime supports them.
  Python exposes `result.explanations` and its pandas view; Java exposes `explanations()` and its Arrow view.
  `EXPLAIN PREDICT` requests explanations, including each forecast step; client-assembled rankings refuse them.
  Python also accepts manually supplied `PredictionResult(explanations=...)` mappings.

- `Schema` gains `infer_metadata`, `infer_links`, `link_report`,
  `metadata_report` and `print_metadata`/`print_links`, `link_origins`
  (as `RelationshipOrigin` pairs), `with_relationship`, `without_relationship`,
  `without_table`, `with_source` and `with_sql_column`, and `to_mermaid`,
  `to_html` and `visualize` for an offline diagram.
- `RelationalTask` and `Schema.materialize_examples` ask a relational
  question from labelled examples without PQL; `build_timeseries` builds a
  forecast's input from a series.
- `PredictiveQuery.labels` computes a query's historical labels without
  calling a model.
- `InferenceConfig` holds relational inference settings.
- `ExplanationSummarizer` writes a short summary of an explanation with an
  LLM endpoint.
- `Table.end_time_column` and `SourceTable.end_time_column` name a row's end
  time.
- `PredictionResult.merge(parts, columns=...)` takes the column analysis to
  keep on the merged result.

### Changed

- `structured-data-models-client` is now a native wheel, `cp310-abi3`, built
  for Linux x86-64 and aarch64 (manylinux), macOS arm64 and x86-64, and Windows
  AMD64. 1.0.0 was pure Python. It no longer installs on PyPy, musl Linux,
  32-bit platforms or Windows ARM64.
- A failure in the data source a schema reads, such as a warehouse query that
  errors, raises `StructuredDataError` with code `PROVIDER_ERROR`, and
  `details['driver_class']` names the driver's exception class. 1.0.0 raised
  these from the relational engine as `INTERNAL_ERROR`. A semantic view or
  metric view whose expression names a column its base table lacks is one of
  them: the warehouse refuses the read, where 1.0.0 refused the view as
  `INVALID_REQUEST` before reading it.
- A column of a type a described table does not take, such as a list of
  booleans, a struct or a map, raises `UnsupportedRequestError` with code
  `UNSUPPORTED_REQUEST` and `details['rule']` `column_type`. The message and
  details name the column, its table and its type, as the source declares it
  where it does, such as a Snowflake `VARIANT`. 1.0.0 raised a `ValueError`
  naming the column and table.
- Describing an empty table raises `StructuredDataError` with code
  `INVALID_REQUEST` and `details['rule']` `empty_table`. 1.0.0 raised a
  `RuntimeError`, so `except RuntimeError` no longer catches it.
- A column whose name holds a dot is kept and sent. 1.0.0 left it out with a
  warning.
- `predict()` on a request gathered in several calls that share one context
  scores them through a temporary endpoint session, so the context is
  uploaded once. 1.0.0 sent every call stateless. An expired session is
  recreated once; if the endpoint loses it again, as replicas that do not
  share sessions do, the remaining calls are sent stateless.

- A related table sampled from frames sends its own primary key, where
  1.0.0 sent a generated `__node_id`.
- Each `materialize()` on a schema from `Schema.from_source` reads its
  tables' catalog again, one query per table, so a column dropped or
  retyped since is found. A table whose catalog declares a column otherwise
  is read again in full, its keys and first rows included, as are SQLite's
  tables and a semantic model's, with its definition. 1.0.0 read them once
  and kept them on the schema. A table emptied since it was described is
  refused as empty by a warehouse's row count; elsewhere it reads as a
  table with no times, as 1.0.0 read it.
- An HTTP 408 from `POST /v1/predictions` is retried, as 429, 502, 503 and
  504 are.
- A foreign key holding integers and nulls is sent as exact integers beside
  nulls. 1.0.0 sent such a column as `float64`, so its keys arrived as floats
  and any past 2**53 was rounded.
- A relational request reached over three or more hops fills a foreign key
  from every path that reaches the row, where 1.0.0 took the first path and
  left the rest null. The model sees more links, so predictions can differ.
- Related rows with the same timestamp keep their source order. 1.0.0 used
  NumPy's unstable sort, whose order depends on the platform, so sampled
  rows and predictions can differ where timestamps tie.
- `SUM` and `AVG` labels, and the lagged target columns built from them, are
  summed in float32 with Kahan compensation, as
  pandas' Cython grouped sum computes them. 1.0.0 left this to the installed
  pandas, whose answer depends on its version: pandas 3.1 sums sorted groups
  with NumPy and averages in float64. A label can differ in its last digit,
  and a threshold on it can flip.
- Sampled Tabular context uses stable Rust index ordering, including duplicate labels and mixed numeric labels.
  Equal labels retain sampled order; mixed numbers compare by their represented values.
  Context order and predictions can differ from platform-specific NumPy ordering; sampled membership and warehouse sampling are unchanged.

- Requests, sampling and result assembly run through Rust for both languages.
  Local Python model execution uses the same prepared requests and responses.
- Ordinary inference options are validated in Rust before local or remote model
  execution, including fitted sessions, for both Python and Java.
  NumPy scalar options are accepted at Python request admission.

- Gradient explanations have no separate one-row or 1,000-context-row client restriction.
  Endpoint capabilities and ordinary request limits still apply.
  The singular raw `result.explanation` view remains supported.
- A temporal link prediction asked without candidates, such as
  `PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) FOR EACH users.user_id`,
  raises `UnsupportedRequestError` with code `UNSUPPORTED_REQUEST` and
  `details['rule']` `temporal_link_prediction`. `materialize()` and
  `materialize_examples()` refuse it before any source is read, and
  `predict()` refuses a request that names the task before anything is sent.
  No released Kumo Relational model serves it, so `kumo-relational` no longer
  lists it among its tasks. Rank an explicit candidate set with `RANK TOP k`
  and `candidates=` instead. 1.0.0 built the request, and the endpoint
  refused it.

### Removed

- The client no longer enforces its own copies of the endpoint's limits, so a deployment that raises them is not held back: the 10,000 relational context rows (`num_context_samples` may be larger), the 10,000 predict rows per call (`rows_per_call` may be larger; 10,000 stays the default batch), the tabular refusal past 10,000 context rows (`materialize()` still samples to 10,000 by default, with a warning), the 30 MiB request body, the 15-table and 6-hop caps, and the 100,000-entity and 100-call guards on `materialize()`.
  The endpoint refuses what it does not take.
  The Databricks gateway's 16 MB check, `MOST_PARTS`, `top_k` of at most 20 and the 64 MiB response cap stay.
  The contract is version 73.

### Fixed

- Explicit empty outputs remain invalid with `EXPLAIN`; unsupported local-runtime outputs are refused before model weights load.
- Gradient identities and merging run in Rust, with generated transport keys distinguished from genuine source keys.
  Merged explanations and predictions now use matching row indices, including repeated result parts.

- A predictive query longer than 1,024 characters is answered. A relational
  request carries the query in `metadata.source_query` for logging, which the
  contract caps at 1,024 characters, so a NIM refused a longer one with
  `VALIDATION_FAILED`. The logged copy is now cut to fit.
- `read_table` reads DuckDB results through a relation, avoiding the extra
  copy and warning introduced by the DuckDB 2.0 DB-API cursor.
- A relational prediction over a graph without a time column works on pandas
  3.1. Its anchor is the earliest timestamp pandas holds, and 3.1 refused to
  take even an empty date offset from it.
- A request's description leaves out a wire table carrying an integer past
  the float range on every pandas release. pandas 3.1 had started to build
  a frame around one, where earlier releases refused it.
- On Databricks through Spark, a `TIMESTAMP` column is read in UTC, so the
  anchor time and the rows a window takes no longer move with the client
  machine's time zone. 1.0.0 read its range as local times.
- On Databricks, through Spark or a SQL warehouse, a `DECIMAL` value with a
  fraction is sent exactly, such as `997.67`. 1.0.0 sent it through a float, such as
  `997.6700000000001`.

### Removed

- `structured-data-models-pql` (`predictive_query`) and
  `structured-data-models-relational-engine` (`sdm_relational_engine`) are no
  longer distributed, and importing either fails. Write predictive queries
  through `sdm_client.PredictiveQuery`.
- The `[relational]` extra installs nothing extra. It is kept so existing
  install commands still work.
- Features that lived only in the relational engine are gone: Snowflake SPCS
  serving, the engine's telemetry and progress logging, and its extras
  `snowflake-serving`, `explain` and `relbench`. RelBench datasets load with
  `Schema.from_relbench`. The LLM summary of an explanation is now
  `ExplanationSummarizer`, a different API: `summarize(view, query, prediction)`
  takes an `ExplanationView` and raises on failure, where 1.0.0's
  `generate_summary` returned fallback text.

### Upgrading from 1.0.0

Uninstall the two retired packages before upgrading:

```bash
pip uninstall structured-data-models-relational-engine structured-data-models-pql
```

The 1.0.0 engine pins `structured-data-models-api==1.0.0`, which conflicts
with 2.0.

## 1.0.0 - 2026-10-15

First public release. The five distributions are versioned and published
together, and a given version of one is tested only against the same version
of the others.

### Added

- A typed client for the `kumo-tabular` and `kumo-relational` models: one-shot
  predictions, fitted sessions that reuse a pinned context, predictive queries,
  and results as pandas frames.
- `Schema.from_data` for tables already in memory, and `Schema.from_source` for
  sampling tables where they live, without loading a warehouse into pandas.
- Connectors for SQLite, DuckDB, PostgreSQL, Lakebase, Snowflake, Databricks,
  local CSV and Parquet, and S3.
- Conversion of a Databricks metric view and a Snowflake semantic view into a
  schema, including the metric view's filter, and guarded evaluation of the
  expressions a view's author wrote.
- `SDMClient.for_databricks_serving(endpoint)` for a model served on Databricks,
  with `warehouse()` and `lakebase()` for the SQL warehouse and Lakebase.
- A Databricks get-started notebook,
  `examples/databricks/kumo_relational_quickstart.ipynb`, which deploys the
  model to a serving endpoint, loads the AdventureWorks sample tables into
  your schema, and checks each prediction against what actually happened.
- A Databricks metric view name may hold a part SQL reads only quoted, such as
  the catalog in `my-catalog.sales.metrics`, without backticks.
- Wheels for the relational engine on CPython 3.10 to 3.13, manylinux_2_28,
  x86-64 and aarch64.
- A stated compatibility and deprecation contract, in
  [`docs/reference/stability.md`](docs/reference/stability.md).
- Publishing to PyPI through NVIDIA's Kitmaker service, which owns the PyPI
  project, and a CycloneDX SBOM attached to each GitHub Release.
- A relational `materialize()` gathers more entities than one call scores as
  several calls: batches of 10,000, or 200 for a temporal link prediction, sent
  one call each and returned as one result. Each call's rows are numbered from
  where the last call's ended, so `to_pandas()`, which orders by that number,
  gives the indices back in the order they were asked for rather than
  interleaving the batches. `materialize(rows_per_call=)` asks in smaller
  batches than the endpoint's cap, which is what a graph wide enough to outrun
  the 30 MiB body needs, and what a deployment holding to less than 10,000
  needs. It only goes down: a larger call buys a refusal rather than a bigger
  call.
- A call that fails part way through carries the answers already paid for on
  the error, as `details['answered']` and `details['rows_answered']`, rather
  than discarding them. A forecast keeps its answered steps the same way, with
  how many under `details['parts_answered']`. A ranking keeps only that count,
  because an order over some of its candidates is a different answer rather
  than part of the one that was asked for.
- In the relational engine, `batch_mode(batch_size='max')` means 10,000
  entities a batch, following the size the endpoint answers.

### Distributions

| Distribution | Import |
| --- | --- |
| `structured-data-models-client` | `sdm_client` |
| `structured-data-models-api` | `sdm_api` |
| `structured-data-models-pql` | `predictive_query` |
| `structured-data-models-connectors` | `sdm_connectors` |
| `structured-data-models-relational-engine` | `sdm_relational_engine` |

They pin each other by exact version and are upgraded together.
`predictive_query` is an internal dependency: write predictive queries through
`sdm_client.PredictiveQuery`.

### Behaviour worth knowing before you rely on it

**`materialize()` picks the model from the schema's shape unless you name
one.** A schema sampled in place, or holding more than one table, or holding a
relationship, is gathered relationally; anything else is tabular. Pass
`model=` when that is not what you want.

**`id` holds the key you asked about, on both models.**

**A time column is inferred by rule, not by name alone.** A timestamp empty in
more than half the rows is not taken, nor one holding a single value wherever
it holds any unless the table has one row, nor one named for when a row was
loaded or synced, nor a date of birth. A last-edit stamp such as
`ModifiedDate` is not taken on a table another table links to, and on a table
nothing links to only when it is the one timestamp left. Of several left, a
`created` stamp wins, otherwise the one whose values start earliest, with
placeholder dates such as 1900-01-01 and 1970-01-01 ignored. These read the
whole table, and a column they could not read stays a candidate. Pass
`time_columns=` to decide it yourself.

**A `DECIMAL` column travels as the number its type allows.** A decimal
declared with no scale, holding whole numbers that fit, is sent as `int64`;
one too wide for `int64` is sent as its digits in a string, which keeps every
one of them; anything else, including a decimal whose declared scale is above
zero, is sent as `float64`. A value a `float64` JSON reader would not read
back exactly is refused rather than quietly rounded. Send the column as
strings to keep a fractional decimal exactly.

**A CSV or Parquet column with blanks keeps its type instead of becoming
`float64`.** That is what stops an identifier past 2**53 being rounded before
the wire rules could catch it.

**`Schema.from_source` counts each key's repeats and missing values by
default.** A schema that would once have been built quietly may now warn or
raise. Pass `check_keys=False` to skip it.

**A semantic model's expressions may call only pure, deterministic built-in
functions, from an allowlist per dialect.** Pass `trust_expressions=True` to
keep a column that calls anything else.

**A batched `materialize()` is bounded, and a question asked in parts is not
batched.** Every batch is built, with its own context, before the first is
sent, so more than 100,000 entities, or more than 100 calls, in one
`materialize()` is refused rather than held in memory at once. A ranking or a
forecast is asked in parts, each part is already a call, and the fan-out budget
counts those. Where the context is itself over the 30 MiB body no batch is
small enough, and the refusal says which options do come down instead.

**`predictive_query` is not public API.** Write predictive queries through
`sdm_client.PredictiveQuery`.

**A public name hands out a public type**, bar a few exceptions recorded with
their reasons in `tests/test_release_contract.py`, which can only shrink.

### Compatibility

- The four pure-Python distributions support Python 3.10 to 3.14.
- The `snowflake` extra holds `snowflake-connector-python` below 5. The 5.0
  driver changed the shape of its connection surface, and this release is
  tested against 4.
- The relational engine ships Linux wheels only, for CPython 3.10 to 3.13 on
  manylinux_2_28 x86-64 and aarch64. There is no source distribution, so
  `structured-data-models-client[relational]` does not install on macOS,
  Windows, musl, or an older glibc.
- What a version bump means is in
  [`docs/reference/stability.md`](docs/reference/stability.md); how a request
  is built and sent is in [the architecture
  notes](docs/about/architecture.md).
