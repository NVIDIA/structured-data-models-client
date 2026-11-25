# Changelog

Kept in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) shape, and
versioned by [semantic versioning](https://semver.org) as
[`docs/reference/stability.md`](docs/reference/stability.md) defines it for
these packages. The three Python distributions share one version.

## Unreleased

## 2.0.0 - 2026-11-25

2.0 rebuilds the client on one Rust core and adds a Java library on top of it.
The core prepares requests, samples data and assembles results, so Python and Java callers get the same requests, the same errors and the same predictions.

Highlights:

- **One engine for Python and Java.**
  `structured-data-models-client` is now a native wheel that contains the Rust core.
  The separate relational-engine and PQL packages are retired.
- **A Java library** with the same API as Python's, built and tested in CI.
  It is not yet published to a Maven repository.
- **One `Source` interface.**
  The built-in sources for DuckDB, SQLite, PostgreSQL, Databricks and Snowflake implement it, and so can your own code, in either language.
- **Models in process.**
  `SDMClient.in_process()` runs Kumo Tabular and Kumo Relational in the Python process, through the same calls as a NIM.
- **Better context for filtered queries.**
  A query that keeps few entities now gathers the context it asks for, and every request reports how its context was drawn.
- **Gradient explanations** for both models.

This is a major release with breaking changes; read [Upgrading from 1.0.0](#upgrading-from-100) before you upgrade.

### Upgrading from 1.0.0

1. Uninstall the retired packages first.
   The 1.0.0 engine pins `structured-data-models-api==1.0.0`, which conflicts with 2.0.

   ```bash
   pip uninstall structured-data-models-relational-engine structured-data-models-pql
   ```

   Importing `sdm_relational_engine` or `predictive_query` now fails.
   Write predictive queries with `sdm_client.PredictiveQuery`.
2. Import the built-in sources from `sdm_client` instead of `sdm_connectors`:

   ```python
   from sdm_client import DatabricksSource, DuckDBSource, PostgresSource, SnowflakeSource, SQLiteSource
   ```

   They take the same connections and raise the same `ConnectorError` codes.
   `Source(connection)`, which picked a class from the connection's driver, is removed: name the class instead.
   `semantic_model(...)` returns a source of the same class, and `SemanticModel` is removed.
3. Check your platform.
   The client wheel is `cp310-abi3` for Linux x86-64 and aarch64 (manylinux), macOS x86-64 and arm64, and Windows x86-64.
   It no longer installs on PyPy, musl Linux, 32-bit platforms or Windows ARM64.
4. Check code that catches errors (see [Changed](#changed)).
   An empty table raises `StructuredDataError` instead of `RuntimeError`, an unsupported column type `UnsupportedRequestError` instead of `ValueError`, and a failing data source `PROVIDER_ERROR` instead of `INTERNAL_ERROR`.
5. Expect some predictions to differ slightly from 1.0 (see [Changed](#changed)).

### Added

- A Java library with the same API as Python's, for the same platforms, built from the shared Rust core.
  It is built and tested in CI but not yet published.
- `Source`, one interface for every data source in Python (`sdm_client.Source`, an abstract base class) and Java (`com.nvidia.sdm.Source`).
  Implement `capabilities()` and `answer()` to read data through your own code instead of a database connection, such as a Trino cluster.
  The core sends typed data requests, never SQL, and your source answers each with Arrow data.
  See [Your own source](docs/reference/data-providers.md) and the Trino example in both languages.
- `SDMClient.in_process()` runs `kumo-tabular` and `kumo-relational` in the Python process with structured-data-models, with no NIM.
  It supports predictions, fitted sessions, rankings, forecasts and explanations, and installs with the `in-process` extra.
- Gradient explanations for both models: `EXPLAIN PREDICT` returns them, including for each forecast step.
  Python exposes `result.explanations` and `explanation_view`; Java exposes `explanations()`.
  There is no longer a one-row limit on explanations.
- `Request.context_report` (Java `contextReport()`) says how the context examples were drawn and why drawing stopped.
- `context_time_limit` (Java `MaterializeOptions.Builder.contextTimeLimit`) bounds the time spent drawing context, five minutes by default.
- `Request.sanitization_report` says which rows were left out of an in-memory relational request, and why.
- Schema tools: `infer_metadata`, `infer_links`, `metadata_report`, `link_report` and `link_origins`; `with_relationship`, `without_relationship`, `without_table`, `with_source` and `with_sql_column`; and `to_mermaid`, `to_html` and `visualize` for an offline diagram.
- `Schema.from_relbench` loads pinned RelBench datasets, with verified caches and offline mode.
- `RelationalTask` and `Schema.materialize_examples` ask a relational question from labelled examples, without a predictive query.
  `build_timeseries` builds a forecast's input from a series.
- `PredictiveQuery.labels` computes a query's historical labels without calling a model.
- `Table.end_time_column` and `SourceTable.end_time_column` name a row's end time.
- `InferenceConfig` holds relational inference settings.
- `ExplanationSummarizer` summarizes an explanation with an LLM endpoint.
- `SDMClient(..., exact_url=True, bearer_token=True)` (Java `ClientOptions.builder().exactUrl(true).bearerToken(true)`) calls an endpoint at one exact URL with a bearer token, such as a hosted model in the NVIDIA API catalog.
- A Snowflake session the client opens identifies itself to Snowflake as `nvidia/kumo-sdm`, unless you named the application yourself.
- Snowflake Container Runtime quickstarts for Kumo Relational and Kumo Tabular in `examples/snowflake/`.

### Changed

- **Breaking:** the built-in sources moved from `sdm_connectors` to `sdm_client`.
  `sdm_connectors` keeps its readers, `quote_ident`, `resolve_sql` and the Databricks connection helpers.
- **Breaking:** the client wheel is native (`cp310-abi3`) instead of pure Python; see [Upgrading](#upgrading-from-100) for platforms.
- **Breaking:** describing an empty table raises `StructuredDataError` with code `INVALID_REQUEST` (`details['rule']` is `empty_table`), so `except RuntimeError` no longer catches it.
- **Breaking:** a column of an unsupported type, such as a struct, a map or a list of booleans, raises `UnsupportedRequestError` with code `UNSUPPORTED_REQUEST`, naming the column, table and type, instead of `ValueError`.
- **Breaking:** a failure in a data source, such as a warehouse query that errors, raises `StructuredDataError` with code `PROVIDER_ERROR` instead of `INTERNAL_ERROR`; `details['driver_class']` names the driver's exception.
- **Breaking:** a temporal link prediction without candidates, such as `PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) FOR EACH users.user_id`, is refused with `UNSUPPORTED_REQUEST` before any data is read, because no released model serves it.
  Rank an explicit candidate set with `RANK TOP k` and `candidates=` instead.
- **Breaking:** `Schema.from_source` and `with_source` take only a `Source`; an object that merely has the same methods is refused.
- The client no longer enforces its own copies of the endpoint's limits, so a deployment that raises them is not held back.
  This lifts the 10,000-row context and predict caps, the 30 MiB request size, the 15-table cap and the `materialize()` entity and call guards; the endpoint refuses what it does not take.
  The defaults stay: 10,000 context rows and 10,000 predict rows per call.
- A schema reads each table's catalog again on every `materialize()`, so a column dropped or retyped since the schema was built is noticed.
- `predict()` over several calls that share one context uploads the context once, through a temporary endpoint session.
- An HTTP 408 from the endpoint is retried, as 429, 502, 503 and 504 are.
- A column whose name contains a dot is sent; 1.0 left it out with a warning.
- A `text` column is left out of a Kumo Relational request, with a warning, since the model does not read text.
- A schema refuses to be pickled.
- Predictions can differ slightly from 1.0 because the data sent is more exact or more consistent:
  - a related table sampled from frames sends its own primary key, where 1.0 sent a generated `__node_id`;
  - a foreign key with nulls is sent as exact integers, where 1.0 sent floats and rounded keys past 2**53;
  - a row reached over three or more hops fills its foreign keys from every path, where 1.0 used only the first;
  - rows with equal timestamps, and sampled Tabular context, keep a stable order on every platform;
  - `SUM` and `AVG` labels are computed the same way on every pandas version, so a label can differ in its last digit;
  - a filtered query draws its context from the entities that qualify, so the same seed draws different entities.

### Removed

- `structured-data-models-pql` (`predictive_query`) and `structured-data-models-relational-engine` (`sdm_relational_engine`) are no longer distributed.
- `sdm_connectors.Source`, `DatabricksSource`, `SnowflakeSource`, `PostgresSource`, `DuckDBSource`, `SQLiteSource` and `SemanticModel`; use the classes in `sdm_client`.
- `Source(connection)`, which chose a source class from the connection's driver, and `Source.backend`.
- Features that lived only in the relational engine: Snowflake SPCS serving, the engine's telemetry and progress logging, and the extras `snowflake-serving`, `explain` and `relbench`.
  RelBench datasets load with `Schema.from_relbench`.
  `generate_summary` is replaced by `ExplanationSummarizer.summarize(view, query, prediction)`, which raises on failure instead of returning fallback text.
- The `[relational]` extra installs nothing; it is kept so existing install commands still work.

### Fixed

- A query whose `WHERE` keeps few entities, or whose target most entities lack, gathers the context it asks for.
  The query's conditions go to the source with the sample, so context is drawn from entities that may qualify; before, a filter keeping one entity in sixty gathered about a sixtieth of the context, or none.
  This covers windowed conditions such as `WHERE COUNT(orders.*, -30, 0, days) > 1` on DuckDB, PostgreSQL, Snowflake and Databricks.
- A predictive query longer than 1,024 characters is answered; a NIM refused it before.
- On Databricks through Spark, `TIMESTAMP` columns are read in UTC, so results no longer depend on the client machine's time zone.
- On Databricks, a `DECIMAL` value with a fraction is sent exactly (`997.67`, not `997.6700000000001`).
- On PostgreSQL, a whole-number `NUMERIC` key above `2**63 - 1` is kept exactly, where 1.0 raised `OverflowError`.
- A relational prediction over a graph without a time column works on pandas 3.1.
- `read_table` on DuckDB avoids the extra copy and warning from DuckDB 2.0's cursor.
- The wheel's notices name the IANA Time Zone Database its native module embeds, and the installation docs say that the `postgres` and `lakebase` extras install LGPL-3.0-only psycopg.

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
