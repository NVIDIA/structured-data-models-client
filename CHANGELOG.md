# Changelog

Kept in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) shape, and
versioned by [semantic versioning](https://semver.org) as
[`docs/reference/stability.md`](docs/reference/stability.md) defines it for
these packages. The three Python distributions share one version.

## Unreleased

### Fixed

- A Snowflake `TIME` column is now read as ISO text (`HH:MM:SS`, with the
  fraction of a second where there is one) instead of as a timestamp, which
  dropped every value of the column (#88).

## 2.0.0 - 2026-11-25

2.0 moves the client's core logic into one shared Rust engine and adds Java support.
Python and Java now build the same requests, raise the same errors and return the same predictions.
This release has breaking changes, so read [Upgrading from 1.0.0](#upgrading-from-100) first.

### Highlights

- **One engine for Python and Java.**
  Validating queries, sampling data, building requests and reading results now happen in one Rust engine inside the client package.
  The separate relational-engine and PQL packages are retired.
- **Java support.**
  You can build the Java library from source; it has the same API as Python and uses the same engine.
- **Your own data sources.**
  One `Source` interface covers the built-in databases and any source you write yourself, in Python or Java.
- **Models without a NIM.**
  `SDMClient.in_process()` runs Kumo Tabular and Kumo Relational inside your Python process.
- **More and better context.**
  Filtered queries now get the full number of context examples they ask for, and every request tells you how its context was gathered.
- **Explanations.**
  `EXPLAIN PREDICT` shows which inputs drove each prediction, for both models.

### Upgrading from 1.0.0

1. **Uninstall the retired packages before upgrading.**

   ```bash
   pip uninstall structured-data-models-relational-engine structured-data-models-pql
   ```

   The old engine pins `structured-data-models-api==1.0.0`, which conflicts with 2.0.
   Write predictive queries with `sdm_client.PredictiveQuery` instead of `predictive_query`.
2. **Import the built-in sources from `sdm_client`.**

   ```python
   from sdm_client import DatabricksSource, DuckDBSource, PostgresSource, SnowflakeSource, SQLiteSource
   ```

   They take the same connections as before.
   Name the class you want: `Source(connection)`, which guessed it from the connection, is removed.
   A semantic model is now `source.semantic_model(name)`, which returns the same source class.
3. **Check your Python and platform.**
   2.0 contains compiled code, so it needs standard Python (CPython) 3.10 or newer on Linux (x86-64 or ARM64), macOS (Intel or Apple silicon) or Windows (x86-64).
   It no longer installs on PyPy, Alpine Linux, 32-bit systems or Windows on ARM.
4. **Update error handling.**
   - An empty table raises `StructuredDataError` (code `INVALID_REQUEST`) instead of `RuntimeError`.
   - A column of an unsupported type, such as a struct or a map, raises `UnsupportedRequestError` instead of `ValueError`.
   - A failing database query raises `StructuredDataError` with code `PROVIDER_ERROR` instead of `INTERNAL_ERROR`.
5. **Expect small differences in predictions.**
   2.0 sends more complete and more exact data than 1.0; see [Improved](#improved).
6. **Features of the old engine that are gone:** Snowflake SPCS serving, the engine's telemetry and progress logging, and the extras `snowflake-serving`, `explain` and `relbench`.
   `generate_summary` is replaced by `ExplanationSummarizer`.
   A schema can no longer be pickled.

### Added

- **Java support.**
  Build the JAR from source as described in [`java/README.md`](java/README.md).
  It has the same API as Python and runs on the same platforms.
  It is not yet published to a Maven repository.
- **Your own data source.**
  Write a class with two methods, `capabilities()` and `answer()`, to read data from any system, such as Trino.
  The client asks it for data and it answers with Arrow tables; it never has to handle SQL from the client.
  See [Your own source](docs/reference/data-providers.md).
- **Running models in process.**
  `SDMClient.in_process()` runs both models in your Python process, with the same calls as a NIM.
  Install it with the `in-process` extra.
- **Explanations.**
  `EXPLAIN PREDICT` returns gradient explanations for both models, including each forecast step.
  `result.explanations` also shows which context columns go with each outcome.
  `ExplanationSummarizer` writes a short summary of an explanation with an LLM.
- **Context report and time limit.**
  `Request.context_report` tells you how many examples were drawn, how many qualified and why drawing stopped.
  `context_time_limit` caps the time spent drawing context (five minutes by default).
- **Schema tools.**
  Infer metadata and links (`infer_metadata`, `infer_links`), review them (`metadata_report`, `link_report`), edit a schema (`with_relationship`, `without_table`, `with_sql_column` and more), and draw it (`to_mermaid`, `to_html`, `visualize`).
- **Questions without a predictive query.**
  `RelationalTask` and `Schema.materialize_examples` ask a question from labelled examples, and `build_timeseries` builds a forecast's input from a series.
  `PredictiveQuery.labels` computes a query's historical labels without calling a model.
- **RelBench datasets** load with `Schema.from_relbench`.
- **Hosted endpoints.**
  `exact_url=True, bearer_token=True` calls an endpoint at one exact URL with a bearer token, such as a model in the NVIDIA API catalog.
- **Snowflake quickstarts** for both models in `examples/snowflake/`.
- **Smaller additions:** `end_time_column` on tables, `InferenceConfig` for relational settings, and `Request.sanitization_report`, which lists rows left out of an in-memory request.

### Improved

- **Large populations in one call.**
  `materialize()` gathers any number of entities, and `predict()` splits them into calls of 10,000 rows and returns one result in order.
  If a call fails, the answers already received are kept on the error, so you only ask again for the rest.
- **No client-side limits.**
  The client no longer applies its own copies of the endpoint's limits, such as 10,000 context rows or a 30 MiB request, so a deployment that allows more is not held back.
  The defaults stay the same.
- **Faster cached Tabular predictions.**
  Predicting with a fitted Tabular session no longer re-encodes the cached context; a one-row prediction went from about 470 ms to 50 ms in our benchmark.
- **Context uploaded once.**
  A prediction split over several calls sends its shared context once, through a temporary session.
- **Fresh schema on every run.**
  Each `materialize()` re-reads the table definitions, so a column dropped or changed since you built the schema is noticed.
- **Newer libraries.** DuckDB 2 and pandas 3.1 are supported.
- **More reliable requests.** An HTTP 408 timeout is retried, like 429, 502, 503 and 504.
- **More complete and exact data,** which can change predictions slightly:
  - a related table sends its own primary key instead of a generated `__node_id`;
  - a foreign key reached by several paths is filled from all of them, not just the first;
  - integer keys stay exact even next to missing values (1.0 rounded keys past 2^53);
  - a column whose name contains a dot is sent instead of dropped.
- **Same results on every platform,** which can also change predictions slightly:
  - rows with equal timestamps, and sampled Tabular context, keep a stable order;
  - `SUM` and `AVG` labels are computed the same way whatever pandas version is installed.

### Fixed

- **Filtered queries got too few context examples.**
  The client sampled 10,000 entities for 1,000 examples and applied the `WHERE` afterwards, so a filter that keeps 1 entity in 60 left only about 170.
  The conditions now run in the database while sampling, so the model gets all 1,000 examples, which should improve predictions, and the database sends only the rows that qualify.
  This includes windowed conditions such as `COUNT(orders.*, -30, 0, days) > 1`.
- **Text columns made Kumo Relational refuse the request.**
  A `text` column is now left out of the request with a warning, since the model does not read text.
- **Queries longer than 1,024 characters** were refused by the NIM; they are now answered.
- **Databricks timestamps** depended on the client machine's time zone; they are now read in UTC.
- **Databricks decimals** lost precision (`997.6700000000001` instead of `997.67`); they are now sent exactly.
- **Large PostgreSQL `NUMERIC` keys** above `2**63 - 1` raised `OverflowError`; they are now kept exactly.
- **pandas 3.1:** a relational prediction over tables without a time column failed; it now works.
- **DuckDB 2:** `read_table` made an extra copy and printed a warning; it no longer does.
- **Link predictions without candidates** were sent and then refused by the endpoint; the client now refuses them right away with a clear error.
  Use `RANK TOP k` with `candidates=` instead.

### Removed

- The `structured-data-models-pql` and `structured-data-models-relational-engine` packages.
- `sdm_connectors.Source`, `DatabricksSource`, `SnowflakeSource`, `PostgresSource`, `DuckDBSource`, `SQLiteSource` and `SemanticModel`; use the classes in `sdm_client`.
- `Source(connection)` and `Source.backend`.
- The old engine's Snowflake SPCS serving, telemetry, progress logging, and the `snowflake-serving`, `explain` and `relbench` extras.
- The `[relational]` extra now installs nothing; it is kept so existing install commands still work.

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
