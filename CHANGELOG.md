# Changelog

Kept in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) shape, and
versioned by [semantic versioning](https://semver.org) as
[`docs/reference/stability.md`](docs/reference/stability.md) defines it for
these packages. All five distributions share one version and one entry.

## Unreleased

### Changed

- A relational `materialize()` takes more entities than one call scores. They
  are gathered in batches of 10,000, or 200 for a temporal link prediction,
  and `predict()` sends one call per batch and returns one result in the order
  the indices were given. It used to refuse more than 1,000 and ask for the
  calls to be split by hand.

  `materialize()` takes `rows_per_call` to ask in smaller batches than the
  endpoint's cap, which is what a graph wide enough to outrun the 30 MiB body
  needs, and what a deployment holding to less than 10,000 needs. It only goes
  down: a larger call buys a refusal rather than a bigger call. Where the
  context is itself over the body, no batch is small enough, and the refusal
  says which options do come down instead.

  Two things are still refused. Every batch is built before the first is sent,
  so more than 100,000 entities, or more than 100 calls, in one
  `materialize()` is refused rather than held in memory at once; and a
  question asked in parts, a ranking or a forecast, is not batched, because
  each part is already a call and the fan-out budget counts those.

- A call that fails part way through carries the answers already paid for on
  the error, as `details['answered']` and `details['rows_answered']`, rather
  than discarding them. A question asked in parts keeps its answered parts the
  same way.

- In the relational engine, `batch_mode(batch_size='max')` now means 10,000
  entities a batch rather than 1,000, following the size the endpoint answers.
  A graph wide enough that 10,000 entities exceed the 30 MiB body is refused
  on the first batch where it used to be gathered as batches of 1,000; pass an
  explicit `batch_size=` for the old behaviour.

## 1.0.0 - 2026-10-13

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

**`predictive_query` is not public API.** Write predictive queries through
`sdm_client.PredictiveQuery`.

**A public name hands out a public type**, bar a few exceptions recorded with
their reasons in `tests/test_release_contract.py`, which can only shrink.

### Compatibility

- The four pure-Python distributions support Python 3.10 to 3.14.
- The relational engine ships Linux wheels only, for CPython 3.10 to 3.13 on
  manylinux_2_28 x86-64 and aarch64. There is no source distribution, so
  `structured-data-models-client[relational]` does not install on macOS,
  Windows, musl, or an older glibc.
- What a version bump means is in
  [`docs/reference/stability.md`](docs/reference/stability.md); how a request
  is built and sent is in [the architecture
  notes](docs/about/architecture.md).
