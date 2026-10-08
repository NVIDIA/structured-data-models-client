# Changelog

Kept in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) shape, and
versioned by [semantic versioning](https://semver.org) as
[`docs/reference/stability.md`](docs/reference/stability.md) defines it for
these packages. All five distributions share one version and one entry.

## 1.0.0 - 2026-10-08

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
- Wheels for the relational engine on CPython 3.10 to 3.13, manylinux_2_28,
  x86-64 and aarch64.
- A stated compatibility and deprecation contract, in
  [`docs/reference/stability.md`](docs/reference/stability.md).
- Publishing through PyPI Trusted Publishing with PEP 740 attestations, and a
  CycloneDX SBOM attached to each GitHub Release.

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

**`materialize()` takes the model it is gathering for.** It does not infer one.

**`id` holds the key you asked about, on both models.**

**A schema infers a time column by rule, not by name alone.** A timestamp
empty in more than half the rows, or holding one value wherever it holds any,
is never taken, nor is one named for when a row was loaded or synced, nor a
date of birth. A last-edit stamp such as `ModifiedDate` is never taken on a
table another table links to, and on a table nothing links to only when it is
the one timestamp left. Of several left, a `created` stamp wins, otherwise the
one whose values start earliest, placeholder dates such as 1900-01-01 and
1970-01-01 ignored. Pass `time_columns=` to decide it yourself.

**A `DECIMAL` column travels as the number it is, and a fractional one
rounds.** A whole-number decimal is sent as `int64`, one too wide for `int64`
as its digits in a string, and a fractional one as `float64`, which rounds.
Above 2**53 a float cannot hold even the integer part, so the value is refused
rather than quietly mangled. Send the column as strings to keep a fractional
decimal exactly.

**A CSV or Parquet column with blanks keeps its type instead of becoming
`float64`.** That is what stops an identifier above 2**53 being rounded on the
way in.

**`Schema.from_source` checks every key against every row**, so a schema may
warn or raise where the keys do not hold.

**A semantic model's expressions may call only standard SQL functions**,
unless you pass `trust_expressions=True`.

**`predictive_query` is not public API.** Write predictive queries through
`sdm_client.PredictiveQuery`.

**Nothing public hands out a type that is not public.**

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
