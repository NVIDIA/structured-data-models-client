# sdm-connectors

Shared data-source connectors for the [`sdm-client`](../sdm-client/README.md) client and its
model engines. One place that knows how to reach each warehouse, so the client
(flat table reads) and the Kumo Relational driver (warehouse connections for
its graph samplers) don't each carry their own copy.

It provides:

- `connect(backend, **kwargs)`: open a connection to `sqlite` / `duckdb` /
  `snowflake` / `databricks` (one driver per backend).
- `read(source, **kwargs) -> DataFrame`: read a table or query as a flat pandas
  DataFrame (also handles `local` DataFrames / CSV / Parquet, and `s3` object
  URIs: `read('s3', path='s3://bucket/table.parquet', storage_options=...)`).
- `read_table(connection, table=…, query=…) -> DataFrame`: driver-agnostic
  fetch over an open connection, in that connection's own session.
- `quote_ident(ident, char='"')` and `resolve_sql(table=…, query=…)`: SQL
  identifier quoting and a safe table/query guard.

File reads (`local` and `s3`) accept `.csv` / `.txt` (optionally compressed) and
`.parquet` / `.pq` / `.parq`, plus a directory as a Parquet dataset. Any other
suffix is rejected with `INVALID_CONNECTOR_ARGS` rather than parsed as CSV; pass
`format='csv'` or `format='parquet'` to read a file whose name carries no
recognised suffix.

The two file-backed backends, `sqlite` and `duckdb`, accept `database=` and
`uri=` as aliases for the same argument (supplying both is an error). The two
warehouse backends, `snowflake` and `databricks`, are addressed by connection
keywords instead, and reject any their driver does not declare,
`driver_options={...}` passes anything else straight through. The
`snowflake` backend reuses an active Snowpark session when no authentication
arguments are given; a borrowed session cannot be reconfigured, so passing
session-scoped arguments such as `schema=` alongside it is an error.

`table=` accepts only plain, unquoted, dot-separated ASCII identifiers, and
interpolates them as written, so they are subject to each backend's default
case folding (Snowflake upper-cases, Databricks lower-cases). A name that needs
quoting (spaces, non-ASCII characters, a leading digit) is rejected with
`INVALID_CONNECTOR_ARGS`. A reserved word such as `select` is a plain
identifier by that rule, so it is accepted here and instead fails at execution
as `QUERY_FAILED`. Either way, reach the table through `query=` with
`quote_ident`, choosing the quote character your backend uses (`"` for
SQLite/DuckDB/Snowflake, `` ` `` for Databricks):

```python
read('duckdb', database='w.db', query=f'SELECT * FROM {quote_ident("café")}')
```

Failures raised by `connect` / `read` / `read_table` surface as
`ConnectorError` with a stable `code`: `UNKNOWN_CONNECTOR`,
`INVALID_CONNECTOR_ARGS`, `CONNECT_FAILED`, `QUERY_FAILED`, `READ_FAILED`, or
`NOT_FOUND`, with the original driver exception chained as `__cause__`. Two
cases stay unwrapped by design: a missing optional driver raises
`MissingBackendError`, and a broken or incompatible driver installation raises
`ImportError` (consumers map it to their own broken-install category, e.g.
sdm-client's `DRIVER_LOAD_FAILED`).

Pure-python. Database drivers are optional extras:

```bash
pip install "sdm-connectors[sqlite]"      # or [duckdb] / [snowflake] / [databricks] / [s3] / [all]
```

Consumers depend on it and surface these extras under their own name, e.g.
`sdm-client[snowflake]` and `sdm_relational_engine[databricks]` both resolve the corresponding
`sdm-connectors` extra.

## Local development

```bash
pip install -e ".[sqlite,duckdb,test]"
pytest tests
```
