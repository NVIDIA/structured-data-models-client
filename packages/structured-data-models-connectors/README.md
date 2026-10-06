# structured-data-models-connectors

Shared data-source connectors for the [`structured-data-models-client`](../structured-data-models-client/README.md) client and its
model engines. One place that knows how to reach each warehouse, so the client
(flat table reads) and the Kumo Relational driver (warehouse connections for
its graph samplers) don't each carry their own copy.

It provides:

- `DatabricksSource`, `SnowflakeSource`, `PostgresSource`, `DuckDBSource` and `SQLiteSource`: a connection and the namespace its tables live in, for `Schema.from_source` to sample in place.
  Each takes the driver's own connection, opened with the driver's own `connect`, or a Spark or Snowpark session.
  A source never closes it: the caller owns the connection and closes it.
- `sdm_connectors.databricks.warehouse(warehouse_id, workspace=w)` and `lakebase(endpoint, database=..., workspace=w)`: open a Databricks SQL warehouse or Lakebase connection as the caller's own `WorkspaceClient`.
  Each returns the driver's native connection, for a source to take and the caller to close.
  Databricks requires a partner product to identify every connection and API request it opens, so each names sdm-client as `nvidia_sdm-client/<version>`.
  A warehouse connection carries it in the driver's User-Agent, ahead of any `user_agent_entry` you pass, and tags its statements `sdm_client_purpose: sampling` beside your own query tags, unless you set `sdm_client_purpose` yourself.
  A Lakebase connection carries it as `application_name`, ahead of any you pass, within the 63 bytes Postgres keeps.
  The `WorkspaceClient` requests they make carry it too, through a copy of your client, which is left unchanged.
  A connection or Spark session you open yourself and pass to a source is used as it is, without the identifier or the tag.
- `read(path, *, format=None, storage_options=None) -> DataFrame`: read a CSV or Parquet file as a flat pandas DataFrame, from a local path, a `file:` URI or an `s3://` URI.
- `read_table(connection, table=…, query=…) -> DataFrame`: driver-agnostic fetch over an open connection, in that connection's own session.
- `quote_ident(ident, char='"')` and `resolve_sql(table=…, query=…)`: SQL identifier quoting and a safe table/query guard.

```python
import duckdb
from databricks.sdk import WorkspaceClient
from sdm_connectors import DatabricksSource, DuckDBSource, PostgresSource
from sdm_connectors.databricks import lakebase, warehouse

w = WorkspaceClient()
uc = DatabricksSource(
    warehouse(warehouse_id, workspace=w), catalog='main', schema='sales'
)
lb = PostgresSource(
    lakebase(endpoint, database='databricks_postgres', workspace=w),
    schema='public',
)
dd = DuckDBSource(duckdb.connect('sales.duckdb', read_only=True))
```

A file-backed database opened for writing is created empty when its file is missing, and one opened read-only is refused instead.
Open DuckDB with `read_only=True`, and SQLite through a read-only URI such as `adbc_driver_sqlite.dbapi.connect('file:sales.sqlite?mode=ro')`.

File reads accept `.csv` / `.txt` (optionally compressed) and `.parquet` / `.pq` / `.parq`, plus a directory as a Parquet dataset.
Any other suffix is rejected with `INVALID_CONNECTOR_ARGS` rather than parsed as CSV.
Pass `format='csv'` or `format='parquet'` to read a file whose name carries no recognised suffix.
`storage_options` applies to an `s3://` path only, and goes to `s3fs` as it is.

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
connection = duckdb.connect('w.db', read_only=True)
read_table(connection, query=f'SELECT * FROM {quote_ident("café")}')
```

Failures raised by the sources, `read`, `read_table` and the Databricks helpers surface as `ConnectorError` with a stable `code`: `UNKNOWN_CONNECTOR`, `INVALID_CONNECTOR_ARGS`, `CONNECT_FAILED`, `QUERY_FAILED`, `READ_FAILED`, or `NOT_FOUND`.
Where a driver raised, its exception is chained as `__cause__`; an unknown connector, and an argument refused before any driver is reached, have none.
A missing driver, or a databricks-sdk too old for `lakebase()`, is refused with code `MISSING_BACKEND`; a missing driver raises it as `MissingBackendError`, which is also a `ModuleNotFoundError`.
A broken or incompatible driver installation stays unwrapped by design and raises `ImportError` (consumers map it to their own broken-install category, e.g. structured-data-models-client's `DRIVER_LOAD_FAILED`).

Pure-python. Database drivers are optional extras:

```bash
pip install "structured-data-models-connectors[sqlite]"      # or [duckdb] / [postgres] / [snowflake] / [databricks] / [lakebase] / [s3] / [all]
```

Consumers depend on it and surface these extras under their own name, e.g.
`structured-data-models-client[snowflake]` and `sdm_relational_engine[databricks]` both resolve the corresponding
`structured-data-models-connectors` extra.

## Local development

```bash
pip install -e ".[sqlite,duckdb,postgres,databricks,test]"
pytest tests
```

The exactness tests check that a value read here survives all the way onto the
wire, so they need `structured-data-models-client` as well. They skip without it; install it too
to run them:

```bash
pip install -e ../structured-data-models-api -e ../structured-data-models-pql -e ../structured-data-models-client
```
