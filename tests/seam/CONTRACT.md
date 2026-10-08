# Phase 3 acceptance contract

The acceptance checks in this directory test behavior through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

## `Source._run`

- Every source class in `sdm_connectors` (`DatabricksSource`, `SnowflakeSource`, `PostgresSource`, `DuckDBSource`, `SQLiteSource` and `SemanticModel`) has an internal method `_run(sql, params)` (deviation DV13).
- No source class has a public attribute named `run`, so the public API snapshot doesn't change.
- `sql` is the statement's text, passed to the driver unchanged.
- `params` is `None` or a list of values, bound positionally with the database's own placeholders: `?` for SQLite, DuckDB, Databricks and Snowflake, `%s` for PostgreSQL.
- It returns a `pyarrow.Table`, with every column the statement returns, in order, under the name the database gives it.
- A PostgreSQL column's Arrow type follows its declared type, including when it holds a `NULL` or the result has no rows: `bigint` is `int64`, `double precision` is `float64`, `text` is `string`, `boolean` is `bool`, `timestamp` is a timestamp.
- The engine reaches a database only by calling `_run`: during a prediction, every driver call the engine causes is made beneath a `sdm_connectors` function named `_run`, the first `sdm_connectors` function the engine's own code calls.
  A backend's module-level `run` in `sdm_connectors/backends/<db>.py` may do the work, beneath `_run`.
- `DatabricksSource` takes a notebook's Spark session as it does today, recognised by its class name and module, and runs each statement with `session.sql(sql)` or `session.sql(sql, args=params)`.
- On Snowflake, a statement with parameters reaches the driver while the connection's parameter style is `qmark`, and the connection's own style is back as it was once `_run` returns.
- On PostgreSQL without autocommit, `_run` leaves no transaction open that wasn't open before it, never commits a transaction the caller opened, and a statement that fails inside the caller's transaction leaves that transaction usable.

## Dialect modules

- `sdm_relational_engine.rfm.backend.<db>.dialect` exists for `databricks`, `snow`, `postgres`, `duckdb` and `sqlite`.
- Its SQL-building functions are plain functions (module level, or on a class defined in the module) that return `(sql, params)`: a tuple, or a named tuple, whose first item is the SQL text.
  A function returning a list of such pairs is also accepted.
- They touch no connection: a dialect module calls no driver method and no `run` or `_run`, imports no driver, `sdm_connectors.source` or `sdm_connectors.backends`, and no function in it takes a `connection`, `conn`, `cursor` or `session`.
- The SQL text a dialect function returns is the text the driver receives, byte for byte.
  The checks prove it by appending a comment to every returned SQL text and requiring every statement the driver receives to end with it.
- The checks patch each such function in place, so it must not be a closure; a function the module imports from elsewhere is not one of its functions.

## Diagnostics catalog

- `packages/sdm-client/diagnostics.json` is a JSON object with a `diagnostics` array.
- Each entry is an object with:
  - `kind`: `raise` or `warning`;
  - `class`: the exception or warning class, by name or dotted path;
  - `code`: the `code=` the raise passes, as a string, or `null` where there is none or it isn't a literal;
  - `template`: the message, with each interpolated part written as a placeholder and every literal part as it is in the source;
  - `file`: the repository-relative path of the file, with `/` separators;
  - `line`: a line of the `raise` statement or the `warnings.warn` call.
- `scripts/diagnostics.py` generates it from the source files of the repository it is in, found from its own location:
  - `scripts/diagnostics.py --out PATH` writes the catalog to `PATH`;
  - `scripts/diagnostics.py --check [PATH]` compares `PATH` (default `packages/sdm-client/diagnostics.json`) with a fresh catalog, and exits 0 only if they are identical.

## Recordings

- The DuckDB and SQLite recordings under `tests/parity/transcripts/` are re-recorded with this phase's code; every other recording is unchanged.
- A key list is passed as one statement parameter holding a JSON array, as text.

## Warehouse SQL

- `golden/warehouse_sql.json` pins, for Databricks (SQL connection, Spark session, metric view) and Snowflake (direct, semantic view), the statements and parameters a prediction over the stub drivers in `_warehouse.py` sends, and the payloads and warnings it builds.
  It was written at `BASE` with `python tests/seam/_warehouse.py write-golden` and must not change.
- A statement without parameters may reach a stub driver with `None`, `[]` or `()`.
