# Reading tables that stay in a warehouse

`Schema.from_data` wants DataFrames you already hold. `Schema.from_source`
describes tables where they are, and `materialize()` samples them with SQL, so
only the rows a prediction needs ever leave the warehouse.

Everything else is the same afterwards: the schema, the query, the request.

## A source

A source is a connection plus the namespace its tables live in. Each takes the
connection its own driver makes, and says so if given another.

| Source | Connection it takes | Namespace | Extra |
|---|---|---|---|
| `SnowflakeSource` | `snowflake.connector.connect()`, or a Snowpark session | `database`, `schema` | `snowflake` |
| `DatabricksSource` | `databricks.sql.connect()`, or a notebook's Spark session | `catalog`, `schema` | `databricks` |
| `PostgresSource` | `psycopg.connect()`, synchronous | `schema` | `postgres` |
| `DuckDBSource` | `duckdb.connect()` | — | `duckdb` |
| `SQLiteSource` | `adbc_driver_sqlite.dbapi.connect()` | — | `sqlite` |

Install the one you need: `pip install 'sdm-client[databricks]'`. Lakebase is
`PostgresSource` with the `lakebase` extra.

**SQLite is the one that catches people.** It wants an ADBC connection, not the
standard library's `sqlite3`:

```
SQLiteSource needs an ADBC SQLite connection from
adbc_driver_sqlite.dbapi.connect(); got Connection
```

## The whole interface

```python
import duckdb
from sdm_client import KumoRelational, PredictiveQuery, Relationship, Schema
from sdm_connectors import DuckDBSource

source = DuckDBSource(duckdb.connect('shop.db'))

schema = Schema.from_source(
    source,
    primary_keys={'users': 'user_id', 'orders': 'order_id'},
    time_columns={'orders': 'ordered_at'},
    relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
)

request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
).materialize(indices=[1, 2], model=KumoRelational())
```

Given no `tables`, it discovers what the namespace holds. Given a mapping, it
takes those and renames them: `Schema.from_source(source, {'people': 'users'})`
gives a schema of one table called `people`.

`primary_keys`, `time_columns` and `relationships` work as they do for
`from_data`, and are inferred the same way when left out. `check_keys=False`
skips the uniqueness check, which costs a query per table.

## Inside the warehouse's own notebook

The session the notebook already holds is a connection, so there is nothing to
open and no credential to handle.

In a Databricks notebook, pass `spark`:

```python
source = DatabricksSource(spark, catalog='main', schema='sales')
```

It samples through `spark.sql`, so **no SQL warehouse is needed**. That path
uses `DataFrame.toArrow()`, which Databricks Runtime 17.3 LTS and serverless
environment 4 have; on an older runtime pass a SQL warehouse connection
instead.

A connection or session you pass is used as it is, so it carries no
sdm-client partner identifier or query tag. One opened through the helper
carries both:

```python
from databricks.sdk import WorkspaceClient
from sdm_connectors.databricks import warehouse

connection = warehouse(warehouse_id, workspace=WorkspaceClient())
source = DatabricksSource(connection, catalog='main', schema='sales')
```

You close what it hands back.

In a Snowflake notebook, a Snowpark `session` passes as it is:

```python
source = SnowflakeSource(session, database='SALES', schema='PUBLIC')
```

## A metric view, or a semantic view

A metric view already names the tables a question is about and how they join,
so it can be the source itself:

```python
source = DatabricksSource(connection, catalog='sales', schema='public')
schema = Schema.from_source(source.semantic_model('sales_metrics'))
```

Snowflake's semantic views work the same way, through the same method. Both
are what the Open Semantic Interchange and dbt call a semantic model.

What a schema takes from one:

- its source table, named as it is in the catalog
- each join, as a table named after the join
- each dimension, as a column of the table it reads
- its `filter`, limiting the rows of the source table

What it leaves out:

- **measures and metrics**, because they are aggregations rather than row
  values. Ask for them in the query instead: a measure that sums an amount is
  `SUM(orders.amount, 0, 30, days)`.
- **a filter on a joined table**, which is refused rather than ignored.
- **dimensions whose expressions call functions beyond the pure built-ins**,
  left out with a message. They would run as SQL under your credentials every
  time a request is gathered. Pass `trust_expressions=True` to keep them, and
  read them first.

A metric view is also skipped by plain auto-discovery, by design. Name it
through `semantic_model()`; listing it as an ordinary table surfaces a raw
server error.

## A file, rather than a connection

`read()` takes a path and gives a DataFrame, for Parquet, CSV and the rest,
local or on object storage with the `s3` extra:

```python
from sdm_connectors import read

orders = read('s3://bucket/orders.parquet', storage_options={...})
```

That is a DataFrame like any other, so it goes to `Schema.from_data`.

## When it refuses

`ConnectorError`, from `sdm_connectors`, is its own exception and **not** a
`StructuredDataError` — catching the one does not catch the other, so a
warehouse path needs both. `MissingBackendError` is a `ConnectorError` and
names the `pip install` that fixes it.
