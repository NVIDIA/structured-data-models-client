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

Install the connector **and** the engine, since `Schema.from_source` needs it:
`pip install 'sdm-client[relational,databricks]'`. With the connector alone the
first call raises `MissingExtraError`.

`Source(connection)` picks the right one for you when you would rather not
name it. Lakebase is `PostgresSource`, with the `lakebase` extra and
`sdm_connectors.databricks.lakebase(endpoint, workspace=...)`, which mints the
credential and gives back a psycopg connection.

**SQLite is the one that catches people.** It wants an ADBC connection, not the
standard library's `sqlite3`:

```
SQLiteSource needs an ADBC SQLite connection from
adbc_driver_sqlite.dbapi.connect(); got Connection
```

## The whole interface

```python
import duckdb
import pandas as pd
from sdm_client import KumoRelational, PredictiveQuery, Relationship, SDMClient, Schema
from sdm_connectors import DuckDBSource

connection = duckdb.connect()
connection.register('users', pd.DataFrame({'user_id': range(12)}))
connection.register(
    'orders',
    pd.DataFrame(
        {
            'order_id': range(120),
            'user_id': [i % 12 for i in range(120)],
            'ordered_at': pd.date_range('2026-01-01', periods=120, freq='D'),
        }
    ),
)
connection.execute('CREATE TABLE u AS SELECT * FROM users')
connection.execute('CREATE TABLE o AS SELECT * FROM orders')

source = DuckDBSource(connection)

schema = Schema.from_source(
    source,
    {'users': 'u', 'orders': 'o'},
    primary_keys={'users': 'user_id', 'orders': 'order_id'},
    time_columns={'orders': 'ordered_at'},
    relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
)

request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
).materialize(indices=[1, 2], model=KumoRelational())

with SDMClient(url='http://localhost:8000', timeout=600) as client:
    print(client.predict(request, model=KumoRelational()).to_pandas())
```

**Verified** against a live endpoint:

```
   row_index  id  prediction
0          0   1    2.415527
1          1   2    2.410645
```

From there it is the same as any other request: the answer comes back keyed by
`row_index` and the entity, as [`../SKILL.md`](../SKILL.md) describes.

`duckdb.connect()` with no argument gives an in-memory database, which is the
quickest way to try this.

Given no `tables`, it discovers what the namespace holds. Given a mapping, it
takes those and renames them: `Schema.from_source(source, {'people': 'users'})`
gives a schema of one table called `people`.

`primary_keys`, `time_columns` and `relationships` work as they do for
`from_data`. Left out, a source infers more than a frame can: it reads the
catalog's own primary and foreign keys first, and falls back to column names
only where there are no constraints. `check_keys=False` skips the uniqueness
check, which is one statement that scans each eligible table.

**`duckdb.connect('shop.db')` creates the file if it is not there**, so a
mistyped path gives an empty database and a schema that holds no table. Pass
`read_only=True` and it refuses the missing path instead.

## A warning from the engine to ignore

Sampling a DuckDB or SQLite source prints a suggestion from the engine:

```
we strongly suggest to create indices ... e.g., automatically by instantiating
Kumo Relational via KumoRelational(graph, optimize=True).
```

**That API is not this one.** `KumoRelational()` here takes no arguments and
there is no graph object to pass it. The message is the engine talking about
its own surface; nothing in this client acts on it.

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

Snowflake's semantic views are reached the same way, through the same method.
Both are what the Open Semantic Interchange and dbt call a semantic model, but
they are not shaped alike, so what a schema takes differs.

From a **Databricks metric view**:

- its source table, named as it is in the catalog
- each join, as a table named after the join
- each dimension, as a column of the table it reads
- its `filter`, limiting the rows of the source table

From a **Snowflake semantic view**: its logical tables, the relationships
between them, and their dimensions, time dimensions and facts. A fact becomes
a column.

What neither contributes:

- **measures and metrics**, because they are aggregations rather than row
  values. Ask for them in the query instead: a measure that sums an amount is
  `SUM(orders.amount, 0, 30, days)`.
- **a filter on a joined table**, which is refused rather than ignored.
- **dimensions whose expressions call functions beyond the pure built-ins**,
  left out with a message. They would run as SQL under your credentials every
  time a request is gathered. Pass `trust_expressions=True` to keep them, and
  read them first.

A metric view is skipped by plain auto-discovery, by design. Naming one as an
ordinary table is refused, and the refusal says what to do:

```
'sales_metrics' is a metric view, not a table: its measures are aggregations
with no value per row. Read it as a semantic model instead, with
source.semantic_model('sales_metrics').
```

## A file, rather than a connection

`read()` takes a path and gives a DataFrame. CSV and Parquet only, by suffix
(`.csv`, `.txt`, `.parquet`, `.parq`, `.pq`, optionally compressed) or a
Parquet directory, local or on object storage with the `s3` extra. Anything
else is refused, `format='json'` included:

```python
from sdm_connectors import read

orders = read('s3://bucket/orders.parquet', storage_options={...})
```

That is a DataFrame like any other, so it goes to `Schema.from_data`.

`read_table(connection, table=...)` or `read_table(connection, query=...)`
does the same from a connection, for one flat table, without building a
source.

## When it refuses

`ConnectorError`, from `sdm_connectors`, is its own exception and **not** a
`StructuredDataError` — catching the one does not catch the other, so a
warehouse path needs both. `MissingBackendError` is a `ConnectorError` and
names the `pip install` that fixes it.
