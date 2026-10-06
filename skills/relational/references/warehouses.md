# Reading tables that stay in a warehouse

`Schema.from_data` wants DataFrames you already hold. `Schema.from_source`
describes tables where they are, and `materialize()` samples them with SQL, so
what leaves the warehouse is bounded rather than the table.

Bounded, not nothing. `Schema.from_source` reads up to a thousand rows of each
selected table there and then, to work out what its columns hold -- all of a
table that holds a thousand or fewer -- and `check_keys` adds a scan per
eligible table. After that only the rows a prediction needs leave.

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
`pip install 'structured-data-models-client[relational,databricks]'`. With the connector alone the
first call raises `MissingExtraError`.

`Source(connection)` picks the right one for you when you would rather not
name it. Lakebase is `PostgresSource`, with the `lakebase` extra and
`sdm_connectors.databricks.lakebase(endpoint, database=..., workspace=...)`,
which mints the credential and gives back a psycopg connection. Both keywords are
keyword-only; `database` is required, and `workspace` falls back to an ambient
`WorkspaceClient()` when it is left out.

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
mistyped path gives an empty database, and describing it is refused for being
empty rather than for the path being wrong:

```
DuckDBSource() holds no table to describe
```

Pass `read_only=True` and the connection refuses the missing path instead,
which says what actually went wrong.

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
from sdm_connectors import DatabricksSource

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

from sdm_connectors import DatabricksSource
from sdm_connectors.databricks import warehouse

databricks_connection = warehouse(warehouse_id, workspace=WorkspaceClient())
source = DatabricksSource(
    databricks_connection, catalog='main', schema='sales'
)
```

You close what it hands back.

In a Snowflake notebook, a Snowpark `session` passes as it is:

```python
from sdm_connectors import SnowflakeSource

source = SnowflakeSource(session, database='SALES', schema='PUBLIC')
```

## A metric view, or a semantic view

A metric view already names the tables a question is about and how they join, so it can be the source itself:

```python
from sdm_client import Schema
from sdm_connectors import DatabricksSource

source = DatabricksSource(
    databricks_connection, catalog='sales', schema='public'
)
schema = Schema.from_source(source.semantic_model('sales_metrics'))
```

Snowflake's semantic views are reached the same way, through the same method.
Both are what the Open Semantic Interchange and dbt call a semantic model, but they are not shaped alike, so what a schema takes differs.

From a **Databricks metric view**:

- its source table, named as it is in the catalog;
- each join, nested ones too, as a table named after the join and keyed by the column it joins on;
- each dimension, as a column of the table it reads, and a wildcard such as `source.*` or `customers.* EXCEPT (rowguid)` as the columns it stands for;
- the columns a measure aggregates, when it reads one table, added to that table where no dimension names them, so a query can aggregate them too;
- its `filter`, limiting every read of the source table;
- each parameter the filter or a dimension references, read as its default, as a query of the view without arguments reads it.

The source table's time column is a date the view holds, chosen as for any table, with two differences.
A dimension that truncates or casts a date, such as `DATE_TRUNC('MONTH', OrderDate)`, is offered as the column it truncates, added when no dimension names it, so a 30-day window counts days rather than months.
The date a window measure orders by is offered first, as the time axis the view's author chose.

From a **Snowflake semantic view**: its logical tables, the relationships between them, and their dimensions, time dimensions and facts.
A fact becomes a column.

**Measures and metrics** are not columns, since they are aggregations rather than row values.
Ask for them in the query instead.
Converting a metric view notes how it read the view in `schema.conversion_notes`: the aggregation matching each simple measure, such as `'revenue' is SUM(order_lines.LineTotal, 0, 30, days)`, the time column taken in place of a date the view truncates, and each parameter read as its default.
What it left out is said in one `SDMWarning`, and a view that loses nothing converts without one.

Left out, each with a message in that warning:

- a join on more than one column or on an expression, which one graph link cannot express;
- a join whose table cannot be read, being missing, empty, unreadable or itself a metric view;
- a join whose key repeats a value, since a graph table holds one row per key;
- with each of those joins, the tables and dimensions that hang off it, all named;
- a dimension calling functions beyond the pure built-ins, which would run as SQL under your credentials every time a request is gathered; pass `trust_expressions=True` to keep it, and read it first;
- a dimension referencing a parameter with no default, `materialization`, and `fields` beside `dimensions`, which Databricks ignores as well;
- a column whose name holds a dot, a backtick or a line break, which no predictive query can name.

Refused, with a `StructuredDataError` (`INVALID_REQUEST`) whose message names the way out:

- a filter on a joined table, or on a parameter with no default: create a regular view with that `WHERE` clause and point the metric view's source at it;
- a filter calling functions beyond the pure built-ins, unless `trust_expressions=True`;
- a SQL query as the source: create a regular view over the query instead;
- another metric view as the source: point the source at its underlying table.

A metric view is skipped by plain auto-discovery, by design. Naming one as an
ordinary table is refused, and the refusal says what to do:

```
'sales_metrics' is a metric view, not a table: its measures are aggregations
with no value per row. Read it as a semantic model instead, with
source.semantic_model('sales_metrics').
```

## A file, rather than a connection

`read()` takes a path and gives a DataFrame. CSV and Parquet only, by suffix:
`.csv` or `.txt`, compressed or not, and `.parquet`, `.parq` or `.pq`
uncompressed, or a Parquet directory, local or on object storage with the `s3`
extra. Anything else is refused, `format='json'` included:

```python
from sdm_connectors import read

orders = read('s3://bucket/orders.parquet', storage_options=credentials)
```

`storage_options` is a dictionary handed to the filesystem underneath, which
is where a key and a secret go.

A compression suffix on a Parquet name is taken as Parquet and read as one, so
`orders.parquet.gz` fails on the file rather than on the name. Compress the CSV
if it has to be compressed, or leave the Parquet as it is -- it is compressed
inside already.

That is a DataFrame like any other, so it goes to `Schema.from_data`.

`read_table(connection, table=...)` or `read_table(connection, query=...)`
does the same from a connection, for one flat table, without building a
source.

## When it refuses

`ConnectorError`, from `sdm_connectors`, is its own exception and **not** a
`StructuredDataError` — catching the one does not catch the other, so a
warehouse path needs both. `MissingBackendError` is a `ConnectorError` and
names the `pip install` that fixes it.
