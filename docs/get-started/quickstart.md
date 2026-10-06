# Quickstart

## One table

You have labelled rows and rows you want scored. That is the whole input.

```python
import pandas as pd
from sdm_client import Request, SDMClient, KumoTabular

labelled = pd.DataFrame(
    {
        'age': [31.0, 44.5, 28.0, 51.0, 39.0],
        'spend': [120.0, 430.0, 90.0, 510.0, 260.0],
        'churn': [0, 1, 0, 1, 0],
    }
)
to_score = pd.DataFrame({'age': [33.0, 47.0], 'spend': [150.0, 480.0]})

request = Request(
    context=labelled, predict=to_score, target='churn', task='classification'
)

with SDMClient(url='http://localhost:8000') as client:
    print(client.predict(request, model=KumoTabular()).to_pandas())
```

The model can score rows absent from the labelled half.

## Several related tables

Describe the tables, ask in PQL, and name the entities to score.

```python
import pandas as pd
from sdm_client import PredictiveQuery, SDMClient, KumoRelational

users = pd.DataFrame({'user_id': range(20)})
orders = pd.DataFrame(
    {
        'order_id': range(200),
        'user_id': [i % 20 for i in range(200)],
        'ordered_at': pd.date_range('2026-01-01', periods=200, freq='D'),
    }
)

query = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
    {'users': users, 'orders': orders},
)
request = query.materialize(indices=[7, 12, 3])

with SDMClient(url='http://localhost:8001') as client:
    print(client.predict(request, model=KumoRelational()).to_pandas())
```

`materialize()` resolves the query, finds
labelled examples in history, samples each entity's neighbourhood and encodes
the result. This expensive work uses a separate call.

**Pick a window your data can support.** An example's features come from
before its anchor and its label from the window after it, so a query over
`0, 30, days` needs a dataset spanning comfortably more than 30 days. A
dataset shorter than the window leaves no anchor with a later label and
nothing to learn from. The client reports this condition.

**The window opens after the anchor and closes on its last day.** `0, 30,
days` covers `(anchor, anchor + 30 days]`: a row at the anchor itself does not
count, one a day later does, one exactly thirty days later does, one thirty-one
days later does not. If you compute the same aggregate to check a baseline,
reversing either edge changes every number.

## Tables that stay in the warehouse

Tables too large to read into memory stay where they are.
Describe them with `Schema.from_source`, and `materialize()` samples them there with SQL: only the rows a prediction needs leave the warehouse.

```python
from databricks.sdk import WorkspaceClient
from sdm_client import PredictiveQuery, Schema
from sdm_connectors import DatabricksSource
from sdm_connectors.databricks import warehouse

# In a Databricks notebook this is the notebook's own identity. Elsewhere it is
# whatever the Databricks SDK finds: a profile in ~/.databrickscfg, chosen with
# WorkspaceClient(profile=...), or DATABRICKS_HOST and a credential in the
# environment.
workspace = WorkspaceClient()
connection = warehouse(warehouse_id, workspace=workspace)
source = DatabricksSource(connection, catalog='main', schema='sales')
schema = Schema.from_source(
    source,
    {'users': 'crm_users', 'orders': 'shop_orders'},
    time_columns={'orders': 'ordered_at'},
)
request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
).materialize(indices=[7, 12, 3])
```

`warehouse()` looks the SQL warehouse up by its id and returns the `databricks.sql` driver's own connection, authenticated as `workspace`.
It reads no credential from the environment itself, so how `workspace` authenticates is the one place to look.
The connection is yours: the source never closes it, so close it once you are done sampling.
`warehouse()` comes with the `databricks` extra.
In a Databricks notebook, `DatabricksSource(spark, catalog=..., schema=...)` samples through the notebook's own session instead of a SQL warehouse (Databricks Runtime 17.3 LTS or serverless environment 4 and later).
`SnowflakeSource`, `PostgresSource`, `DuckDBSource` and `SQLiteSource` work the same way, each taking its driver's own connection and naming its own namespace.
For Lakebase, `lakebase(endpoint, database='databricks_postgres', workspace=workspace)`, from the same module, opens a `psycopg` connection for `PostgresSource` as `workspace` too; it comes with the `lakebase` extra.
Leave `tables` out to take every table the source holds.
Each is described from the catalog, which on a warehouse costs seconds per table, so on a large schema name the ones you need.
Each key is then checked against every row, in one statement that scans each table once, because Databricks and Snowflake do not enforce a declared key.
A key that repeats a value in more than 1% of rows is refused, as `Schema.from_data` refuses one; `check_keys=False` skips the scan and the check.
A schema's tables all come from one source.
A semantic model, such as a Databricks metric view or a Snowflake semantic view, is a source of its own, and brings its tables, keys and relationships: `Schema.from_source(source.semantic_model('sales_metrics'))`.
Its expressions run as SQL under your credentials, so a column calling anything but a standard built-in SQL function is left out with a message; `semantic_model(name, trust_expressions=True)` keeps it when you trust whoever wrote the model.
A metric view's `filter` limits every read of its source table; one on a joined table is refused, since only a regular view with that `WHERE` clause can hold it.

## Asking the same context several questions

Use `fit` to upload a context once and score against it repeatedly.

```python
with client.fit(
    request, model=KumoRelational(), holdout=every_key_you_will_score
) as fitted:
    first = fitted.predict(request)
    second = fitted.predict(other_request)
```

**Name every row the session will score in `holdout`.** The context is
chosen once, when you call `fit`, and it excludes only the rows `request`
itself names. Any row a later call scores that was not held back is sitting
in that context with its answer attached, and the answer comes back
flattered. Unlike the one-shot route, nothing can be done about it
afterwards, because the context cannot be changed once pinned.

## A model served on Databricks

A model deployed to Databricks Model Serving is reached by the endpoint's name.
In a notebook the call authenticates as the notebook, so there is no URL or key
to supply.

```python
from sdm_client import KumoRelational, SDMClient

with SDMClient.for_databricks_serving('kumo-relational') as client:
    result = client.predict(request, model=KumoRelational())
```

This needs `pip install 'structured-data-models-client[databricks-serving]'`. A serving endpoint
answers one request at a time, so `fit()` is refused there; `predict()` sends
the context with every call.

## Which rows get scored

A query says *what* to predict. It never says *which rows*.

| You want to score | Schema | Use |
|---|---|---|
| existing rows | single table | `materialize(indices=[...])` |
| new rows supplied separately | single table | `Request(context=…, predict=…)` or `materialize(predict=…)` |
| existing entities | multiple related tables | `materialize(indices=[...])` |

There is no all-rows default, because it would silently score a whole table.

For multiple tables, `materialize()` also gathers records related to each
entity.

## Which model it is gathered for

The two models read a request differently, so it is gathered for one of them.
By default, the schema's shape decides. One table with no relationships uses
`kumo-tabular`. A second table or a self-referencing table uses
`kumo-relational`.

Name a model when that default is not what you want. A single table is a graph
of one node, and the relational model answers it:

```python
request = query.materialize(indices=[7, 12, 3], model='kumo-relational')
client.predict(request, model='kumo-relational')
```

Gather and send for the same model. A request built for one is refused by the
other, before anything is sent.

## Ranking and forecasting

The endpoint serves classification and regression. A ranking and a multi-step
forecast are neither, so the client asks each as the questions it does serve
and puts the answers back together.

A ranking asks once per candidate: *will this entity reach this one?* It orders
the candidates by answer confidence. You must name the set because no ranking
task chooses it:

```python
request = PredictiveQuery(
    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 3 '
    'FOR EACH users.user_id',
    schema,
).materialize(indices=[1, 2, 3], candidates=items['item_id'])

result = client.predict(request, model=KumoRelational())
result.to_pandas()   # row_index, id, prediction, rankings
```

`prediction` holds the top candidates in order, and `rankings` pairs each with
its score. Ask for one of them by name to get only that one. The ranked column
has to be a relationship into the table whose rows you want ordered; ranking an
ordinary column is refused, because which labels a column takes is a different
question.

A candidate that no entity in its sampled context reached in the window cannot
be scored, so it is left out of every row and named on
`result.unscored_candidates`, and a row can hold fewer than `k`.
[Prediction output](../reference/prediction-output.md) says why, and how a
table of (entity, candidate) pairs scores every candidate.

A forecast is asked once per step, each over the window moved along by its own
span. Every entity comes back once per step, and `forecast_step` numbers them
from one:

```python
request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FORECAST 4 TIMEFRAMES '
    'FOR EACH users.user_id',
    schema,
).materialize(indices=[1, 2, 3])
```

Both use several calls. A ranking over fifty candidates uses fifty calls.
`request.context_rows` and `request.predict_rows` count the rows actually
sent. Neither can be pinned with `fit()` because each part needs the context
that supplies its own labels, while a session pins one context.
