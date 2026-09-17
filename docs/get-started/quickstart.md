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

The rows being scored need not appear in the labelled half. That is the point:
this model exists to score rows it has never seen.

## Several related tables

Describe the tables, ask in PQL, and name the entities to score.

```python
from sdm_client import PredictiveQuery, SDMClient, KumoRelational

query = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
    {'users': users_df, 'orders': orders_df},
)
request = query.materialize(indices=[7, 12, 3])

with SDMClient(url='http://localhost:8001') as client:
    print(client.predict(request, model=KumoRelational()).to_pandas())
```

`materialize()` is where the work happens: it resolves the query, finds
labelled examples in history, samples each entity's neighbourhood and encodes
the result. It is a separate call because it is the expensive one.

**Pick a window your data can support.** An example's features come from
before its anchor and its label from the window after it, so a query over
`0, 30, days` needs a dataset that spans comfortably more than 30 days: one
shorter than the window leaves no anchor with a label after it, and nothing to
learn from. You will be told if it does not have it.

**The window opens after the anchor and closes on its last day.** `0, 30,
days` covers `(anchor, anchor + 30 days]`: a row at the anchor itself does not
count, one a day later does, one exactly thirty days later does, one thirty-one
days later does not. It is worth knowing if you are computing the same
aggregate yourself to check a model against a baseline, because taking either
edge the other way moves every number.

## Asking the same context several questions

Uploading a context once and scoring against it repeatedly is what `fit` is
for.

```python
with client.fit(request, model=KumoRelational()) as fitted:
    first = fitted.predict(request)
    second = fitted.predict(other_request)
```

## Which rows get scored

A query says *what* to predict. It never says *which rows*.

| You want to score | Schema | Use |
|---|---|---|
| existing rows | single table | `materialize(indices=[...])` |
| new rows supplied separately | single table | `Request(context=…, predict=…)` or `materialize(predict=…)` |
| existing entities | multiple related tables | `materialize(indices=[...])` |

There is no all-rows default, because it would silently score a whole table.

For multiple tables, `materialize()` additionally gathers records related to each
entity.

## Which model it is gathered for

The two models read a request differently, so it is gathered for one of them.
Left unsaid, the schema's shape decides: a schema of one table with no
relationships is gathered for `kumo-tabular`, and anything wider — a second
table, or a table that points at itself — for `kumo-relational`.

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

A ranking is asked once per candidate — *will this entity reach this one?* —
and the candidates are ordered by how sure each answer was. The set is yours to
name, because there is no ranking task to choose it for you:

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

Both are sent as several calls, so they cost several calls: a ranking over
fifty candidates is fifty. `request.context_rows` and `request.predict_rows`
count what will actually be sent. Neither can be pinned with `fit()`, because
each part needs the context its own labels come from and a session pins one.
