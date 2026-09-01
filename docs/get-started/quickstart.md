# Quickstart

## One table

You have labelled rows and rows you want scored. That is the whole input.

```python
import pandas as pd
from sdm_client import Request, StructuredDataClient, KumoTabular

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

with StructuredDataClient(url='http://localhost:8000') as client:
    print(client.predict(request, model=KumoTabular()).to_pandas())
```

The rows being scored need not appear in the labelled half. That is the point:
this model exists to score rows it has never seen.

## Several related tables

Describe the tables, ask in PQL, and name the entities to score.

```python
from sdm_client import PredictiveQuery, StructuredDataClient, KumoRelational

query = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
    {'users': users_df, 'orders': orders_df},
)
request = query.materialize(indices=[7, 12, 3])

with StructuredDataClient(url='http://localhost:8001') as client:
    print(client.predict(request, model=KumoRelational()).to_pandas())
```

`materialize()` is where the work happens: it resolves the query, finds
labelled examples in history, samples each entity's neighbourhood and encodes
the result. It is a separate call because it is the expensive one.

**Pick a window your data can support.** A query over `0, 30, days` needs 30
days of history before the anchor, and you will be told if it does not have it.

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

| You want to score | Use |
|---|---|
| entities the tables already contain | `materialize(indices=[...])` |
| rows the tables do not contain | `Request(context=…, predict=…)`, or `materialize(predict=…)` |

There is no all-rows default, because it would silently score a whole table.
