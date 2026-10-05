---
name: sdm-relational
description: "Ask kumo-relational a question about an entity and the records linked to it: write the question as a predictive query over a schema of related tables, covering classification, regression, forecasting and link prediction. Triggers: kumo-relational, KumoRelational, PREDICT, FOR EACH, predictive query, PQL, Schema.from_data, Schema.from_source, Relationship, ranking, forecast, link prediction, explanation."
---

# Asking kumo-relational

An entity, and the records linked to it. The question is written as a
predictive query, the client gathers each entity's neighbourhood, and the model
reads that neighbourhood in one pass.

Load [`../tabular/SKILL.md`](../tabular/SKILL.md) instead when the question is
about rows of one table with no links.

Writing the query itself is in [`pql.md`](pql.md).

## What it can be asked

Two different questions, and they have two different answers:

```python
from sdm_client import SDMClient, KumoRelational

with SDMClient(url='http://localhost:8000') as client:
    client.capabilities(KumoRelational())  # what this client will send
    client.models()                        # what the endpoint advertises
```

`capabilities()` is the client's own table, tagged `source='sdk'`, and needs no
connection. It is the wider of the two: a deployment may refuse something the
client is willing to ask. What the client will send:

| | `kumo-relational` |
|---|---|
| Tasks | `binary_classification`, `classification`, `multiclass_classification`, `regression`, `forecasting`, `temporal_link_prediction` |
| Outputs | `prediction`, `probabilities`, `quantiles`, `rankings`, `scores`, `embeddings`, `explanation` |
| Several tables | yes |
| Relationships | yes |
| Sessions | yes |

**The outputs narrow again at the endpoint, by task.** The shipping deployment
serves `prediction` and `quantiles` on a regression, `prediction` and
`probabilities` on either classification, and nothing else — including no
`explanation`, on any task:

```
422 UNSUPPORTED_OUTPUT output(s) ['explanation'] not supported by model
'kumo-relational' for task 'regression'; supported: ['prediction', 'quantiles']
```

So ask for an explanation, catch the refusal, and fall back. Do not build a
flow that assumes one arrives.

The task is not an argument. It follows from the query: see
[`pql.md`](pql.md).

## The whole interface

`KumoRelational` needs the engine, which is an extra:

```
pip install 'sdm-client[relational]'
```

Without it the first call raises `MissingExtraError` naming that command.

```python
import pandas as pd
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    SDMClient,
    Schema,
)

users = pd.DataFrame({'user_id': range(12), 'segment': ['a', 'b'] * 6})
orders = pd.DataFrame(
    {
        'order_id': range(120),
        'user_id': [i % 12 for i in range(120)],
        'item': [i % 4 for i in range(120)],
        'price': [10.0 + i for i in range(120)],
        'ordered_at': pd.date_range('2026-01-01', periods=120, freq='D'),
    }
)
items = pd.DataFrame({'item': range(4), 'kind': list('abcd')})

schema = Schema.from_data(
    {'users': users, 'orders': orders, 'items': items},
    primary_keys={'users': 'user_id', 'orders': 'order_id', 'items': 'item'},
    time_columns={'orders': 'ordered_at'},
    relationships=[
        Relationship('orders', 'user_id', 'users', 'user_id'),
        Relationship('orders', 'item', 'items', 'item'),
    ],
)

request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
).materialize(indices=[1, 2, 3], model=KumoRelational())

with SDMClient(url='http://localhost:8000', timeout=600) as client:
    answer = client.predict(request, model=KumoRelational())
print(answer.to_pandas())
```

**Verified** against a live endpoint:

```
   row_index  id  prediction
0          0   1    2.173828
1          1   2    2.181641
2          2   3    2.212891
```

`id` is the entity, `row_index` its position in `indices`. Join on the key when
it is there, otherwise on `row_index`, and never on row order.

## Describing the data

A `Schema` says which tables exist, how they are keyed, which column carries
each table's time, and how they link.

```
Schema.from_data(frames, primary_keys=, time_columns=, relationships=)
```

`Relationship(from_table, from_column, to_table, to_column)` points from the
many side to the one side: an order names a user, so the relationship is
`Relationship('orders', 'user_id', 'users', 'user_id')`.

Leave `primary_keys` or `time_columns` out and they are inferred, with a
warning naming anything doubtful and the override that settles it. The
inference deliberately refuses some columns, among them a stamp recording when
a row was last *edited* on a table other rows point at, since that moves after
the fact and would leak the future into the past. Pass
`time_columns={'t': None}` to say a table has no time column at all.

For tables that stay in a warehouse, `Schema.from_source` describes them in
place and `materialize()` samples them with SQL, so only the rows a prediction
needs ever leave.

## Which rows get scored

`materialize(indices=[...])` names the entities. It gathers each one's
neighbourhood, checks every limit before sending, and refuses with the limit it
would break rather than letting the endpoint return a 500.

| Limit | Value |
|---|---|
| context rows, by default | `1,000` |
| context rows, with `num_context_samples` | `10,000` |
| rows scored at once | `1,000` |
| rows scored for a link prediction | `200` |
| rows scored with an explanation | `1` |
| tables in the gathered graph | `15` |
| hops gathered | `6` |
| classes | `10` |

**The table limit is on the graph a request carries, not on the schema.** A
schema may describe far more than fifteen tables; what counts is how many the
query actually reaches. An explanation has a second limit besides its one row:
it must fit the 1,000-row context budget.

## Asking for more than one answer each

A ranking, a forecast and an explanation are each asked as several calls, and
each has limits and refusals of its own. They are in
[`references/beyond-one-answer.md`](references/beyond-one-answer.md).

## Reading the answer

`result.to_pandas()` gives one row per entity, except a forecast, which gives
one per entity per step, and a ranking, which gives a list in one row.

A regression's `prediction` is the **median** of the predicted distribution,
not the mean. Ask for the mean with
`materialize(..., prediction_statistic='mean')` when predictions will be summed
across entities, since medians do not add up.

## When it refuses

Everything is a `StructuredDataError` or a subclass, and each names what to
change. The ones worth knowing:

| Refusal | Cause |
|---|---|
| a candidate's context holds one class | every gathered entity reached it, or none did, so there is nothing to tell apart; widen with `num_context_samples` or ask over a table of pairs |
| no entity existed at the anchor | the entity table's time column starts after the anchor, so nothing existed to ask about |
| more entities than the model scores | split them across calls |
| an output this question cannot give | `rankings` on a regression, `explanation` on a ranking or on more than one row |
| `422 UNSUPPORTED_OUTPUT` | the client sent an output the deployment does not serve, such as `explanation` |
| an aggregation the model does not answer | `COUNT_DISTINCT`, `FIRST`, `LAST`, or any aggregation of a category |
