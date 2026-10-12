---
name: sdm-tabular
description: "Ask kumo-tabular a question about rows of one table: classification or regression over labelled rows and rows to score, with no relationships and no time windows. Triggers: kumo-tabular, KumoTabular, single table prediction, tabular classification, tabular regression, score these rows, Request(context=, predict=), PredictiveQuery on one table."
---

# Asking kumo-tabular

One table. You hold rows whose answer is known and rows whose answer you want.
Both go in one request, the model reads them together, and the answer comes
back a row at a time.

Load [`../relational/SKILL.md`](../relational/SKILL.md) instead when the
question is about an entity and the records linked to it. The two models share
this client's objects -- `Schema`, `PredictiveQuery`, `Request` -- and the
language. What differs is almost everything that can be asked in it.

## What it can be asked

Read it, do not recall it:

```python
from sdm_client import SDMClient, KumoTabular

with SDMClient(url='http://localhost:8000') as client:
    client.capabilities(KumoTabular())  # what this client will send
    client.models()                     # what the endpoint advertises
```

| | `kumo-tabular` |
|---|---|
| Tasks | `classification`, `regression` |
| Outputs | `prediction`, `probabilities`, `quantiles`, `explanation` |
| Several tables | no |
| Relationships | no |
| Explanations | yes |
| Sessions | yes |
| Context rows | 10,000 |

Anything outside that list belongs to the relational model. Asking for
`rankings` or a forecast here is refused before a request is sent. An
`explanation` covers one scored row at a time, and asking for one over more
rows is refused before sending.

Outputs narrow again by task, and that narrowing happens at the endpoint rather
than in the client: `probabilities` on a classification, `quantiles` on a
regression, and asking for the wrong one comes back as
`422 UNSUPPORTED_OUTPUT`. `quantiles` gives a column per decile,
`quantile_0.1` through `quantile_0.9`, beside the point prediction.

## The whole interface

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
    answer = client.predict(
        request, model=KumoTabular(), outputs=['prediction', 'probabilities']
    )
print(answer.to_pandas())
```

**Verified** against a live endpoint:

```
   row_index  prediction  probability_0  probability_1
0          0           0       0.850108       0.149892
1          1           1       0.049825       0.950175
```

`row_index` is the position in the frame you passed as `predict`. Join on it.

A probability column is named for the class it belongs to, not numbered by
position, so a target of `'a'`, `'b'` and `'c'` gives `probability_a`,
`probability_b` and `probability_c`. Read the class out of the column name
rather than assuming an order.

## Choosing the task

`task=` is yours to state, and it is worth stating. A target holding two whole
numbers is read as a classification, which is usually right and sometimes not:
a count of 0 and 1 visits is a quantity, not a label. The client warns when it
guesses that way and names the argument that settles it.

- `task='classification'` for a label, however it is spelled: `0`/`1`,
  `'churned'`/`'stayed'`, more than two classes.
- `task='regression'` for a quantity.

## Which rows get scored

`Request(context=..., predict=...)` takes two frames and sends exactly those
rows. It trims neither. A context above 10,000 rows is refused with the count
and the cap rather than quietly dropping rows you named.

The rows to score need not appear in the labelled half, and usually should not:
they are the rows whose answer you do not have.

To score rows of a frame you already hold, naming them by key, ask a predictive
query instead. The same frame then supplies both halves: the rows you name are
scored, the rest become context.

```python
from sdm_client import PredictiveQuery, Schema

customers = pd.DataFrame(
    {
        'customer_id': range(20),
        'age': [30.0 + i for i in range(20)],
        'spend': [100.0 + 20 * i for i in range(20)],
        'churn': [i % 2 for i in range(20)],
    }
)
schema = Schema.from_data(
    {'customers': customers}, primary_keys={'customers': 'customer_id'}
)
request = PredictiveQuery(
    'PREDICT customers.churn FOR EACH customers.customer_id', schema
).materialize(indices=[7, 12, 3], model=KumoTabular())

with SDMClient(url='http://localhost:8000') as client:
    answer = client.predict(
        request, model=KumoTabular(), outputs=['prediction', 'probabilities']
    )
print(request.context_rows, 'rows of context')
print(answer.to_pandas())
```

**Verified** against a live endpoint:

```
17 rows of context
   row_index  id  prediction  probability_0  probability_1
0          0   7           0       0.664439       0.335561
1          1  12           1       0.242538       0.757462
2          2   3           0       0.751313       0.248687
```

Twenty rows in, three named, and `request.context_rows` is 17: the scored rows
are held out of their own context. `materialize()` also samples down to the
10,000-row cap.

On this model the language is cut down to a column of one table as the target:
no aggregation, no window, no comparison, no forecast, no ranking, no entity
filter. A schema with a second table is refused and names the model that can
answer it. [`references/pql.md`](references/pql.md) is the whole of it, with
each refusal quoted.

## Asking the same context more than once

A fitted context can be held open and asked repeatedly, which is worth doing
when the labelled half is large and the rows to score arrive in batches:

```python
first_rows = Request(
    context=labelled, predict=to_score, target='churn', task='classification'
)
more_rows = Request(
    context=labelled,
    predict=pd.DataFrame({'age': [25.0, 60.0], 'spend': [80.0, 600.0]}),
    target='churn',
    task='classification',
)

with SDMClient(url='http://localhost:8000') as client:
    fitted = client.fit(first_rows, model=KumoTabular())
    first = fitted.predict(first_rows)
    second = fitted.predict(more_rows)
    fitted.close()
print(second.to_pandas())
```

**Verified** against a live endpoint:

```
   row_index  prediction  probability_0  probability_1
0          0           0       0.998261       0.001739
1          1           1       0.018095       0.981905
```

`predict()` on a fitted model takes a `Request`, not a bare frame, and the
pinned context is the one it answers from. `fit()` is the only way to build
one: `FittedModel(...)` raises and names `fit()`.

A session pins the context into one process's memory on the endpoint, so a
deployment behind several replicas needs requests for one session to reach the
same one.

## When it refuses

Every refusal is a `StructuredDataError` or one of its subclasses, and each
names what to change.

| Refusal | Cause |
|---|---|
| a task the model does not serve | `forecasting`, `temporal_link_prediction` and the rest are relational |
| an output it does not produce | `rankings`, `embeddings`, `scores` |
| an explanation over more than one row | the endpoint explains one row at a time |
| a context above the cap | more than 10,000 labelled rows in a direct `Request` |
| relationships | this model reads one table; use `kumo-relational` |

Catch `StructuredDataError` to catch all of them.

## Then

- [`references/pql.md`](references/pql.md) — the query language on one table
- [`../relational/SKILL.md`](../relational/SKILL.md) — when the question is about an entity and what it links to
