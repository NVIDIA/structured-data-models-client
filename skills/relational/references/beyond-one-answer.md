# Rankings, forecasts and explanations

Three questions that are not one prediction per entity.
[`../SKILL.md`](../SKILL.md) is the plain path; this is what sits beyond it,
and it builds its own schema so it can be read on its own:

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
```

A ranking asks which of a set of candidates an entity will reach, and comes
back as a list in one row rather than a row per candidate, because expanding it
would stop the key identifying a row:

```python
request = PredictiveQuery(
    'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2 '
    'FOR EACH users.user_id',
    schema,
).materialize(indices=[1, 2], candidates=items['item'], model=KumoRelational())

with SDMClient(url='http://localhost:8000', timeout=600) as client:
    print(client.predict(request, model=KumoRelational()).to_pandas())
```

**Verified** against a live endpoint:

```
   row_index  id prediction                                           rankings
0          0   1     [1, 3]  [{'candidate': 1, 'score': 0.9960135221481323}...
1          1   2     [2, 3]  [{'candidate': 2, 'score': 0.9957279562950134}...
```

**Two columns carry the ranking.** `prediction` is the candidates in order,
as a bare list; `rankings` is the same order with a score each, as
`{'candidate', 'score'}` dicts. Neither is asked for: a ranking brings both
without an `outputs=` argument.

`candidates` is the set to rank: every value an entity might reach, as a
Series or a plain list. It is required, and the refusal names it if you forget.
Each candidate costs a call, so the 200-call fan-out is counted in candidates,
not in `k`.


A candidate whose gathered context holds only one answer cannot be scored, so
it is left out of the ranking and named on `result.unscored_candidates`, with a
warning. That happens when the labels do not vary — every entity in the sample
reached that candidate, or none did — and the second is the common one. The ranking still fills `TOP k` from
whatever was scored, and comes back shorter only when fewer than `k` candidates
survive — so a short list means check `result.unscored_candidates`, and a full
one does not mean every candidate was considered.

A forecast asks the same question over several future windows, one call per
step, and gives a row per entity per step. The clause goes after the target:

```python
request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FORECAST 3 TIMEFRAMES '
    'FOR EACH users.user_id',
    schema,
).materialize(indices=[1, 2], model=KumoRelational())

with SDMClient(url='http://localhost:8000', timeout=600) as client:
    print(client.predict(request, model=KumoRelational()).to_pandas())
```

**Verified** against a live endpoint:

```
   row_index  id  forecast_step  prediction
0          0   1              1    2.173828
1          0   1              2    2.929688
2          0   1              3    2.144531
3          1   2              1    2.181641
4          1   2              2    2.912109
5          1   2              3    2.185547
```

`forecast_step` counts from one, and `row_index` repeats, so neither alone
identifies a row: a forecast is keyed by the pair.

Neither can be fitted into a session. Each part needs the context its own
labels come from, so a ranking and a forecast are asked with `predict()` and
`fit()` refuses them.

An explanation is requested with `outputs=[..., 'explanation']` or `EXPLAIN PREDICT`, not an `explain=` argument.
For `sdm_gradient_v1`, read each prediction through `result.explanations.gradients` and its numerical gradients through `to_pandas()`.
The model must support that output; client-assembled rankings refuse it locally.
`result.explanation` remains the singular raw compatibility view, including unknown formats.

Each entry contains `outputs` and `related`, one block per sampled table, including the entity's own table at hop zero.
Blocks carry submitted `rows`, model-derived numerical `columns` and `gradients[output][row][column]`.
Source keys can identify related rows; synthetic transport keys are not user keys.
Only scored rows and their sampled related rows receive gradients, not the labelled context examples.
A gradient measures sensitivity to a preprocessed feature, not a causal effect.

```python
entry = result.explanations.gradients[1]  # the entity whose id is 1
block = entry['related']['orders']        # one block per table, by name
block['rows'][0]                          # the first row's values, as sent
block['gradients'][0][0]                  # first output, first row: one gradient per name in block['columns']
```

Unique prediction IDs key the collection; missing/repeated IDs use `row_index`.
Forecast keys include the step, and merging preserves distinct rows and steps.
`result.explanations.gradients.to_pandas()` flattens entries with table, hop, related-row identity, output and column.
`EXPLAIN` ranking is rejected before gathering; explanations requested later through `outputs` are checked before prediction.

```python
result.explanations.gradients[(1, 2)]      # on a forecast: entity 1, step 2
result.explanations.gradients.to_pandas()  # id, forecast_step on a forecast, table, hop, key, row, output, column, gradient
```

Column explanations are on `result.explanations.columns`, which is `None` unless `explanation` was asked for.
This client computes them from the labelled context `materialize()` gathered; the endpoint does not.
That context includes the rows held back from the call because they carried the answer.
Column explanations group the labelled entities into cohorts by each column of each table they reach.
Key columns and the columns a relationship joins on are left out.
Each table also gets `COUNT(*)`, how many of its rows each entity reaches.
An entity whose rows fall in several cohorts is divided among them in proportion to its rows: a user with one category-A order and nine category-B orders counts as 0.1 of an entity in cohort A and 0.9 in cohort B.
`share` and `mean` are computed over these fractional weights, so they measure how the entities' rows are associated with each cohort, not the plain rate among entities that ever touched it.
`share` is the fraction of the population included in that column analysis that belongs to the cohort; `share: 0.2` means 20%.
Missing numerical values are excluded, so this population can differ between columns.
Consider how well each cohort is represented before interpreting differences; a small share does not necessarily mean a small absolute sample size.
For regression, `mean` is the average observed target within the cohort, in the target's units, and `class` is `None`.
For example, for an order total in dollars, `mean: 120` means an average observed total of $120.
For classification, `class` names the target label and `mean` is the proportion of the cohort with that observed label.
For example, `class: "churned"` and `mean: 0.3` means a weighted 30% of the cohort churned; it is not an individual prediction's probability.
Compare means within the same column and class, and the same table and hop.
These are associations in the context, not gradients and not causes.
A forecast gets one analysis, over the context of every step.
If the statistics cannot be computed, `result.explanations.columns` is `None` and a warning says so; the predictions still come back.

```python
analysis = result.explanations.columns
if analysis is not None:
    cohorts = analysis.to_pandas()  # table, hop, column, stype, cohort, class, share, mean, score
    cohorts[cohorts['table'] == 'orders']  # the cohorts of each column of orders
```

See [the result contract](../../../docs/reference/prediction-output.md).
