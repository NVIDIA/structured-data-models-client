# Prediction output

`predict()` always returns a `PredictionResult`, not a DataFrame or an
explanation.

```python
result = client.predict(request, model=KumoRelational())
result.to_pandas()  # always a frame
result.explanations  # empty unless you asked for them
result.metadata  # whatever the endpoint reported
```

## How many rows come back

The shape depends on the **task**, not the model.

| Task | Rows |
|---|---|
| classification | one per entity |
| multiclass classification | one per entity, with a probability per class **as columns** |
| regression | one per entity |
| ranking / temporal link prediction | one per entity, the `k` candidates **in a list** |
| forecasting | one per entity per step |

Ranking is the exception to the row-per-answer rule: expanding the candidates
would make `row_index` no longer identify a row, so they arrive under
`rankings`, and under `prediction` when that is asked for, as a list per
entity. Three entities ranked `TOP 3` come back as three rows, not nine.

## A ranking can hold fewer than k candidates

A ranking is asked as one yes-or-no question per candidate: will this entity link to it in the window?
Each question learns from a sampled context of other entities, and if none of them linked to a candidate in the window, that context holds one answer only.
The endpoint refuses such a question, so the client leaves that candidate out rather than refusing the whole ranking.
No call is sent for it, and it gets no score: it is in no row's `prediction` or `rankings`.

The result names the candidates it left out, in the order you passed them:

```python
result = client.predict(request, model=KumoRelational())
result.unscored_candidates  # (13, 14), or () when every candidate was scored
```

Each row then ranks only the scored candidates, so it can hold fewer than `k`.
The client warns once, naming the candidates it left out.
A candidate's context is shared by every entity in the ranking, so a candidate left out for one entity is left out for all.
If every candidate is left out there is nothing to rank, and `predict()` raises `UnsupportedRequestError` before sending anything.

A larger `num_context_samples` can reach entities that did link to a candidate.
On sparse data, such as a catalogue where most products sell rarely, it often can't.

**Recommended pattern: rank over a table of pairs.**
When you can provide a table, or a view, with one row per (entity, candidate) pair, ask one classification over the pairs instead.
Each pair is scored from the features of its entity and its candidate, and the context pools every pair, so a candidate nobody linked to recently is still scored.
It is one call rather than one per candidate.
The events table needs a `pair_id` column linking each event to its pair.
The pairs table should hold the pairs you want scored, plus enough other pairs, some that linked and some that didn't, for the context to learn from.

```python
schema = Schema.from_source(
    source,
    primary_keys={
        'customers': 'customer_id',
        'products': 'product_id',
        'orders': 'order_id',
        'pairs': 'pair_id',
    },
    time_columns={'orders': 'ordered_at'},
    relationships=[
        Relationship('orders', 'pair_id', 'pairs', 'pair_id'),
        Relationship('pairs', 'customer_id', 'customers', 'customer_id'),
        Relationship('pairs', 'product_id', 'products', 'product_id'),
    ],
)
# Here a pair_id is '<customer_id>_<product_id>'; rank customer 42's candidates.
request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH pairs.pair_id',
    schema,
).materialize(indices=[f'42_{product}' for product in candidates])
scores = client.predict(request, model=KumoRelational()).to_pandas()
ranking = scores.sort_values('probability_True', ascending=False)
```

## What a regression reports

The relational model predicts a distribution for a regression, and `prediction` is its median.
That matches what the engine returns.
Pass `materialize(..., prediction_statistic='mean')` for the mean instead.

Choose `'mean'` when predictions will be added up across entities, such as total demand over products.
Means add up and medians don't, so on a right-skewed target the sum of the medians understates the total.
The mean is approximate: the NIM averages the model's 999 predicted quantiles, which trims the tails.
On a zero-inflated target neither statistic is pinned to 0, and both can land slightly below it, because the predicted quantiles are not clipped to the target's range.

A regression on the tabular model reports the mean, that model's default, and doesn't take `prediction_statistic`.

## Joining

Every row carries `row_index`. It also carries the entity `id` when the model
has one; a single-table request may have none.

**Join on the key when it is there, otherwise on `row_index`. Never on row
order.** The endpoint is not required to return rows in the order they were
sent, and a request split across several calls certainly will not.

```python
answers = result.to_pandas()
joined = to_score.reset_index(names='row_index').merge(answers, on='row_index')
```

## Columns

| Field | Column(s) |
|---|---|
| `prediction` | `prediction` |
| `probabilities` | `probability_<class>`, one per class |
| `quantiles` | `quantile_<level>`, one per level |
| `scores`, `embeddings` | one column holding a list |
| `rankings` | one column holding a list of dicts |

## Explanations

Ask for one by adding `'explanation'` to `outputs`. There is no row limit:
every scored row is explained. A NIM and a client opened with
`SDMClient.in_process()` both serve it, with the same request and the same
`result.explanations`.

```python
result = client.predict(request, model=KumoRelational(),
                        outputs=['prediction', 'probabilities', 'explanation'])
```

In the query, `EXPLAIN PREDICT …` means the same as `outputs=['explanation']`;
passing both is allowed and redundant. A ranking refuses it, since it is
assembled client-side from per-candidate classifications; a forecast explains
each step.

### Keys

`result.explanations` maps a row's key to that row's explanation. It is kept
out of `result.to_pandas()`. `result.explanation` is deprecated, see
`result.explanations`; there is no `{'by_forecast_step': ...}`.

The key is whatever identifies the row in `to_pandas()`, by the same rule on
both models:

| Request | Key | Example |
|---|---|---|
| has an entity id, none repeated | the row's `id`, the same value as in `to_pandas()['id']` | `result.explanations['user_5']` |
| forecast | `(id, forecast_step)`, or `(row_index, forecast_step)` when ids repeat | `result.explanations[('user_5', 3)]` |
| no id column, or its ids repeat | `row_index` | `result.explanations[0]` |

### One entry

Each entry is a plain nested dict:

```python
{
  'outputs':  [...],                        # the explained scores, in order
  'instance': {                             # kumo-tabular only
      'columns':   [...],                   # model's preprocessed numerical columns
      'gradients': [[...], ...],            # [output][column]
  },
  'related': {                              # kumo-relational only: every table,
                                            # the entity's own included
      '<table>': {
          'keys':      [...],               # each row's key; only when a relationship joins on the key column
          'rows':      [{...}, ...],        # each row's values as sent to the model
          'columns':   [...],               # model's preprocessed numerical columns
          'gradients': [[[...], ...], ...], # [output][row][column], rows aligned with keys and rows
      },
  },
}
```

`outputs` is copied from the endpoint:

- classification gives every class probability in declared order, such as
  `{'kind': 'probability', 'class': 'yes'}`;
- regression gives the returned prediction,
  `{'kind': 'prediction', 'statistic': 'mean'}` (or `'median'`), plus each
  returned quantile when `quantiles` is requested.

A gradient is the change in that output score per unit change in that
preprocessed column. It is a diagnostic, not a causal effect or a
feature-importance ranking.

`columns` are the model's preprocessed numerical columns, not the columns you
sent. Categoricals and datetimes may appear as derived numeric columns, and a
table without numerical features has an empty `columns`.

### Which tables get gradients

| Model | Tables with gradients |
|---|---|
| kumo-tabular | the one table: each predicted row's own columns |
| kumo-relational | the predicted entity's own table, plus every related table in the subgraph the client gathered for that entity (all tables reached within the hop limit, up to 6) |

A `kumo-tabular` entry has no `related` key: its one table is the `instance`.
A `kumo-relational` entry has no `instance` key: the entity's own table is one
of the tables under `related`, at hop 0.

Only the rows being predicted and their related rows get gradients. The
context rows, the labelled examples the model compares against, get none.

Gradients are per row, not per path. A row reached through several paths, for
example a product bought in two of the user's orders, appears once. Its
gradient is its total effect through all of them, and the model gives no split
by path.

### Related rows

A related table's rows always carry `rows`, each row's values as sent to the
model. `keys` is present when the table's key column is one the model receives,
i.e. a column a relationship joins on; otherwise rows are identified by their
values.
Keyed or keyless changes only how a row is labelled; both get gradients in the
same way.

### Examples

Values are illustrative. `kumo-tabular`, request with an id column:

```python
result.explanations['cust_17']
# {'outputs': [{'kind': 'probability', 'class': 'no'},
#              {'kind': 'probability', 'class': 'yes'}],
#  'instance': {'columns': ['age', 'balance'],
#               'gradients': [[0.01, -0.03], [-0.01, 0.03]]}}
```

`kumo-relational`, user 5 with one order and two login events. `users` carries
`keys` because `user_id` is the join column; `orders` and `events` are
identified by their values. All are part of the same entry:

```python
result.explanations['user_5']
# {'outputs': [{'kind': 'probability', 'class': 'no'},
#              {'kind': 'probability', 'class': 'yes'}],
#  'related': {
#      'users':  {'keys': ['user_5'],
#                 'rows': [{'user_id': 'user_5', 'tenure_days': 410}],
#                 'columns': ['tenure_days'],
#                 'gradients': [[[-0.05]], [[0.05]]]},
#      'orders': {'rows': [{'user_id': 'user_5', 'order_id': 'order_9003', 'amount': 12.0}],
#                 'columns': ['amount'],
#                 'gradients': [[[-0.02]], [[0.02]]]},
#      'events': {'rows': [{'user_id': 'user_5', 'kind': 'login', 'at': '2026-10-01T09:00:00'},
#                          {'user_id': 'user_5', 'kind': 'login', 'at': '2026-10-03T18:20:00'}],
#                 'columns': ['kind_login', 'at_dayofweek'],    # model-derived names, illustrative
#                 'gradients': [[[0.01, 0.00], [0.02, -0.01]],
#                               [[-0.01, 0.00], [-0.02, 0.01]]]}}}
```

### As one table

`result.explanations.to_pandas()` gives one line per (row, output, column),
across all entities:

```
id      table     hop  key         row  output           column       gradient
user_5  users     0    user_5      0    probability_yes  tenure_days   0.05
user_5  orders    1    None        0    probability_yes  amount        0.02
user_5  products  2    prod_77     0    probability_yes  price         0.004
user_5  events    1    None        1    probability_yes  kind_login   -0.02
```

- `table` is the table's name. On `kumo-relational` the entity's own table
  (`users` here) is hop 0. On `kumo-tabular` the one table is shown as
  `instance`, at hop 0.
- `hop` is how many links the row is from the entity, 0 for the entity's own
  table. The client works it out from the relationships and foreign-key
  values it sent. It is `None` for a row no relationship reaches.
- `key` is the row's entry in `keys`, or `None` when the table has no `keys`.
  On a `kumo-tabular` `instance` line it is the row's `id`, or `None` when the
  request has no entity key. `row` is the row's position in that entity's
  `related[table]['rows']`, and 0 on a `kumo-tabular` `instance` line.
- `output` names the score by the column of `result.to_pandas()` that holds
  it: `probability_yes`, `prediction`, `quantile_0.1`.
