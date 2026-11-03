# Prediction output

`predict()` always returns a `PredictionResult`, not a DataFrame or an
explanation.

```python
result = client.predict(request, model=KumoRelational())
result.to_pandas()  # always a frame
result.explanation  # None unless you asked for it
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

## Reading an explanation

`result.explanation` remains the raw endpoint payload.
`result.explanation_view` provides an immutable snapshot of that payload, or `None` when there is no explanation.
The view exposes `raw`, nested `details`, `summary`, `warnings`, `cohorts` and `subgraphs`.
Missing fields are `None`; malformed requested fields raise `StructuredDataError` with code `INVALID_REQUEST` and a field path.
Unknown fields remain available through `raw`.

```python
view = result.explanation_view
if view is not None:
    view.print()
    importance = view.feature_importance(normalize=True)
    if importance is not None:
        print(importance.to_pandas())
```

Feature importance is available for `kumo_rfm_v2_1` subgraph cell attribution.
It sums signed scores by table and column, skipping missing/null scores and rejecting nonnumeric or nonfinite scores.
Normalization divides aggregated scores by their total absolute magnitude: `-2, 6` becomes `-0.25, 0.75`.
Zero totals remain zero.
An absent or unsupported attribution returns `None`, while an explicitly empty attribution returns an empty report.
The view reads no prediction rows and makes no warehouse, model or summary-generation calls.
Only an explicit feature-importance request invokes local Rust numerical processing.

Java offers `result.explanationView()`, nullable accessors with the same meaning, `toText()` and `featureImportance(boolean normalize)` returning immutable records.
Python text, print and pandas views are local presentation helpers.

## Schema metadata reports

`schema.metadata_report()` and `schema.link_report()` snapshot the schema's existing table descriptors and directed relationships.
Individual tables expose `table.metadata_report()`.
Reports provide immutable records, `to_text()`, `to_pandas()` and `print()`; `schema.print_metadata()` and `schema.print_links()` are shortcuts.
They do not read source rows, infer new metadata or change the schema.
Java provides corresponding `metadataReport()`, `linkReport()` and `toText()` methods with immutable records.
