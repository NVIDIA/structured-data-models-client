# Prediction output

`predict()` always returns a `PredictionResult`. Never a DataFrame sometimes
and an explanation other times.

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
`rankings`, and under `prediction` when that is asked for, as a list per entity.
Three entities ranked `TOP 3` come back as three rows, not nine.

## What a regression reports

The relational model predicts a distribution for a regression, and `prediction` is its median.
That matches what the engine and kumo-relational-client return.
Pass `materialize(..., prediction_statistic='mean')` for the mean instead.

Choose `'mean'` when predictions will be added up across entities, such as total demand over products.
Means add up and medians don't, so on a right-skewed target the sum of the medians understates the total.
The mean is approximate: the NIM averages the model's 999 predicted quantiles, which trims the tails.
On a zero-inflated target the median is exactly 0 for every entity the model predicts to be zero at least half the time.

A regression on the tabular model reports the mean, that model's default, and doesn't take `prediction_statistic`.

## Joining

Every row carries `row_index`. It also carries `id` – the entity – when the
model has one; a single-table request may have none.

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
