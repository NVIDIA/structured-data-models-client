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
| ranking / temporal link prediction | `k` per entity |
| forecasting | one per entity per step |

## Joining

Every row carries `row_index`. It also carries `id` — the entity — when the
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
