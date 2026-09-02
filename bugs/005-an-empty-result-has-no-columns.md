# 5. An empty `predict` frame returns a result with no columns at all

**Severity: medium** — a `KeyError` in the caller's code, several lines after a
call that returned `200`.

## What is not the bug

A zero-row table is legal. `nim-sd.openapi.yaml` puts `minItems: 1` on
`ArraysTableData.columns` and on `OutputSpec.fields`, and deliberately does
**not** put it on `rows`:

```yaml
columns:
  type: array
  minItems: 1        # <- required to be non-empty
rows:
  type: array
  maxItems: 10000    # <- no minItems
```

So sending zero rows is within the contract, and an earlier draft of this
report was wrong to propose refusing it. A zero-row batch inside a loop is a
legitimate no-op.

## The bug

The result of that no-op is not an empty result — it is a frame with **no
columns**:

```python
result = client.predict(
    Request(context=context, predict=frame.head(0), target='income',
            task='classification'),
    model=KumoTabular())
result.to_pandas()
# Empty DataFrame
# Columns: []
# Index: []
```

so the documented way to use a result fails on it:

```python
frame['prediction']                          # KeyError: 'prediction'
to_score.merge(frame, on='row_index')        # KeyError: 'row_index'
```

An empty result is a different *type* from a non-empty one, so a caller cannot
write one piece of code that handles both. `predict(df[df.segment == s])` works
for every segment until one filters to nothing, and then it is a `KeyError`
naming a column the caller never mistyped.

## A second case, which is not legal

An empty **context** is accepted and the endpoint answers
`500 INTERNAL_ERROR Unexpected server error.` Unlike a zero-row `predict`, this
one is locally knowable and cannot be meaningful — a model that predicts from
labelled examples cannot be given none. The client already refuses the
equivalent on the other entry point:

```python
query.materialize(indices=[])
# StructuredDataError: indices is empty, so there is nothing to score
```

## Location

`packages/sdm-client/src/sdm_client/result.py` — `to_pandas` builds columns
from the returned predictions, so no predictions means no columns.

`packages/sdm-client/src/sdm_client/request.py` — `Request.__init__` validates
columns, dtypes and the target, not row counts.

## Why it needs fixing

The wasted round trip is the minor half. The shape of the result is the real
one: it makes the empty case fail differently from the non-empty case, in the
caller's code, at a point that gives no hint where the problem came from.

## Proposed fix

- `to_pandas()` should return an empty frame **with the columns the request
  implies** — `row_index`, `prediction`, and the probability columns for a
  classification task — so a join and a column access both still work on it.
  The task and requested outputs are on the result already.
- Refuse an empty **context** locally, in the voice of the `indices=[]`
  refusal, rather than letting it become a 500.
- Leave an empty `predict` accepted. It is contract-legal, and a no-op batch is
  a reasonable thing to hand a loop.

Tests to add: an empty `predict` returns a zero-row frame whose columns match a
non-empty one for the same request, and `merge(on='row_index')` succeeds
against it; an empty `context` raises `StructuredDataError` without reaching
the transport.
