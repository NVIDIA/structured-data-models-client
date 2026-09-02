# 5. An empty `predict` frame costs a round trip and returns a result with no columns

**Severity: medium** — a `KeyError` in the caller's code, from a call that
looked like it succeeded.

## Problem

`Request(predict=<0-row frame>)` is accepted, sent to the endpoint, and comes
back as a DataFrame with **no columns at all**:

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

So the documented way to use a result fails on it:

```python
frame['prediction']                          # KeyError: 'prediction'
to_score.merge(frame, on='row_index')        # KeyError: 'row_index'
```

A zero-row frame is what a filter returns when nothing matches, so this lands
in the middle of an ordinary loop: `predict(df[df.segment == s])` works for
every segment until one is empty, and then it is a `KeyError` rather than an
empty result.

The SDK already refuses the equivalent on the other entry point:

```python
query.materialize(indices=[])
# StructuredDataError: indices is empty, so there is nothing to score
```

A related case: an **empty context** is also accepted, and the endpoint answers
`500 INTERNAL_ERROR Unexpected server error.` A context with no rows is
locally knowable — a model that predicts from labelled examples cannot be given
none — and should not become a 500.

## Reproduced

Live against the tabular NIM. `predict=` with 0 rows: HTTP 200, empty result,
no columns. `context=` with 0 rows: `[500 INTERNAL_ERROR] Unexpected server
error.`

## Location

`packages/sdm-client/src/sdm_client/request.py` — `Request.__init__` validates
columns, dtypes and the target, but not row counts.

`packages/sdm-client/src/sdm_client/client.py:231-244` — `predict` sends
whatever it is given.

## Why it needs fixing

Two things are wrong and only one of them is the round trip. The wasted call is
minor; the shape of the result is not. An empty result that carries no columns
is a different type from a non-empty one, so a caller cannot write code that
handles both. Every other refusal in this SDK is a sentence; this one is a
`KeyError` several lines later, in the caller's code, naming a column the
caller never mistyped.

## Proposed fix

- Refuse an empty `predict` frame in `Request.__init__` with the same voice as
  the `indices=[]` refusal: *"predict has no rows, so there is nothing to
  score"*, code `INVALID_REQUEST`.
- Refuse an empty `context` frame likewise: a model predicts from labelled
  examples, and zero of them is not a question that can be asked.
- If refusing an empty `predict` is judged too strict — an empty batch inside a
  loop is arguably a legitimate no-op — then return an empty result **with the
  correct columns** (`row_index`, `prediction`, and the probability columns the
  task implies) so that a join and a column access both still work. Refusing is
  simpler and matches the rest of the SDK; either is better than today.

Tests to add: an empty `predict` raises with a message naming the frame; an
empty `context` raises; neither reaches the transport (assert by replacing the
transport with a failing sentinel).
