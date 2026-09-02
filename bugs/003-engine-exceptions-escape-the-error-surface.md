# 3. Engine exceptions escape the SDK's error surface, one advertising an API this SDK does not have

**Severity: medium** — `except StructuredDataError` does not catch everything
this SDK raises, so ordinary error handling has holes.

## Problem

`PredictiveQuery.materialize()` delegates gathering to the relational engine
and does not translate what the engine raises. Engine exception types reach the
caller unchanged:

```python
query.materialize(indices=ids, anchor_time=pd.Timestamp('1990-01-01'))
# ValueError: Anchor timestamp '1990-01-01 00:00:00' is before the earliest
#             timestamp '2016-07-04 00:00:00' in the data.

query.materialize(indices=ids_not_in_the_table)
# TaskReferenceError: Task references 2 unresolved row(s) in table 'Orders'
#                     after graph sanitization

query.materialize(indices=fifteen_hundred_ids)
# ValueError: Cannot predict for more than 1,000 entities at once (got 1,500).
#             Pass `batch_size=` to `predict(...)` (for example
#             `batch_size=500`, or `batch_size='max'`) to process entities in
#             batches.
```

Every one of these is a *user input* error that the SDK documents as its own:
the module docstring and every other refusal in `materialize` raise
`StructuredDataError`. A caller who writes the documented `except
StructuredDataError` catches most bad input and not these.

Two further problems in that third message:

- `batch_size=` is not a parameter of this SDK. `predict()` takes
  `(request, model, outputs)`; `materialize()` takes `indices`, `predict`,
  `num_context_samples` and sampler options. The message instructs the user to
  call an API that does not exist here.
- It names a limit (1,000 entities) that the SDK already knows —
  `RELATIONAL_LIMITS['predict_rows_max']` — and could state itself, before
  doing any work.

`TaskReferenceError` is also not exported from `sdm_client`, so a caller cannot
name it in an `except` clause without importing from the engine package.

## Location

`packages/sdm-client/src/sdm_client/query.py`

- `:321` — `model._get_task_table(...)`, uncaught
- `:340` — `model.materialize_task(...)`, uncaught
- `:299-364` — `_gather`, which has no `try`/`except` around either call

## Why it needs fixing

An SDK's error contract is part of its API. This one is documented, tested and
otherwise consistent — `StructuredDataError` with a `code` — which makes the
gap worse rather than better: the exceptions that escape are the ones a user is
most likely to hit, because they are all caused by ordinary bad input.

The `batch_size=` message is separately damaging for an open-source release: it
sends a reader looking for a parameter that is not there, and it leaks the
shape of a different SDK into this one's output.

## Proposed fix

- Wrap the two engine calls in `_gather` and re-raise as `StructuredDataError`
  with an appropriate `code`, keeping the engine's message as the cause. The
  mapping is small: an anchor/lookup/reference problem is `INVALID_REQUEST`, a
  capacity problem is `UNSUPPORTED_REQUEST`.
- Pre-flight `len(indices)` against `RELATIONAL_LIMITS['predict_rows_max']` in
  `materialize`, before the engine is called, so the entity-count limit is
  named by this SDK in this SDK's vocabulary — as
  `num_context_samples`'s ceiling already is.
- Grep the vendored engine for other user-facing messages that name a retired
  API (`batch_size=`, `predict(...)`); `scripts/check_repo_hygiene.py` already
  scans shipped files for retired interface names and could cover this one.

Tests to add: each of the three inputs above raises `StructuredDataError`; the
entity-count refusal happens before the engine is reached (the same
sentinel technique used in `test_context_samples.py`); no shipped message
mentions `batch_size`.
