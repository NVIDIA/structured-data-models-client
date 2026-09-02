# 10. A column that is empty in the context turns real values into strings in the rows being scored

**Severity: medium** — changes what the model is asked to predict on, silently.

## Problem

A column's wire dtype is taken from the context frame. If that column happens
to have no values in the context, it is typed `string`, and then the widening
rule turns the whole column to `string` — so the numbers in the frame being
scored are sent as text.

```python
Request(
    context=pd.DataFrame({'row_id': [1, 2], 'x': [None, None], 'y': [0, 1]}),
    predict=pd.DataFrame({'row_id': [3], 'x': [42]}),
    target='y', task='classification')
```

produces

```
schema.instance_table.columns['x'] = {'dtype': 'string', 'stype': 'categorical'}
predict.instance_table.rows        = [[3, '42']]
```

`42` is now the category `'42'`. The model sees a categorical feature with one
never-before-seen level instead of a number, and answers accordingly. No error,
no warning.

`infer_tfm_dtype` is deliberately careful here — it reports `UNTYPED` for an
object column with no non-null values, *"so that a frame which happens to be
missing a feature entirely cannot re-type the same column in another frame"*.
That is exactly the right reasoning. `_dtype_of` then converts `UNTYPED` to
`'string'` before the widening runs, which discards it.

## Reproduced

Live, against the tabular NIM: the request is accepted and two rows are scored.
Nothing surfaces the substitution; it is visible only by inspecting the payload.

## Location

`packages/sdm-client/src/sdm_client/schema.py:369` — the schema dtype is taken
from the context column.

`packages/sdm-client/src/sdm_client/schema.py:380-384` — `_dtype_of` maps
`UNTYPED` to `'string'`.

`packages/sdm-client/src/sdm_client/adapters/tabular.py:131-136` — widening
`'string'` with `'int64'` gives `'string'`
(`wire/dtypes.py:110-111`).

## Why it needs fixing

A feature that is null throughout a sample is normal — a column that only
applies to some segment, a recently added field, a small context batch. The SDK
already anticipated the problem and built `UNTYPED` for it; the value is thrown
away one function later.

The failure is quiet and it changes the question, which is worse than a
rejected request: the caller gets an answer that looks fine and was computed on
different data than they supplied.

## Proposed fix

Let `UNTYPED` survive to the widening, where it already means "contributes
nothing":

- Keep `UNTYPED` in the per-frame dtype and let `widen_tfm_dtype` ignore it, so
  the column takes the dtype of whichever frame actually has values — `int64`
  in the example above.
- If **every** frame is untyped for a column, then there genuinely is no
  information, and `'string'` is the right last resort; keep it for that case
  only.
- Consider naming it: a column that is entirely null in the context and
  populated in `predict` is worth a warning, because the model has no examples
  of it.

Tests to add: an all-null context column with integers to score is declared
`int64` and sent as numbers; the same with floats, timestamps and strings; a
column null in both frames is still `string`; the existing "all-null column"
tests keep passing.
