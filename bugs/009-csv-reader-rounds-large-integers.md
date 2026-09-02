# 9. The CSV reader rounds large integers when the column has a blank, on a path the SDK already guards elsewhere

**Severity: high** — silent data corruption in the connector, before any of the
wire-encoding guards can see it.

## Problem

`_read_file` calls bare `pd.read_csv`. A blank in an integer column makes
pandas promote the whole column to `float64`, which rounds every value past
2**53. By the time the frame reaches the SDK it is already wrong, so the
serialization guards — which do handle this correctly for a real integer
column — have nothing left to catch.

```csv
row_id,x,y
9007199254740993,1,0
,2,1
9007199254740995,3,0
```

```python
frame = read('local', path='bigint.csv')
frame['row_id'].dtype          # float64
list(frame['row_id'])          # [9007199254740992.0, nan, 9007199254740996.0]
# on the wire: [9007199254740992.0, None, 9007199254740996.0]
```

Both ids are wrong, in different directions, and the file on disk is correct.

The same file with no blank row reads as `int64` and travels exactly. So the
corruption is triggered by a missing value elsewhere in the column.

## The SDK already solves this — for the other path

`_arrow_to_pandas`, in the same module, exists precisely for this:

> The numpy default represents nulls in an integer column by promoting it to
> `float64`, which silently rounds values beyond 2**53. Such columns use a
> null-aware integer dtype instead.

So warehouse reads through Arrow are protected and CSV reads are not. The
analysis is already written down; it just was not applied here.

## Location

`packages/sdm-connectors/src/sdm_connectors/reader.py:383`

```python
return pd.read_csv(path, storage_options=storage_options)
```

No `dtype_backend`, no `dtype` hint, no post-read repair.

Applies to `source='local'` and `source='s3'` alike, and to compressed CSV.

## Why it needs fixing

An id column with occasional blanks is completely ordinary in exported CSV —
it is what a left join, an optional foreign key, or a partial extract produces.
The result is a wrong id, plausibly shaped, with no error anywhere.

This is the same failure the `Decimal` bug ([1](001-decimal-columns-lose-precision.md))
produces by a different route, which is the argument for fixing both: the wire
layer's integer-exactness rule is only as good as the weakest path into it.

## Proposed fix

Read CSV with a null-aware backend so an integer column with a blank stays an
integer:

```python
pd.read_csv(path, storage_options=storage_options, dtype_backend='numpy_nullable')
```

That yields `Int64` rather than `float64`, which `infer_tfm_dtype` already
reports as `int64` and `serialize_cell` already stringifies past the JS-safe
range — the probe for a nullable integer with a null passes today.

Check the knock-on effects before committing to it: `dtype_backend` also
changes float and string columns to nullable dtypes, which the wire layer
handles but the tests pin. If that turns out too broad, the narrower fix is to
re-parse only those columns that came back `float64` and whose non-null values
are all integral, which is the repair `_arrow_to_pandas` performs.

Tests to add: the CSV above round-trips exactly, locally and through S3; a
float column with a blank is still a float; an existing all-integer CSV is
unchanged.
