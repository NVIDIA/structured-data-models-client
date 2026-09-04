# 4. A `uint64` column is declared `int64` while carrying values `int64` cannot hold

**Severity: medium** — the payload contradicts its own schema.

## Problem

`infer_tfm_dtype` reports `int64` for every integer dtype, `uint64` included.
A `uint64` column can hold values up to `2**64 - 1`, which is more than twice
what `int64` can:

```python
series = pd.Series([2**64 - 1, 1], dtype='uint64')
infer_tfm_dtype(series)  # 'int64'
serialize_column(series, 'int64')  # ['18446744073709551615', '1']
```

The schema declares `int64` and the payload carries a value outside `int64`.
What happens then is the server's business — it may reject the request, or
wrap the value, or store it in something wider — but the client should not put
it in the position of finding out.

The `int64` claim is deliberate and documented for narrow widths (`int8`,
`int32` and so on all fit and the server decodes through `int()`), and that
reasoning is sound. It just does not extend to `uint64`, which is the one
integer dtype that does not fit.

## Reproduced

```python
>>> from sdm_client.wire.dtypes import infer_tfm_dtype, serialize_column
>>> s = pd.Series([2**64 - 1], dtype='uint64')
>>> infer_tfm_dtype(s), serialize_column(s, infer_tfm_dtype(s))
('int64', ['18446744073709551615'])
```

The nullable dtype behaves the same way:

```python
>>> s = pd.Series([2**64 - 1, None], dtype='UInt64')
>>> infer_tfm_dtype(s), serialize_column(s, infer_tfm_dtype(s))
('int64', ['18446744073709551615', None])
```

`uint64` is not exotic: it is what Parquet writes for an unsigned column, what
`pd.read_parquet` gives back, and what hash-derived ids commonly use.

The relational path has the same hole in its own encoder —
`sdm_relational_engine/rfm/payload.py`'s `_dtype_name` and `_cell_json_value`
make the same "every integer is `int64`" choice — so a fix in `wire/dtypes.py`
covers the tabular path only.

## Location

`packages/sdm-client/src/sdm_client/wire/dtypes.py:82`

```python
if pd.api.types.is_integer_dtype(dtype):
    return 'int64'
```

`is_integer_dtype` is true for `uint64`. There is no bounds check in
`serialize_cell`'s `int64` branch (`:154-166`), which stringifies anything past
the JS-safe range without asking whether it is past the `int64` range too.

## Why it needs fixing

A payload whose values contradict its declared schema is a bug regardless of
how the server responds, and the response is the part not under this SDK's
control. If the server is lenient today it may not be tomorrow; if it wraps,
the corruption is silent.

## Proposed fix

Two options; the second is better.

1. Report `string` for a `uint64` column. Correct, but it changes what the
   model is asked to predict on, which is the thing `infer_tfm_dtype`'s
   docstring is careful to avoid.
2. Keep `int64` when every value fits, and refuse by name when one does not:
   in `serialize_cell`'s `int64` branch, raise `StructuredDataError` for a
   value outside `[-2**63, 2**63 - 1]`, saying which value and suggesting the
   column be sent as strings. That keeps the common case unchanged — a
   `uint64` column of small ids still travels as `int64` — and turns the
   impossible case into a sentence instead of a contradiction.

Tests to add: `uint64` inside the `int64` range keeps working; `2**63` and
`2**64 - 1` are refused by name, in both `uint64` and `UInt64`; the refusal is
a `StructuredDataError`; the same holds on the relational payload path.

Worth doing first: send `2**64 - 1` to a live endpoint and record what it
actually does. If it wraps or truncates rather than rejecting, this is silent
corruption and belongs with the high-severity three.
