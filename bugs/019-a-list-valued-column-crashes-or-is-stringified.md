# 19. A list-valued column crashes in one place and is silently stringified in another

**Severity: high** — one of the two outcomes is wrong data on the wire, with no
error.

## Problem

Trying a multi-label target — the reasonable thing to try, since SDM's own
`Task.classification` is documented as *"Binary, multi-class, or multi-label
classification"* — produced this:

```python
ctx = pd.DataFrame({'row_id': range(60), 'v': [...],
                    'tags': [['a', 'b'] if i % 2 else ['b'] for i in range(60)]})
Request(context=ctx, predict=pr, target='tags', task='classification')
# TypeError: unhashable type: 'list'
```

raised from inside pandas' hashtable, because `_stype_of` calls
`series.nunique()` to choose between `categorical` and `text`. A bare
`TypeError` from a library the caller never called — the error-surface leak
[3](003-engine-exceptions-escape-the-error-surface.md) closed everywhere else.

That is the visible half. The quiet half is worse:

```python
>>> infer_tfm_dtype(pd.Series([['a', 'b'], ['b']]))
'string'
>>> serialize_column(pd.Series([['a', 'b'], ['b']]), 'string')
["['a', 'b']", "['b']"]
```

`pd.api.types.infer_dtype` reports a list column as `mixed`, which falls through
to `'string'`, and `serialize_cell` ends in `str(value)`. So a list column that
reached the wire travelled as the **repr of a Python list** — the literal text
`"['a', 'b']"` — as an ordinary category. No crash, no warning, and the model
asked about text that was never in the data.

**Both were reachable.** A schema is built from the context and the related
tables, never from the rows being scored, so a list present only in `predict`
skipped the crash and took the quiet path:

```python
Request(context=<scalar tags>, predict=<list tags>, ...)
# predict rows on the wire: [[900, "['a', 'b']"]]
```

`Request(..., schema=...)` and `PredictiveQuery`'s scoring path skip schema
inference too.

## Multi-label is not the fix

It is worth being explicit, because the two questions look like one. The
contract carries a `stringlist` dtype and a `multicategorical` stype, and the
relational engine's encoder already emits both correctly. But `CellValue` — the
type of a *prediction* — says why they will not help here:

> Scalar JSON value used for prediction outputs. Table rows use TableCellValue
> so that the stringlist dtype can carry JSON arrays **without permitting
> array-valued prediction scalars**.

Excluding an array from a prediction is deliberate. `probabilities` is a flat
`label -> float` map with no stated meaning as independent label marginals and
no threshold, and there is no field for a chosen label set. Every mention of
`stringlist` in the contract scopes it to temporal link prediction labels — the
one task the model owner has confirmed is **not** supported.

So a multi-label answer has no shape to come back in, and enabling it is a
contract change plus an endpoint that serves it. Refusing it clearly is the
whole of what this client can honestly do.

## Location

`packages/sdm-client/src/sdm_client/wire/dtypes.py` — `infer_tfm_dtype` has no
list branch; `serialize_cell` ends in `str(value)`.

`packages/sdm-client/src/sdm_client/schema.py` — `_stype_of`'s `nunique()`.

`packages/sdm-client/src/sdm_client/request.py` — the only place that knows
which column is the target.

## The fix

One check, `require_scalar_cells`, called from three places, because each knows
something the others do not:

- `serialize_column` — every table reaches the wire through it, including the
  rows being scored, which no schema describes. This is the load-bearing one.
- `_infer_table` — before any column is described, so the message can name the
  table and the error arrives before pandas raises its own.
- `Request.__init__` — where the target is known, so a list *target* is refused
  as **"a multi-label target"** rather than as a typing problem. A feature says
  "a list-valued column is not something this client can send".

**The check asks whether a value is a container, not what type it is.** That
took four rounds to arrive at: the list of types was missing a tuple, then a
mapping (an Arrow `struct` column yields dicts, stringified as `"{'a': 1}"`),
then a `range`, then a nested `Series`. Anything iterable that is not a `str`
is refused; a `str` is the exception because that is exactly what a text column
holds, and everything a scalar column legitimately carries — a number, a
`Decimal`, a `UUID`, a timestamp — is not iterable at all. Both directions are
pinned: eleven container shapes refused, ten scalar shapes still accepted.

`serialize_column` is on the hot path for every table in every request, so the
check returns immediately for a dtype that cannot hold a sequence. That question
is asked of the dtype rather than by elimination, because `is_object_dtype`
alone would call an **Arrow-backed list column** scalar — and that is exactly
what `pd.read_parquet(dtype_backend='pyarrow')` gives for one, which would have
put `"['a', 'b']"` back on the wire by a different door.

Arrow is asked through `pa.types.is_nested`, not by listing the nested types:
an earlier attempt enumerated list, large_list, fixed_size_list, struct and map,
and a **list view** went straight through the gap. The types that *wrap* another — a dictionary,
a run-end encoding, an extension type — are unwrapped first, since a dictionary
of lists, or an extension over a list, is still a column of lists. The
extension case is tested with `BaseExtensionType` rather than `ExtensionType`,
because pyarrow's own canonical extensions are the former: `fixed_shape_tensor`
stores a fixed-size list and pandas yields Python lists for it. On a 10,000-row frame the guard costs 0.02 ms for an integer column,
0.00 ms for an Arrow string one, and 0.80 ms for an `object` string one.

Tests in `packages/sdm-client/tests/test_list_valued_columns.py`: the target,
the feature, the predict-only case that skipped the schema, the repr that used
to go out, each sequence type, and that an ordinary frame is unaffected.
