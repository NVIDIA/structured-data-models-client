# 1. A `Decimal` column loses precision, and walks past the guard that exists to stop it

**Severity: high** — silent data corruption, on the default integer type of two
of the four warehouses this SDK ships connectors for.

## Problem

`DECIMAL` / `NUMBER` is how a warehouse spells an exact number. The Databricks
driver hands those columns back as Python `Decimal` objects in an `object`
column. The SDK types such a column as `float64` and sends every cell through
`float()`, so any value that `float64` cannot represent exactly is silently
rounded before it leaves the process.

```
Decimal('9007199254740993')                -> 9007199254740992.0
Decimal('123456789012345678901234567890')  -> 1.2345678901234568e+29
Decimal('12345678901234567.8901')          -> 1.2345678901234568e+16
```

The same value as a Python `int` is **refused**:

```python
serialize_cell(9007199254740993, 'float64')          # StructuredDataError
serialize_cell(Decimal('9007199254740993'), 'float64')  # 9007199254740992.0
```

The guard is already written. It just does not fire, because it asks whether
the value is an `Integral`, and `Decimal` is not one:

```python
>>> isinstance(9007199254740993, numbers.Integral)
True
>>> isinstance(Decimal('9007199254740993'), numbers.Integral)
False
```

The guard is type-based rather than value-based, so `Decimal` is not the only
thing that walks past it. A numpy integer does too, because numpy's comparison
promotes both sides to float before comparing:

```python
serialize_cell(9007199254740993, 'float64')            # StructuredDataError
serialize_cell(Decimal('9007199254740993'), 'float64') # 9007199254740992.0
serialize_cell(np.int64(9007199254740993), 'float64')  # 9007199254740992.0
```

Two more instances of the same blind spot in the same file:

- `_require_finite` tests `isinstance(value, (float, np.floating))`, so
  `Decimal('Infinity')` is not caught. It is not silently sent — `requests`
  encodes with `allow_nan=False` and the call fails — but it fails in the wrong
  layer, as `StructuredDataError(TRANSPORT_ERROR)`: *"Request to
  .../v1/predictions failed: Out of range float values are not JSON
  compliant: inf"*. A `float('inf')` in the same column is refused before the
  request is built, by name, as `INVALID_REQUEST`. Same value, same column, two
  different error codes and two different layers.
- `serialize_cell`'s `int64` branch checks fractional loss only for `float`, so
  a `Decimal` with a fractional part would be truncated by `int()` rather than
  refused — unreachable today only because `decimal` never maps to `int64`.

## Reproduced

Live, against Databricks (`kumo_test_catalogue.nvidia_erp`):

```sql
create table t (id decimal(38,0), amount decimal(38,4), label string);
insert into t values (9007199254740993, 12.3456, 'a');
```

```python
frame = read_table(connection, query='select * from t')
frame['id'].dtype                      # object
type(frame['id'].iloc[0])              # decimal.Decimal
serialize_column(frame['id'], infer_tfm_dtype(frame['id']))
# ['9007199254740992.0']  <- the id is now a different id
```

Snowflake happens to escape this: its driver returns `NUMBER(38,0)` as `int64`,
which takes the correct path and stringifies past the JS-safe range. So the
same table modelled in two warehouses gives two different answers.

## Location

`packages/sdm-client/src/sdm_client/wire/dtypes.py`

- `:43` — `_OBJECT_CONTENT_DTYPES['decimal'] = 'float64'`
- `:170` — `if isinstance(value, Integral) and number != value:` — the guard
  that `Decimal` walks past
- `:132` — `_require_finite`, same blind spot for `Decimal('Infinity')`
- `:155` — the `int64` fractional-loss check, same blind spot

## Why it needs fixing

An id is the one column that must survive exactly. A rounded id joins a
prediction to the wrong row, and nothing downstream can detect it — the value
is a perfectly plausible number. It is worse than an outright failure because
it is invisible, and it lands on the path this SDK actively recommends: read
from a warehouse, predict.

The wire contract requires integers outside the JS-safe range to travel as
base-10 strings precisely to prevent this. `Decimal` bypasses that rule.

## Proposed fix

Classify a `decimal` object column by its contents rather than assuming
`float64`, and make the three guards value-based rather than type-based.

- In `infer_tfm_dtype`, when the content is `decimal`, report `int64` if every
  non-null value has a zero exponent and fits `int64`, and `float64` otherwise.
  That routes exact integers — ids, counts, cents-as-integers — into the branch
  that already stringifies past 2**53.
- Widen the `float64` precision guard from `isinstance(value, Integral)` to
  anything integral **by value**, `Decimal` and `np.integer` included, so a
  value `float64` cannot hold is refused whatever type carried it.
- Widen `_require_finite` to any non-finite numeric, so `Decimal('Infinity')`
  is refused as `INVALID_REQUEST` before the request is built, the way
  `float('inf')` already is, rather than as a transport error afterwards.
- A `Decimal` too large for `int64` should be refused by name rather than
  rounded — say which column and value, and suggest sending it as a string.

**A fractional `Decimal` is not fixed by any of this, and should be decided
explicitly.** `Decimal('12345678901234567.8901')` cannot be represented by
`float64`, and the wire has no exact decimal type, so routing it to `float64`
is lossy however it is classified. Two honest options: refuse it, naming the
column and suggesting it be sent as a string; or accept the loss and say so in
the docs. What should not stand is the current position, which is to lose the
precision silently. This report's title is about integers; the fractional case
is a smaller, separate decision that the same fix should force someone to make.

Tests to add: a `Decimal` past 2**53 in both an `int64`-shaped and a fractional
column, `np.int64` past 2**53 in a `float64` column, `Decimal('Infinity')`,
`Decimal('NaN')`, and a regression test asserting that a Python `int`, the
equal `Decimal` and the equal `np.int64` are all treated identically.
