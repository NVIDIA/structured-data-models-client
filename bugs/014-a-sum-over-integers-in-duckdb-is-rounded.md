# 14. A `SUM` over an integer column in DuckDB comes back rounded

**Severity: high** — an ordinary aggregate returns the wrong number, silently.
This is the fourth route past the integer-exactness rule.

## Problem

DuckDB widens `sum(BIGINT)` to `HUGEINT`, and hands it over Arrow as
`decimal128(38, 0)`. `_arrow_to_pandas` maps a fixed list of Arrow integer
types to pandas nullable integers; `decimal128` is not on it, so `to_pandas`
falls back to `float64` and the value is rounded on arrival.

```python
connection.execute('create table t as select 9007199254740993::BIGINT id from range(3)')
read_table(connection, query='select sum(id) s from t')['s']
# dtype float64, value 2.702159776422298e+16
# true sum:                27021597764222979
```

The Arrow table is exact — `27021597764222979` — and the loss happens in the
conversion the connector performs.

The same applies to a plain `HUGEINT` column and to any DuckDB expression that
produces one: `sum`, `count(*) * something`, `int128` literals, and arithmetic
that would overflow `BIGINT`.

## This is the same rule failing a fourth way

`_arrow_to_pandas` was written for exactly this hazard, and its docstring says
so:

> The numpy default represents nulls in an integer column by promoting it to
> `float64`, which silently rounds values beyond 2**53.

It fixes that for `int8` … `uint64`. It does not cover the decimal types, which
is how the same rounding gets back in. Verified clean by comparison: Parquet
`Int64` with a null, Parquet `int64`, DuckDB `BIGINT` with a null, SQLite
`integer` with a null, and CSV without a blank all travel exactly.

Note the link to [1](001-decimal-columns-lose-precision.md): Arrow
`decimal128(38, 0)` is also what a Snowflake `NUMBER(38,0)` and a Databricks
`DECIMAL(38,0)` are. Databricks' driver converts it to Python `Decimal` objects
and the wire layer rounds them; DuckDB's converts it to `float64` and the
connector rounds it. One Arrow type, two conversions, both lossy, two different
places to fix.

## Location

`packages/sdm-connectors/src/sdm_connectors/reader.py:32-41` —
`_NULLABLE_INTEGER_DTYPES`, which has no decimal entry.

`packages/sdm-connectors/src/sdm_connectors/reader.py:198` —

```python
df = arrow_table.to_pandas(types_mapper=_NULLABLE_INTEGER_DTYPES.get)
```

`types_mapper` is a function, so it can decide per-type rather than by lookup.

## Why it needs fixing

`SELECT sum(...)` is not an edge case; it is the most common thing anyone does
to a numeric column. The answer is wrong, plausible, and arrives with no
warning. Anyone summing cents, bytes, event counts or ids past 2**53 gets a
number that is close enough to look right.

DuckDB is also the connector most likely to be used casually — it needs no
credentials and reads Parquet in place — so this is on the shortest path a new
user takes.

## Proposed fix

Replace the dict lookup with a function that also handles decimals:

```python
def _pandas_dtype_for(arrow_type):
    if arrow_type in _NULLABLE_INTEGER_DTYPES:
        return _NULLABLE_INTEGER_DTYPES[arrow_type]
    if pa.types.is_decimal(arrow_type) and arrow_type.scale == 0:
        return pd.Int64Dtype()   # if the precision fits int64
    return None
```

- Scale-0 decimals that fit `int64` become `Int64`, which the wire layer
  already stringifies past the JS-safe range.
- A scale-0 decimal too wide for `int64` should stay `object`-of-`Decimal` and
  be handled by [1](001-decimal-columns-lose-precision.md)'s fix, which routes
  exact integers to `int64` and refuses what cannot be represented. Do not let
  it become `float64`.
- Decimals with a scale (money) are a separate judgement: `float64` loses
  precision there too, but that loss is inherent to the wire's `float64`, so
  the right answer is probably to keep `Decimal` and let bug 1's guard refuse
  the values that cannot survive.

Both fixes are needed. Neither one alone closes the route, because the two
drivers convert the same Arrow type differently.

Tests to add: `sum()` over a `BIGINT` column past 2**53 is exact; a `HUGEINT`
column is exact or refused, never rounded; a `DECIMAL(38,4)` keeps its scale;
the existing nullable-integer tests still pass.
