# 14. A `SUM` over an integer column in DuckDB comes back rounded

**Severity: high** — an ordinary aggregate returns the wrong number, silently.
This is the fourth route past the integer-exactness rule.

## Problem

DuckDB widens `sum(BIGINT)` to `HUGEINT`. `read_table` tries the driver's fetch
methods in a fixed order, and DuckDB offers two of them — `fetchdf()` and
`fetch_arrow_table()`. `fetchdf()` is tried first, and it rounds.

```python
connection.execute('create table t as select 9007199254740993::BIGINT id from range(3)')
read_table(connection, query='select sum(id) s from t')['s']
# dtype float64, value 2.702159776422298e+16
# true sum:                27021597764222979
```

The other path is exact:

```python
cursor.to_arrow_table().column(0)[0]     # Decimal128Scalar 27021597764222979
cursor.to_arrow_table().to_pandas()['s'] # object, Decimal('27021597764222979')
cursor.fetchdf()['s']                    # float64, 2.702159776422298e+16
```

So the value survives DuckDB, survives Arrow, and is lost by the pandas fetch
the connector happens to reach first.

The same applies to a plain `HUGEINT` column and to any DuckDB expression that
produces one: `sum`, `count(*) * something`, `int128` literals, and arithmetic
that would overflow `BIGINT`.

## This is the same rule failing a fourth way

`_arrow_to_pandas` exists in the same module for exactly this hazard, and its
docstring says so:

> The numpy default represents nulls in an integer column by promoting it to
> `float64`, which silently rounds values beyond 2**53.

The Arrow path is careful. This query never reaches it. Verified clean by
comparison: Parquet `Int64` with a null, Parquet `int64`, DuckDB `BIGINT` with
a null, SQLite `integer` with a null, and CSV without a blank all travel
exactly.

Note the link to [1](001-decimal-columns-lose-precision.md): fixing the fetch
order does not finish the job. The Arrow path yields `object`-of-`Decimal`,
which is precisely the input bug 1 rounds. The two chain — DuckDB `sum()` is
only exact once both are fixed.

## Location

`packages/sdm-connectors/src/sdm_connectors/reader.py:165-183` — the fetch
ladder in `read_table`:

```python
if hasattr(cursor, 'fetch_arrow_all'): ...      # DuckDB: no
if hasattr(cursor, 'fetch_pandas_all'): ...     # DuckDB: no
if hasattr(cursor, 'fetchdf'):                  # DuckDB: yes  <- taken, lossy
    return cursor.fetchdf()
if hasattr(cursor, 'fetch_arrow_table'): ...    # DuckDB: yes, exact, unreached
```

The exact method is already in the list, two lines below the lossy one.

## Why it needs fixing

`SELECT sum(...)` is not an edge case; it is the most common thing anyone does
to a numeric column. The answer is wrong, plausible, and arrives with no
warning. Anyone summing cents, bytes, event counts or ids past 2**53 gets a
number that is close enough to look right.

DuckDB is also the connector most likely to be used casually — it needs no
credentials and reads Parquet in place — so this is on the shortest path a new
user takes.

## Proposed fix

Prefer Arrow over the driver's pandas conversion. Move the
`fetch_arrow_table` / `fetchall_arrow` branches **above** `fetchdf`, so a driver
that offers both is read through the path that preserves the value and goes on
to `_arrow_to_pandas`'s null-aware repair.

That is a reordering, not a rewrite, and it is the same preference the ladder
already expresses at the top — `fetch_arrow_all` is tried before
`fetch_pandas_all`. `fetchdf` simply ended up on the wrong side of it.

Then extend `_arrow_to_pandas` to decide what a scale-0 decimal becomes:

- fits `int64` → `Int64`, which the wire layer already stringifies past the
  JS-safe range;
- wider than `int64` → leave it as `object`-of-`Decimal`, and let
  [1](001-decimal-columns-lose-precision.md)'s fix route or refuse it. Never
  `float64`.

Decimals with a scale are a separate judgement, and the same one bug 1 raises:
`float64` cannot hold them either, so the choice is to refuse or to document
the loss — not to take it silently.

Tests to add: `sum()` over a `BIGINT` column past 2**53 is exact; a `HUGEINT`
column is exact or refused, never rounded; a `DECIMAL(38,4)` keeps its scale; a
driver offering both `fetchdf` and `fetch_arrow_table` is read through Arrow;
the existing nullable-integer tests still pass.
