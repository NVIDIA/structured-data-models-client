# 2. A non-unique primary key is accepted, and the model silently sees one of the duplicates

**Severity: high** — a wrong answer, with no signal that anything went wrong.

## Problem

A primary key is what every relationship in a schema joins on and what a
prediction is correlated back by. Nothing checks that it is unique.

Two paths are affected, and neither refuses:

1. **Inference.** `_infer_primary_key` accepts a column whose distinct count is
   at least 99% of its length. 1,005 rows with 1,000 distinct ids is 99.5%, so
   `user_id` is nominated as the primary key despite five duplicates.

   ```python
   base = pd.DataFrame({'user_id': range(1000), 'v': [float(i) for i in range(1000)]})
   dupes = pd.concat([base, base.head(5)], ignore_index=True)
   Schema.from_data({'users': dupes})['users'].primary_key
   # 'user_id'   <- five rows share an id
   ```

2. **Explicit override.** `primary_keys={'Customers': 'CustomerID'}` is checked
   for existence only (`_require_column`), never for uniqueness. At any
   duplicate rate at all.

## Reproduced

End to end against the relational NIM, on Northwind with one customer row
duplicated under the same key and different attributes:

```
   CustomerID      Country          City
0       ALFKI      Germany        Berlin
93      ALFKI  ZZZ-PLANTED  PLANTED-CITY
```

```
   row_index     id  prediction
0          0  ALFKI    2.178293
```

One prediction, no warning, and no way for the caller to know which of the two
`ALFKI` rows the model was shown — or whether the join to `Orders` attached
one customer's orders to the other's features.

## Location

`packages/sdm-client/src/sdm_client/schema.py`

- `:67` — `_KEY_UNIQUENESS = 0.99`
- `:417` — `if distinct >= len(frame) * _KEY_UNIQUENESS:` — inference
- `:253` — `for name, key in (primary_keys or {}).items():` — the override
  path, which calls `_require_column` and nothing else

## Why it needs fixing

Duplicate keys are not exotic; they are what a real extract looks like after a
fan-out join, an un-deduplicated CDC feed, or an SCD-2 table read without its
`is_current` filter. The SDK is pointed straight at warehouses where all three
are ordinary.

The failure is silent and directional: a duplicate does not raise, it changes
which row's features the model sees, and the answer that comes back is
plausible. A user cannot audit for this, because the SDK reports nothing.

The 99% threshold is also self-inconsistent: it is deliberately loose so that
inference is robust, but the value it produces is then used as an exact join
key. Robustness at nomination time becomes silence at join time.

## Proposed fix

Uniqueness is a property of a key, not a heuristic. Check it where the key is
established, on both paths:

- Inference: require `distinct == len(frame)` (nulls excluded) for a column to
  be *nominated* as a primary key. Keep the 99% heuristic, if it is wanted, as
  a signal for which column to *consider*, but do not return a non-unique one.
  A column that is nearly unique should yield `primary_key = None`, which
  already makes the caller say what they mean.
- Override: validate the named column is unique and non-null, and refuse with a
  message that names the column, the duplicate count, and one offending value.
- The check costs one `nunique()` per candidate column, which the code already
  computes at `:416`.

Tests to add: exactly-unique passes; one duplicate in a thousand is refused on
both the inferred and the explicit path; a null in the key column is refused;
the error names the column and an offending value.
