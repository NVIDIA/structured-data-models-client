# 18. An integer entity key can be declared `float64`, and the endpoint refuses the request

**Severity: high** — a relational request that cannot be served, on a schema
shaped no differently from one that works.

Found after the fifteen, by testing the built wheels rather than the source
tree. It reproduced identically on `main`, so this branch did not introduce it.

## Problem

The transport key the engine generates for the entity being scored,
`__kumo_entity_ref_0`, is declared `float64` while the primary key it joins to
is `int64`. The endpoint refuses the request:

```
422 INVALID_SCHEMA relationship joins '__kumo_entity_ref_0' declared 'float64'
                   to 'user_id' declared 'int64'
```

The payload says so plainly:

```python
schema.instance_table.columns
# {'instance_id': 'int64',
#  '__kumo_entity_ref_0': 'float64',   <- joins to an int64 key
#  '__kumo_anchor_time': 'timestamp[us]',
#  'TARGET': 'string'}
schema.relationships
# [{'source_columns': ['__kumo_entity_ref_0'], 'target_table': 'users',
#   'target_columns': ['user_id']}, ...]
```

## Reproduced

```python
users = pd.DataFrame({'user_id': range(40),
                      'age': [20.0 + i for i in range(40)],
                      'tier': ['gold' if i % 2 else 'silver' for i in range(40)]})
orders = pd.DataFrame({'order_id': range(120),
                       'user_id': [i % 40 for i in range(120)],
                       'price': [10.0 + i for i in range(120)],
                       'placed': pd.date_range('2024-01-01', periods=120, freq='D')})
schema = Schema.from_data({'users': users, 'orders': orders})
PredictiveQuery('PREDICT users.tier FOR EACH users.user_id', schema).materialize(
    indices=[1, 2, 3], num_context_samples=200)
# -> __kumo_entity_ref_0 declared float64; the endpoint then refuses it
```

## The cause

`_entity_values` read the key one row at a time:

```python
values.append(_json_value(df.iloc[row_index][table.primary_key]))
```

`df.iloc[row_index]` builds a Series for the **whole row**, and a row has a
single dtype. Put an integer key beside a float column and the row is `float64`,
so `user_id` comes back as `20.0`:

```python
>>> df = pd.DataFrame({'user_id': [1, 2, 3], 'age': [20.0, 30.0, 40.0]})
>>> df.iloc[0]['user_id']
np.float64(1.0)
>>> df['user_id'].iloc[0]
np.int64(1)
```

That is the whole trigger, and it explains every observation: bisecting the
fixtures, adding the float `age` column to the entity table is what flipped the
reference from `int64` to `float64`. Northwind escaped it because the columns
beside `OrderID` are strings, and an object row keeps the integer. A string key
escaped it because there is nothing to promote to. The context size and the time
column never mattered.

**The correct form was already in this file**, twelve lines below, with a
comment explaining that a row-wise read *"materializes a whole row Series per
edge, which dominates request assembly"* — someone hit the cost of this mistake
and fixed it there. The type side went unnoticed.

## Why it needs fixing

A relational request that the client builds and the endpoint cannot accept is
the worst shape of failure available: the caller did nothing wrong, the message
names two internal column names they have never seen, and there is no argument
they can change to avoid it. It is also silent about which of their tables is
implicated.

## Location

`packages/sdm-relational-engine/src/sdm_relational_engine/rfm/payload.py:536`
— `_entity_values`.

## The fix

Read the column, then the row. The key column is taken once outside the loop
and indexed positionally, which preserves its own dtype — and is what the
gather below it already does.

Tests in `packages/sdm-client/tests/test_entity_reference_dtype.py`: the failing
fixture, a string key, the same graph without a time column, a temporal
aggregation, and the entity values themselves — because the wrong read gave
`1.0` for `1`, so the dtype was only the visible half.

Verified against the live relational NIM: the request that was refused now
scores, and the ids correlate back as `['1', '2', '3']`.
