# 18. An integer entity key can be declared `float64`, and the endpoint refuses the request

**Severity: high** — a relational request that cannot be served, on a schema
shaped no differently from one that works.

**Status: not fixed.** Found after the fifteen, by testing the built wheels
rather than the source tree. It reproduces identically on `main`, so it is not
something this branch introduced, and fixing it is a separate change.

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

## What it is *not*

Worth recording, because the shape of the trigger is what will make this
findable:

- **not** the context size — `num_context_samples` of 10, 20, 37, 38, 39, 50
  and 200 all produce `float64` on this schema
- **not** the time column — the same two tables without one are `int64`
- **not** the key's type — a string `user_id` is declared `string`, correctly
- **not** every relational schema — Northwind (four tables, integer `OrderID`)
  is `int64` and predicts fine, which is why every probe in
  [`../qa`](../qa/README.md) passes: they all use Northwind

The failing and passing fixtures differ in table count and in the columns each
table carries, and the exact condition was not isolated.

## Why it needs fixing

A relational request that the client builds and the endpoint cannot accept is
the worst shape of failure available: the caller did nothing wrong, the message
names two internal column names they have never seen, and there is no argument
they can change to avoid it. It is also silent about which of their tables is
implicated.

## Where to start

The declaration comes from the engine's payload encoder, not from
`sdm_client.wire` — `sdm_relational_engine/rfm/payload.py`'s `_dtype_name`
reports what the column holds, so the column is genuinely float-typed by the
time it is encoded. That points at how the predict rows are assembled in
`rfm.py` (a merge or reindex that introduces a null and promotes the column)
rather than at the encoder.

The first thing to establish is the exact difference between the two fixtures
above, since one works and one does not.

## Tests to add once it is fixed

The failing fixture, asserting `__kumo_entity_ref_0` matches the dtype of the
key it joins to; the Northwind case, unchanged; a string key, unchanged; and a
live prediction on the failing fixture, which is what would have caught it.
