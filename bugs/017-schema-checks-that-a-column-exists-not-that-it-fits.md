# 17. `Schema` checks that a column exists, not that it can mean what you said

**Severity: medium** — the error arrives at the wrong time, in the wrong type,
pointing at the wrong line.

## Problem

Setting a semantic type, a time column or a relationship is validated for
existence and nothing else. Whether the column can actually *be* that is
decided much later, by the engine, which raises a bare `ValueError`:

```python
schema['users']['age'].stype = 'text'          # accepted, age is float64
query.materialize(indices=[1, 2])
# ValueError: Column 'age' received an incompatible semantic type
#             (got dtype='float' and stype='text')
```

```python
Schema.from_data(tables, time_columns={'users': 'age'})   # accepted
schema['users'].time_column                                # 'age'
query.materialize(indices=[1, 2])
# ValueError: Column 'age' received an incompatible semantic type
#             (got dtype='float' and stype='timestamp')
```

```python
Schema.from_data(tables, relationships=[
    Relationship('orders', 'user_id', 'users', 'user_id')])   # accepted
# users.user_id is a string, orders.user_id is an int
query.materialize(indices=['1', '2'])
# ValueError: Edge(src_table='orders', fkey='user_id', dst_table='users') is
#             invalid as foreign key 'user_id' and primary key 'user_id' have
#             incompatible dtypes
```

A fourth case is not caught anywhere: a relationship between two columns that
are not keys at all (`Relationship('orders', 'price', 'users', 'age')`) builds
a schema without complaint.

The engine's messages are good — each names the column and the conflict. Two
things are wrong around them.

1. **The type.** `ValueError` is outside this SDK's documented error surface, so
   `except StructuredDataError` misses it. Same defect as
   [3](003-engine-exceptions-escape-the-error-surface.md), reached a different
   way.
2. **The timing.** The failure happens at `materialize()`, so the traceback
   points at the query rather than at the assignment that caused it. In a
   notebook those are typically different cells, sometimes minutes apart. The
   information needed to reject it — the column's dtype — is present at
   assignment time.

## Location

`packages/sdm-client/src/sdm_client/schema.py`

- `Column.__setattr__` / `_check_stype` — validates the stype is one of the
  five known names, not that it suits the column's dtype
- `_require_column` — existence only, used for `primary_keys`, `time_columns`
  and both ends of a `Relationship`
- `Schema.from_data`'s `relationships` branch — checks both columns exist, not
  that their dtypes can join or that either is a key

## Why it needs fixing

Editing the schema is the documented way to fix inference — the class docstring
demonstrates `schema['users']['zip_code'].stype = 'categorical'` — so a wrong
edit is a normal thing for a user to make, and getting the diagnosis three
cells later is the difference between an obvious mistake and a puzzle.

The compatibility rules are already written down, in the engine, and are
already being enforced. This is about running them where the user can act on
them.

## Proposed fix

Move the checks the engine already performs to where the value is set:

- `Column.stype` setter: reject a combination the engine will reject —
  `timestamp` on a non-datetime column, `numerical` on a string, `text` on a
  numeric — as `StructuredDataError`, reusing the engine's wording. The dtype
  is on the `Column` already.
- `time_columns=`: require the named column be a timestamp.
- `relationships=`: require both columns' dtypes to be joinable and the target
  to be the target table's primary key, refusing with the offending pair named.
- Keep the engine's checks where they are. They are the backstop, and after
  this they should be unreachable from the public API — which is the test.

Tests to add: each of the four cases above raises `StructuredDataError` at the
point of assignment; every legitimate edit in the docs still works; a
relationship between two non-key columns is refused.
