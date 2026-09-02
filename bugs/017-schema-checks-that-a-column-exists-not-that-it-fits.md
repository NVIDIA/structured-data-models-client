# 17. `Schema` checks that a column exists, not that it can mean what you said — and after construction, not even that

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

## After construction there is no check at all

The constructor at least verifies the column exists. Direct assignment — which
is the documented way to correct a schema — does not:

```python
schema['Customers'].primary_key = 'NoSuchColumn'
schema['Customers'].primary_key
# 'NoSuchColumn'

query.materialize(indices=[...])
# KeyError: "Column 'NoSuchColumn' not found in table 'Customers'"
```

A bare `KeyError` from a dictionary lookup — not even the engine's `ValueError`,
which at least explains itself.

```python
schema['Customers'].time_column = 'Country'   # a string column
schema['Customers'].time_column
# 'Country'
```

```python
# 94 rows, 93 distinct CustomerIDs: inference declined to nominate a key
schema['Customers'].primary_key            # None
schema['Customers'].primary_key = 'CustomerID'   # accepted, still not unique
```

`Column.stype` has a setter that validates the name against the five known
values. `Table.primary_key` and `Table.time_column` have no setter at all, so
they accept anything — including a column that is not in the table. The last
example is also the assignment route into
[2](002-non-unique-primary-key-accepted.md): inference was careful enough to
return `None`, and the assignment puts the non-unique key back.

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
- `Table` — `primary_key` and `time_column` are plain attributes with no
  setter, so post-construction assignment is unchecked entirely

## Why it needs fixing

Editing the schema is the documented way to fix inference — the class docstring
demonstrates `schema['users']['zip_code'].stype = 'categorical'` — so a wrong
edit is a normal thing for a user to make, and getting the diagnosis three
cells later is the difference between an obvious mistake and a puzzle.

The compatibility rules are already written down, in the engine, and are
already being enforced. This is about running them where the user can act on
them.

**Scope, honestly stated.** The *stype* half is the weakest part of this
report: setting a column's stype against its dtype is exactly the correction
the docs invite, and the client tests deliberately allow re-typing (a numeric
`age` to `categorical`, for instance), so where to draw the line is a judgement
about how much a user may override. Only the combinations the engine will later
reject outright should be refused, not every combination that looks unusual.

The *existence* half is not a judgement call. `primary_key = 'NoSuchColumn'` is
accepted and produces a `KeyError` later; no reading of "let the user correct
the schema" covers naming a column that is not there.

This report also overlaps [3](003-engine-exceptions-escape-the-error-surface.md)
on the error type. If bug 3 is fixed first, these raise `StructuredDataError`
at materialize time, which is most of the harm gone — what remains is the
timing, which is what makes it worth fixing separately rather than instead.

## Proposed fix

Move the checks the engine already performs to where the value is set:

- `Column.stype` setter: reject a combination the engine will reject —
  `timestamp` on a non-datetime column, `numerical` on a string, `text` on a
  numeric — as `StructuredDataError`, reusing the engine's wording. The dtype
  is on the `Column` already.
- Give `Table.primary_key` and `Table.time_column` setters that run the same
  checks as the constructor arguments, so the two routes cannot diverge. A
  `Table` holds its own frame, so both the existence and the uniqueness checks
  are local.
- `time_columns=` and `.time_column =`: require the named column be a
  timestamp.
- `relationships=`: require both columns' dtypes to be joinable and the target
  to be the target table's primary key, refusing with the offending pair named.
- Keep the engine's checks where they are. They are the backstop, and after
  this they should be unreachable from the public API — which is the test.

Tests to add: each of the four constructor cases raises `StructuredDataError`
at the point of assignment; `primary_key = 'NoSuchColumn'`,
`time_column = <a string column>` and `primary_key = <a non-unique column>` are
each refused; every legitimate edit in the class docstring still works; a
relationship between two non-key columns is refused.
