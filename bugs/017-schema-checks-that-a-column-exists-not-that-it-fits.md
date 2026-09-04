# 17. `Schema`'s table attributes validate nothing, where the class they wrap validates on assignment

**Severity: medium** — a regression against the engine this SDK is built on:
the same checks exist one package over and fire at the point of assignment.

## Problem

Setting a semantic type, a time column or a relationship is validated for
existence and nothing else. Whether the column can actually *be* that is
decided much later, by the engine, which raises a bare `ValueError`:

```python
schema['users']['age'].stype = 'text'  # accepted, age is float64
query.materialize(indices=[1, 2])
# ValueError: Column 'age' received an incompatible semantic type
#             (got dtype='float' and stype='text')
```

```python
Schema.from_data(tables, time_columns={'users': 'age'})  # accepted
schema['users'].time_column  # 'age'
query.materialize(indices=[1, 2])
# ValueError: Column 'age' received an incompatible semantic type
#             (got dtype='float' and stype='timestamp')
```

```python
Schema.from_data(
    tables,
    relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
)  # accepted
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
schema['Customers'].time_column = 'Country'  # a string column
schema['Customers'].time_column
# 'Country'
```

```python
# 94 rows, 93 distinct CustomerIDs: inference declined to nominate a key
schema['Customers'].primary_key  # None
schema['Customers'].primary_key = 'CustomerID'  # accepted, still not unique
```

`Column.stype` has a setter that validates the name against the five known
values. `Table.primary_key` and `Table.time_column` have no setter at all, so
they accept anything — including a column that is not in the table. The last
example is also the assignment route into
[2](002-non-unique-primary-key-accepted.md): inference was careful enough to
return `None`, and the assignment puts the non-unique key back.

The engine's messages are good — each names the column and the conflict. Two
things are wrong around them.

1. **The timing.** The failure happens at `materialize()`, so the traceback
   points at the query rather than at the assignment that caused it. In a
   notebook those are typically different cells, sometimes minutes apart. The
   information needed to reject it — the column's dtype — is present at
   assignment time.
2. **The type.** `ValueError` is outside this SDK's documented error surface, so
   `except StructuredDataError` misses it. Same defect as
   [3](003-engine-exceptions-escape-the-error-surface.md), reached a different
   way.

## The checks already exist, one package over, and fire eagerly

This is not a question of how much a user should be allowed to override. The
engine `sdm_client` is built on already answers it, in
`sdm_relational_engine/rfm/base/`:

```python
# table.py:480 -- primary_key is a property with a setter
@primary_key.setter
def primary_key(self, name):
    ...  # empty composite key -> ValueError
    ...  # collides with the time column -> ValueError
    ...  # collides with the end-time column -> ValueError
    if name is not None and name not in self:
        raise KeyError(f"Column '{name}' not found in table '{self.name}'")
    ...
    self[name].stype = Stype.ID  # and the dtype check fires here

    # table.py:551 -- time_column likewise, ending in
    self[name].stype = Stype.timestamp


# column.py:134 -- Column.__setattr__
if key == 'stype':
    if not val.supports_dtype(self.dtype):
        raise ValueError(
            f"Column '{self.name}' received an incompatible semantic type "
            f"(got dtype='{self.dtype}' and stype='{val}')"
        )
```

Every check this report asks for is written there and runs at assignment. The
same code is in the open-source predecessor this repo was derived from,
`kumo-relational-engine`, unchanged.

`sdm_client.Table` is a plain `@dataclass`: `primary_key` and `time_column` are
bare attributes with no setter, so the value is simply stored and travels to the
engine at `materialize()` — which is why the `KeyError` above, raised by that
same setter, arrives several cells later than it should.

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

There is no line to draw about how strict to be, which is what made this look
like a judgement call at first: the rule is *whatever the engine will reject
anyway*. `Column.__setattr__` rejects a stype the dtype cannot support and
nothing more, so re-typing a numeric `age` to `categorical` — which the client's
tests deliberately allow, and the docs invite — stays legal. What stops being
legal is only what was going to fail later regardless.

This report overlaps [3](003-engine-exceptions-escape-the-error-surface.md) on
the error type. If bug 3 is fixed first, these become `StructuredDataError` at
materialize time, which is some of the harm gone — what remains is the timing,
which is the part worth fixing separately rather than instead.

## Proposed fix

Do what the engine's setters do, at the same moment they do it.

- Give `Table.primary_key` and `Table.time_column` real setters that mirror
  `rfm/base/table.py:480` and `:551`: the column must exist, must not already
  be the other of the two, and must be able to carry the stype the role
  implies. Raise `StructuredDataError`, not `KeyError` or `ValueError`, since
  this is the client's surface.
- Give `Column.stype` the dtype check from `column.py:134` — no stricter, so
  every re-typing the docs invite keeps working.
- Apply the same checks to the `primary_keys=`, `time_columns=` and
  `relationships=` constructor arguments, so the two routes cannot diverge.
  Uniqueness belongs here too, per
  [2](002-non-unique-primary-key-accepted.md).
- `relationships=`: require both columns' dtypes to be joinable and the target
  to be the target table's primary key, refusing with the offending pair named.
- Keep the engine's checks where they are. They are the backstop, and after
  this they should be unreachable from the public API — which is the test.

Tests to add: each of the four constructor cases raises `StructuredDataError`
at the point of assignment; `primary_key = 'NoSuchColumn'`,
`time_column = <a string column>` and `primary_key = <a non-unique column>` are
each refused; every legitimate edit in the class docstring still works; a
relationship between two non-key columns is refused.
