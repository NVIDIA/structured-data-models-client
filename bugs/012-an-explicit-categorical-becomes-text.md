# 12. An explicitly declared `Categorical` becomes `text` once it has 65 levels

**Severity: medium** — the SDK overrides what the user has already said, into a
semantic type the relational endpoint refuses.

## Problem

`_stype_of` decides `categorical` against `text` purely by counting distinct
values, with the cut at 64. It does not look at the pandas dtype, so a column
the caller explicitly declared as `pd.Categorical` is re-typed as free text as
soon as it has a 65th level:

```python
cats = [f'c{i}' for i in range(65)]
schema = Schema.from_data({'t': pd.DataFrame({
    'row_id': range(65), 'cat': pd.Categorical(cats), 'y': [i % 2 for i in range(65)]})})
schema['t']['cat'].stype
# 'text'
```

`pd.Categorical` is the strongest statement pandas offers about a column's
meaning, and it is the one signal `_stype_of` ignores.

This then collides with the relational endpoint, which refuses `text`
outright (`422 UNSUPPORTED_DTYPE`), so the column is not merely mis-typed — it
makes the request fail. US states, ISO country codes, SKUs, airport codes,
account types: all are categorical, all exceed 64 levels, none are prose.

## Location

`packages/sdm-client/src/sdm_client/schema.py:62` —
`_MAX_CATEGORICAL_CARDINALITY = 64`

`packages/sdm-client/src/sdm_client/schema.py:396` —

```python
return 'categorical' if distinct <= _MAX_CATEGORICAL_CARDINALITY else 'text'
```

The threshold itself is a reasonable heuristic for an untyped string column;
the problem is that it is applied to a column that is not untyped.

## Why it needs fixing

The heuristic's own comment explains it is there because *"'text' on a 3-value
column wastes the signal"* — the goal is to infer meaning where none was
declared. When the caller has declared it, inference should defer, not
override. Ignoring an explicit `Categorical` inverts the intent.

The consequence is worse than a poor guess because of the endpoint's refusal:
the column becomes unusable, and the error the user sees names a semantic type
rather than the column (see
[8](008-endpoint-limits-are-not-declared-locally.md)).

## Proposed fix

- In `_stype_of`, return `'categorical'` for any column whose dtype is
  `pd.CategoricalDtype`, regardless of cardinality, before the distinct-count
  branch runs.
- Leave the 64-value threshold exactly as it is for `object` and `string`
  columns, where it is doing the job it was written for.
- `stype` is already writable, so a caller can still override in either
  direction; this only changes what they have to override.

Tests to add: a 65-level `Categorical` infers `categorical`; a 65-distinct
plain string column still infers `text`; a 3-level `Categorical` is unchanged;
the round trip of a 65-level categorical against the relational endpoint
succeeds.
