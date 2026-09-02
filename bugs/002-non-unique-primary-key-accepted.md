# 2. A primary key that is knowingly not unique is used to join and correlate, and nothing says so

**Severity: medium** — a wrong answer that the code already knows might be
wrong, and does not mention.

## What is deliberate, and is not the bug

`_infer_primary_key` accepts a column that is 99% distinct, and the comment
says exactly why:

```python
# A column is only a candidate identifier if nearly every value is distinct.
# Exactly 1.0 would reject a key with a single duplicated row, which is common
# in an extract and still the column the user means.
_KEY_UNIQUENESS = 0.99
```

That reasoning is sound and this report does not argue with it. Nominating a
near-unique column is the right call.

## What is the bug

The nominated key is then used as a join key and as the identity a prediction
is correlated back by — and the duplicate the threshold knowingly tolerated is
never mentioned to anyone.

```python
base = pd.DataFrame({'user_id': range(1000), 'v': [float(i) for i in range(1000)]})
dupes = pd.concat([base, base.head(5)], ignore_index=True)
Schema.from_data({'users': dupes})['users'].primary_key
# 'user_id'  -- 99.5% distinct, five rows share an id, no warning
```

End to end on Northwind, with one customer duplicated under the same key and
different attributes:

```
   CustomerID      Country          City
0       ALFKI      Germany        Berlin
93      ALFKI  ZZZ-PLANTED  PLANTED-CITY
```

```
   row_index     id  prediction
0          0  ALFKI    2.178293
```

One prediction. Which of the two `ALFKI` rows the model was shown is not
knowable from anything the SDK returns, and whether the join to `Orders`
attached one customer's orders to the other's features is not knowable either.

The contract is explicit that this matters. `nim-sd.openapi.yaml` requires a
*"unique non-null `instance_id` for each request instance"* and that
*"`instance_id` is a declared, non-null ID column and is unique within"* the
request. The client's primary key is what becomes that identity, so a duplicate
collapses two entities into one instance before the endpoint can object.

## The override and assignment paths have no threshold at all

The 99% reasoning covers inference. Neither other route is covered by it:

```python
Schema.from_data(tables, primary_keys={'Customers': 'CustomerID'})
# existence-checked only, at any duplicate rate

schema['Customers'].primary_key = 'CustomerID'
# no check whatsoever -- see bug 17
```

The second is the sharper case: on the 94-row Northwind fixture, inference
*declined* to nominate a key (93/94 = 0.989 < 0.99) and returned `None`. A
direct assignment puts the non-unique key straight back, past the judgement
that had just been made.

## Location

`packages/sdm-client/src/sdm_client/schema.py`

- `:67` — `_KEY_UNIQUENESS = 0.99` (deliberate; not the bug)
- `:417` — `if distinct >= len(frame) * _KEY_UNIQUENESS:` — where a tolerated
  duplicate becomes a silent one
- `:253` — the `primary_keys` override, `_require_column` and nothing else
- `Table.primary_key` — a plain attribute, no setter, no check

## Why it needs fixing

The threshold decides to tolerate an imperfection. Tolerating it is fine;
tolerating it *silently* is not, because the consequence is not a degraded
answer but a possibly-misattributed one, and the user has no way to discover it.

Duplicate keys are precisely what the comment says they are — common in an
extract. That is the argument for the threshold, and equally the argument for
saying when it fired.

## Proposed fix

Keep the threshold. Report what it tolerated.

- When inference nominates a key that is not exactly unique, record it on the
  `Table` (e.g. `primary_key_duplicates: int`) and surface it in `Schema`'s
  `__repr__`, so it is visible in the notebook where the schema is printed.
- Emit a `warnings.warn` naming the column, the duplicate count and one
  offending value.
- Apply the same 99% test to the `primary_keys=` override and to direct
  assignment, so all three routes agree. Below the threshold, refuse — an
  override is an assertion, and the user should hear that it is false.
- Consider refusing outright at materialize time when the entity table's key
  has duplicates *among the requested indices*, since that is the case where a
  specific answer is specifically wrong.

Tests to add: an exactly-unique key is silent; one duplicate in a thousand
warns and records the count; an override or assignment below the threshold is
refused naming the column and an offending value; the Northwind fixture with a
duplicated customer is either refused or reported.
