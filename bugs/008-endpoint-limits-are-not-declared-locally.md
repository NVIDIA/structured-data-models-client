# 8. One endpoint limit is not declared locally, and one refusal does not name the column it is about

**Severity: medium** — correct behaviour twice over, but it is what a user meets
in their first hour, and neither message tells them where to look.

## Problem

Two things the relational endpoint enforces reach the caller only as a 422:

```
422 UNSUPPORTED_FEATURE  this model classifies at most 10 classes, and the
                         context carries 17
422 UNSUPPORTED_DTYPE    semantic type 'text' is unsupported
```

They are two different defects, and the repository already says what to do
about each.

## The class ceiling belongs in `limits`

`RuntimeContract.limits` is documented as:

> Sizes the endpoint enforces, so the client can refuse locally rather than eat
> a 500.

A class ceiling is a size the endpoint enforces. It is absent, so the client
builds the request, sends it, and reads back a number it could have compared
locally. `limits` already carries `context_rows_fast`, `context_rows_max`,
`predict_rows_max`, `predict_rows_temporal_link_prediction`,
`explanation_predict_rows`, `hops_max` and `tables_max` — this is the same kind
of entry, missing.

## The `text` refusal is not a missing declaration — it is a missing column name

The endpoint owning the dtype rule is right, and this report does **not** ask
for a `semantic_types` field on `RuntimeContract`. The predecessor SDK settles
that: `kumo-relational-engine` declares `_MAX_PRED_SIZE`, `_MAX_TEST_SIZE`,
`_MAX_CONTEXT_SIZE` and `_MAX_SIZE` locally — **row-count and byte sizes and
nothing else** — and leaves semantic-type support to the server. Sizes are the
client's business; what a model can represent is not.

What is wrong is the message. `semantic type 'text' is unsupported` does not say
*which column*, and `Schema.from_data` marks any string column with more than
64 distinct values as `text`, so a real dataset acquires them in bulk. Northwind
produces fifteen across four tables and the endpoint names none:

```
Customers.CompanyName, ContactName, Address, City, PostalCode, Phone, Fax
Orders.RequiredDate, ShippedDate, ShipName, ShipAddress, ShipCity, ShipPostalCode
Products.ProductName, QuantityPerUnit
```

Setting those to `categorical` makes the request work, so the fix a user needs
is one line — once they know which line.

## The repo already has a name for this fix

`sdm_relational_engine/rfm/rfm.py:453`:

```python
def _cardinality_guidance(detail: str, payload: dict | None) -> str:
    r"""Explain how to resolve a categorical-cardinality rejection.

    The NIM reports the offending count and limit but not the column, so the
    columns are recovered from the request that was just refused and the fix is
    spelled out against the worst offender.
    """
```

That is the identical situation — a NIM naming a limit but not a column — and
it is already solved, with a helper (`high_cardinality_columns(payload, limit)`)
that reads the offending columns out of the payload that was just refused. The
`text` refusal simply was not given the same treatment.

## Location

`packages/sdm-client/src/sdm_client/base.py:52` — `RuntimeContract`, whose
`limits` has no class-count entry.

`packages/sdm-client/src/sdm_client/adapters/relational.py` — `_preflight`,
which checks the explanation limits and the inference options. Its signature is
`_preflight(self, payloads, outputs)`: it receives encoded payloads, so the
stypes are reachable at `payload['schema'][table]['columns']`.

`packages/sdm-relational-engine/src/sdm_relational_engine/rfm/rfm.py:453` —
`_cardinality_guidance`, the pattern to copy.

## Why it needs fixing

Neither is a wrong answer, which is why this is medium and not high. Both are
first-hour experience: pointing the SDK at your own tables is the intended first
thing to do, and a table with a name or an address column fails with a 422 that
names a semantic type the user did not know their data had, and gives them no
way to find it.

## Proposed fix

Two changes, both small, and neither adds a field to `RuntimeContract`.

- Add the class ceiling to `RuntimeContract.limits` and compare the context's
  distinct target count against it in `_preflight`, so the number is named in
  this SDK's vocabulary before a request is sent.
- Enrich the `UNSUPPORTED_DTYPE` refusal the way `_cardinality_guidance`
  enriches the cardinality one: on a 422 naming a semantic type, read the
  columns carrying that stype out of the payload just refused and list them,
  with the one-line fix spelled out —
  `schema['Customers']['Address'].stype = 'categorical'`.

Tests to add: a target with more classes than the ceiling is refused locally,
naming both counts, without reaching the transport; a `text` column produces an
error naming the table-qualified columns responsible; a request with no `text`
columns is unaffected.
