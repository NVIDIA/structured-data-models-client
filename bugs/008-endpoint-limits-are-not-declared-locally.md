# 8. Two endpoint limits are discovered only by being rejected, and one does not name the column

**Severity: medium** — correct behaviour, but it is what a user meets in their
first hour, and the message does not tell them where to look.

## Problem

`RuntimeContract` exists so the client can refuse what a deployment cannot
serve, before spending a round trip, naming what is wrong. It declares tasks,
outputs, explanation formats, session support, inference options and seven row
and shape limits. Two further limits are enforced by the relational endpoint
and are in none of them:

```
422 UNSUPPORTED_DTYPE    semantic type 'text' is unsupported
422 UNSUPPORTED_FEATURE  this model classifies at most 10 classes, and the
                         context carries 17
```

The second message is good: it names both numbers, and a user can act on it.

The first is not, because it does not say **which column**. `Schema.from_data`
assigns `text` to any string column with more than 64 distinct values, so a
real dataset acquires them silently and in bulk — Northwind produces fifteen
across four tables, and the endpoint names none of them. The user is told a
semantic type is unsupported and left to find the columns themselves.

The client has the schema in hand and knows exactly which columns are `text`.

## Reproduced

Northwind (`Customers`, `Orders`, `Order Details`, `Products`) inferred with
`Schema.from_data`, then any relational query:

```
NimRequestError: [422 UNSUPPORTED_DTYPE] semantic type 'text' is unsupported
```

The fifteen columns responsible, which the client could have listed:

```
Customers.CompanyName, ContactName, Address, City, PostalCode, Phone, Fax
Orders.RequiredDate, ShippedDate, ShipName, ShipAddress, ShipCity, ShipPostalCode
Products.ProductName, QuantityPerUnit
```

Setting those to `categorical` makes the request work, so the fix a user needs
is one line — once they know which line.

## Location

`packages/sdm-client/src/sdm_client/base.py:52` — `RuntimeContract` is defined
here (`models.py` only instantiates it) with fields `tasks`, `outputs`,
`explanation_formats`, `sessions`, `limits`, `inference_options`. No field
describes supported semantic types, and `limits` has no class-count entry.

`packages/sdm-client/src/sdm_client/adapters/relational.py` — `_preflight`
checks the explanation limits and the inference options. Note its signature is
`_preflight(self, payloads, outputs)`: it receives encoded payloads, not a
`Schema`, so the fix has to read the stypes out of
`payload['schema'][table]['columns']` — or `_preflight` has to be given the
request.

## Why it needs fixing

This is not a wrong answer, and both refusals are honest. What makes it worse
than "expensive discovery" is when it happens: pointing the SDK at your own
tables is the intended first thing to do, and any table with a free-text column
— a name, an address, a description — fails with a 422 that names a semantic
type the user did not know their data had and gives them no way to find it.

It compounds with [12](012-an-explicit-categorical-becomes-text.md): a column
the user explicitly declared `Categorical` is re-typed `text` at 65 levels and
then rejected here, so the two together make an ordinary categorical column
unusable and explain neither half.

## Proposed fix

- Add `semantic_types: frozenset[str]` to `RuntimeContract` and declare what
  the relational deployment accepts (`numerical`, `categorical`, `ID`,
  `timestamp`). Refuse in `_preflight`, listing the offending
  `table.column` names and telling the caller how to change them:
  `schema['Customers']['Address'].stype = 'categorical'`.
- Add the class ceiling to `limits` and check the context's distinct target
  count against it in the same place, so the number is named in this SDK's
  vocabulary rather than only in a 422.
- Both are already the pattern the explanation limits follow
  (`explanation_predict_rows`, `context_rows_fast`), so this is filling in
  a table rather than adding a mechanism.

A caveat worth recording in the fix: these values were read off one build of
the relational NIM, and `/v1/models` returns only an id, so the client cannot
ask. `RuntimeContract` is already a pinned description for that reason, and
these two entries are pinned the same way.

Tests to add: a `text` column is refused locally, naming the column and the
table; a target with more classes than the ceiling is refused locally, naming
both counts; neither reaches the transport.
