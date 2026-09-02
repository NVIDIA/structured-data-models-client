# 8. Two endpoint limits are discovered only by being rejected, and one does not name the column

**Severity: medium for the class ceiling, arguable for the semantic type** —
correct behaviour either way, but it is what a user meets in their first hour
and the message does not tell them where to look.

The two halves have different standing, so they are separated below. The class
ceiling is squarely inside what `RuntimeContract` already promises. The
semantic type is a judgement call about that class's scope.

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

## The class ceiling is in scope, by the class's own docstring

`RuntimeContract.limits` is documented as:

> Sizes the endpoint enforces, so the client can refuse locally rather than eat
> a 500.

*"this model classifies at most 10 classes"* is a size the endpoint enforces
and the client currently eats a 422 for. That is the declared purpose, unmet.

## The semantic type is the arguable half

`RuntimeContract`'s fields are `tasks`, `outputs`, `explanation_formats`,
`sessions`, `limits` and `inference_options`. A semantic type is not a size,
and one could reasonably say the class describes shapes rather than column
meanings, and that a dtype refusal is the endpoint's to make.

The counter-argument is what a user actually experiences. Pointing the SDK at
your own tables is the intended first thing to do, and any table with a
free-text column — a name, an address, a description — fails with a 422 that
names a semantic type the user did not know their data had and gives them no
way to find it. Northwind produces fifteen such columns and the endpoint names
none of them.

Note that this is also how an ordinary categorical column becomes unusable.
`_stype_of` marks any string column with more than 64 distinct values as
`text`, including one whose pandas dtype is already `Categorical` — which the
relational engine does too, so that is a consistent design choice rather than a
defect. But it means US states, ISO country codes and SKUs all arrive here, and
this refusal is where the user meets the consequence.

**Minimum fix, if the semantic-type half is judged out of scope:** name the
columns. `_preflight` can read the stypes out of the encoded payload's schema
and add *"the columns typed 'text' are Customers.CompanyName, …"* to the error
it re-raises, without `RuntimeContract` growing a field.

## Proposed fix

- Add the class ceiling to `limits` and check the context's distinct target
  count against it in `_preflight`, so the number is named in this SDK's
  vocabulary rather than only in a 422. This one is not optional; it is what
  `limits` is for.
- Then decide the semantic-type half. Either add
  `semantic_types: frozenset[str]` to `RuntimeContract` and refuse in
  `_preflight`, listing the offending `table.column` names and how to change
  them (`schema['Customers']['Address'].stype = 'categorical'`); or leave the
  refusal to the endpoint and enrich its message with the column names, which
  the payload already carries.
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
