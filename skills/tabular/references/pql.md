# Writing a query for kumo-tabular

The same language as [`../../relational/pql.md`](../../relational/pql.md), cut
down to what one table can answer. Read this one when the schema has a single
table; almost every clause the relational reference describes is refused here,
and the refusals say so.

## The whole of it

```
PREDICT <table>.<column> FOR EACH <table>.<key>
PREDICT <table>.<column> FOR <table>.<key> IN (...)
```

`FOR` needs a selector. `FOR customers.customer_id` on its own is refused:
*"When using "FOR" in foundation model queries, you must specify one or
multiple entities to make the prediction"*. `FOR EACH` is the one to reach
for.

The target is a column of the table, stored as it stands. The task follows from
that column's type, and `materialize(task=...)` settles it when whole numbers
could be either a label or a quantity.

```python
import pandas as pd
from sdm_client import KumoTabular, PredictiveQuery, Schema

customers = pd.DataFrame(
    {
        'customer_id': range(40),
        'spend': [10.0 + i for i in range(40)],
        'churn': [i % 2 for i in range(40)],
    }
)
schema = Schema.from_data(
    {'customers': customers}, primary_keys={'customers': 'customer_id'}
)

request = PredictiveQuery(
    'PREDICT customers.churn FOR EACH customers.customer_id', schema
).materialize(indices=[7, 12, 3], model=KumoTabular())
```

`FOR customers.customer_id IN (1, 2)` works too, and `indices` is still
required: it is what says which rows are scored.

## What the relational language has and this does not

| Clause | On one table |
|---|---|
| `COUNT`, `SUM`, any aggregation | refused |
| a window, `0, 30, days` | refused |
| a comparison target, `spend > 200` | refused |
| `RANK TOP k` | refused |
| `FORECAST n TIMEFRAMES` | refused |
| `WHERE` on the entity | refused |
| `ASSUMING` | refused |

Every one of them is the tabular profile of the language refusing, not the
grammar: the relational model takes all seven, within the bounds
[`../../relational/pql.md`](../../relational/pql.md) gives each of them.
Three are worth seeing here, because all three are queries somebody would
reasonably write.

A comparison is a target the relational model answers and this one will not,
and the refusal says what to do instead:

```
PREDICT customers.spend > 200 FOR EACH customers.customer_id
→ The target is computed from the table rather than stored in it, and a
  tabular model predicts a column as it stands. Add the derived value as a
  column of 'customers' and predict that column.
```

An aggregation is refused whatever the schema, because there is no schema that
would make it answerable here:

```
PREDICT COUNT(customers.*, 0, 30, days) FOR EACH customers.customer_id
→ The target carries a time range, and a tabular model predicts a value as it
  stands. No schema makes this answerable by a tabular model; ask for the
  column itself, or use a model that reads relationships.
```

Filtering the entities is refused for a different reason, and the fix is on
your side of the call:

```
PREDICT customers.churn FOR EACH customers.customer_id
WHERE customers.segment = 'a'
→ This query filters the entities it predicts for, and a tabular model scores
  the rows it is given. Filter the rows before constructing the query.
```

## A second table

Refused before a request is built, naming the model that can answer it:

```
kumo-tabular answers from a single table, and this schema has 2.
Pass model=KumoRelational() to gather them as a graph
```

If the question really is about one table, build the schema from that table
alone. If it is about an entity and the records linked to it, it belongs to
`kumo-relational` and [`../../relational/SKILL.md`](../../relational/SKILL.md).
