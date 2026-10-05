# Writing the query

A predictive query is the question. The task is never an argument: it follows
from what the query asks for, and getting the query right is the whole job.

## The shape

```
PREDICT <target> [ RANK TOP k | CLASSIFY | FORECAST n TIMEFRAMES ]
(FOR EACH | FOR) <entity>
[ ASSUMING <condition> ]
```

`CLASSIFY` is in the language and **not reachable on these models**. It asks a
multilabel question, and the refusal says both why and what to ask instead:

```
Foundation Model queries do not support multilabel tasks. Use `LIST_DISTINCT`
on a foreign key column to make a link prediction query. To make predictions
with this query, train a new model.
```

So the two clauses worth knowing are `RANK TOP k` and `FORECAST n TIMEFRAMES`.

Every query names columns as `table.column`, and `table.*` for the rows of a
table rather than a column of it. A keyword may be written in upper case or in
lower case, and in nothing in between: `PREDICT` and `predict` both parse,
`Predict` does not, and nor does a time unit spelled `Days`. The examples here
use upper case for the language and lower case for your names.

## The target decides the task

Each line below was resolved against a schema and the resulting
`request.task.kind` read back:

| Target | Task | |
|---|---|---|
| `COUNT(orders.*, 0, 30, days)` | `regression` | |
| `SUM(orders.price, 0, 30, days)` | `regression` | |
| `COUNT(orders.*, 0, 30, days) > 2` | `binary_classification` | |
| `users.segment` | `multiclass_classification` | a column of three values |
| `COUNT(orders.*, 0, 30, days) FORECAST 3 TIMEFRAMES` | `regression` | once per step |
| `LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2` | `binary_classification` | once per candidate |

The last two are worth reading twice. A forecast and a ranking are not single
calls with their own task kind: a forecast is asked as one regression per step,
and a ranking as one classification per candidate. `request.parts` holds those
calls and is `None` when there is only one, while `request.context_rows` and
`request.predict_rows` count across all of them.

So a comparison makes a classification out of a quantity. `COUNT(...) > 2` asks
whether, not how many, and that is usually the question worth asking.

## Aggregations

The language has nine. **The foundation model answers six of them**, which is
the distinction that matters, because the other three parse and are then
refused:

| Aggregation | |
|---|---|
| Answered | `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `LIST_DISTINCT` |
| Parsed, then refused | `COUNT_DISTINCT`, `FIRST`, `LAST` |

```
PREDICT FIRST(orders.channel, 0, 30, days) FOR EACH users.user_id
→ Aggregation FIRST type is not supported in the foundation model queries.
→ Aggregations of categorical columns are not supported in the foundation
  model queries.
```

The second line of that refusal is a rule of its own: **aggregate a number, not
a category.** The one way to ask about a category is `LIST_DISTINCT` over a
foreign key, which is a ranking.

```
AGGR(column, start, end, unit)
AGGR(column)
```

`start` and `end` are a window measured from the moment being asked about, and
`unit` is one of `days`, `weeks`, `months`, `hours`, `minutes`, `seconds`.

**The window is the future, and both ends must be non-negative.** `0, 30, days`
is the next thirty days. A target reaching backwards is refused:

```
PREDICT COUNT(orders.*, -30, 0, days) FOR EACH users.user_id
→ TARGET cannot have an aggregation in the past. Both start and end need to be
  non negative.
```

That is the one rule that catches people, because the past is where the data
is. The past is what the model reads as context; the query asks about what has
not happened.

## Which entities get scored

**`materialize()` decides that, not the query.** This is the part of the
language most likely to be read wrongly, so it is worth being blunt:

```python
request = PredictiveQuery(query, schema).materialize(
    indices=[1, 2, 3], model=KumoRelational()
)
```

`indices` names the entities to score, by key. One of `indices` or `predict` is
always required — there is no all-rows default, because it would silently score
a whole table — and `predict=<frame>` scores rows the tables do not contain,
which only a single-table schema supports.

Two clauses look like they narrow that set, and neither does:

- `FOR users.user_id IN (1, 2)` is **overridden by `indices`**, without a
  warning. Asked with `indices=[3]`, entity 3 is what comes back.
- `FOR EACH users.user_id WHERE users.segment = 'a'` selects the **labelled
  examples the model learns from**, not the entities scored. The client says so
  when you combine the two:

  ```
  the entity filter "users.segment = 'a'" selects the labelled examples the
  model learns from, and does not narrow the 1 rows named for scoring. Those
  are scored as given, whether or not they satisfy it. Pass only indices that
  satisfy the condition if that is what you meant.
  ```

So to score a particular set of entities, pass that set as `indices`. Use the
entity filter to choose what the model learns from, which is a different
question and a useful one.

## Filtering what is counted

A `WHERE` inside an aggregation narrows the rows it reads, which is different
from narrowing the entities:

```
PREDICT COUNT(orders.* WHERE orders.price > 50, 0, 30, days) FOR EACH users.user_id
```

As with the aggregations, the language offers more comparisons than the model
answers:

| Comparison | |
|---|---|
| Answered | `=`, `!=`, `<`, `<=`, `>`, `>=`, `IS`, `IS NOT`, `IN` |
| Parsed, then refused | `LIKE`, `NOT LIKE`, `CONTAINS`, `NOT CONTAINS`, `STARTS WITH`, `ENDS WITH`, `IS IN` |

**There is no string matching.** Every operation that compares text is refused:

```
Operation CONTAINS between strings is not supported in the foundation model
queries.
```

Compare a string with `=` or `IN` against the values it actually takes.
`IS IN` is the old spelling of `IN` and says so:

```
Operator IS IN is deprecated. Please use operator `IN` instead.
```

Combine conditions with `AND`, `OR`, `NOT` and parentheses. A list is written
`(1, 2, 3)` and needs at least two elements.

## Asking a counterfactual

`ASSUMING` asks the question under a condition that is not true yet. The
condition must carry a time window of its own, because it is about the same
future the target is:

```
PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id
ASSUMING SUM(orders.price, 0, 30, days) > 100
```

A static condition is refused and says why:

```
ASSUMING users.segment = 'a'
→ ASSUMPTION condition users.segment = 'a' has no time range but a temporal
  aggregation is required.
```

## On kumo-tabular

The same language, cut down to what one table can answer: a column of that
table as the target, and nothing else. No aggregation across a relationship, no
window, no forecast, no ranking, no entity filter, no `ASSUMING`.

```
PREDICT customers.churn FOR EACH customers.id
```

A schema with a second table is refused before a request is built, and names
the model that can answer it:

```
kumo-tabular answers from a single table, and this schema has 2.
Pass model=KumoRelational() to gather them as a graph
```

## When it will not parse

A refusal is a `StructuredDataError` naming the row and column it stopped at.
Read the column number: it points at the token, not at the clause.

```
PREDICT COUNT(orders.*, 0, 30, days, 3) FOR EACH users.user_id
→ Invalid Syntax ... Could not process the text: ","; in line 1 and col 35
```

That one is a forecast written as a fifth argument. Forecasts are
`FORECAST n TIMEFRAMES`, after the target.
