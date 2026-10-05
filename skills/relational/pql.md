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
multilabel question. Written the two ways that get as far as the task, over a
window and as a column declared multicategorical, it is refused for being one:

```
Foundation Model queries do not support multilabel tasks. Use `LIST_DISTINCT`
on a foreign key column to make a link prediction query. To make predictions
with this query, train a new model.
Foundation Model queries do not support static multilabel classification. To
make predictions with this query, train a new model.
```

Written other ways it is refused earlier, by a rule that has nothing to do
with `CLASSIFY` — an aggregation the model does not answer, or a target with
no window. Either way there is no query it completes.

So the two clauses worth knowing are `RANK TOP k` and `FORECAST n TIMEFRAMES`,
and each is narrower than it looks:

| Clause | Least | Most |
|---|---|---|
| `RANK TOP k` | `1` | `20` |
| `FORECAST n TIMEFRAMES` | `1` | `10000` |

`k` is how many candidates come back ranked, not how many are considered: you
may rank the top 2 of two hundred. **What costs is the fan-out.** A ranking is
one call per *candidate* and a forecast one call per *step*, and the client
refuses past 200 of them rather than spend the afternoon:

```
201 candidates means 201 calls, one for each, because the model has no task
that answers them together. That is past the 200 this will do without being
told to; ask about fewer, or raise sdm_client.decompose.MOST_PARTS if the wait
is what you want.
```

So in practice a forecast runs to 200 steps and a ranking over 200 candidates,
whatever `k` is, unless you say otherwise.

Past the top of either range the refusal names the bound:

```
Top k 21 exceeds maximum foundation model supported value of 20. To list more
candidates, train a new model.
FORECAST 10001 TIMEFRAMES exceeds the maximum of 10000.
```

Each clause also takes only one shape of target. A forecast asks the same
quantity again and again, so neither a plain column nor a comparison can be
forecast:

```
FORECAST is only supported for numerical targets.
FORECAST can only be used with targets that have a forward-looking time range
```

A ranking asks which candidates an entity reaches, so its target is a
`LIST_DISTINCT` over a foreign key and nothing else:

```
Target 'SUM(orders.price, 0, 30, days)' is not multicategorical. Either drop
the problem type RANK OR change the target to a multicategorical column
```

Every query names columns as `table.column`, and `table.*` for the rows of a
table rather than a column of it. A keyword may be written in upper case or in
lower case, and in nothing in between: `PREDICT` and `predict` both parse,
`Predict` does not, and nor does a time unit spelled `Days`. The examples here
use upper case for the language and lower case for your names.

## The target decides the task

Each line below was resolved against a schema and the resulting
`request.task.kind` read back:

| Target | Task | Calls |
|---|---|---|
| `COUNT(orders.*, 0, 30, days)` | `regression` | `1` |
| `SUM(orders.price, 0, 30, days)` | `regression` | `1` |
| `COUNT(orders.*, 0, 30, days) > 2` | `binary_classification` | `1` |
| `users.segment` | `multiclass_classification` | `1` |
| `COUNT(orders.*, 0, 30, days) FORECAST 3 TIMEFRAMES` | `regression` | `3` |
| `LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2` | `binary_classification` | `4` |

`users.segment` holds three values, which is what makes it a multiclass
question rather than a binary one.

The last two are worth reading twice. A forecast and a ranking are not single
calls with their own task kind: a forecast is asked as one regression per step,
and a ranking as one classification per candidate — three steps and four
candidates in the rows above. `request.parts` holds those calls and is `None`
when there is only one, while `request.context_rows` and `request.predict_rows`
count across all of them.

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
```

`start` and `end` are a window measured from the moment being asked about, and
`unit` is one of `days`, `weeks`, `months`, `hours`, `minutes`, `seconds`.

**The window is not optional.** The language also allows `AGGR(column)`, and
the model will not answer it:

```
TARGET aggregation SUM(orders.price) is missing a time range. A foundation
model has no static aggregation
```

What a target may be without a window is a column, as `users.segment` is, or
a condition over one: `PREDICT users.segment = 'a' FOR EACH users.user_id` is a
binary classification and needs no window. It is aggregating that needs one.

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
language most likely to be read wrongly, so it is worth being blunt.

`indices` names the entities to score, by key. One of `indices` or `predict` is
always required — there is no all-rows default, because it would silently score
a whole table — and `predict=` takes a frame of rows the tables do not hold,
which only a single-table schema supports.

Two clauses look like they narrow that set, and neither does:

- `FOR users.user_id IN (1, 2)` is **overridden by `indices`**, without a
  warning. Asked with `indices=[3]`, entity 3 is what comes back.
- `FOR EACH users.user_id WHERE users.segment = 'a'` selects the **labelled
  examples the model learns from**, not the entities scored.

The client says so when you combine the second of those with `indices`:

```
the entity filter "users.segment = 'a'" selects the labelled examples the
model learns from, and does not narrow the 1 rows named for scoring. Those are
scored as given, whether or not they satisfy it. Pass only indices that
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
| Answered on any value | `=`, `!=`, `IS`, `IS NOT`, `IN` |
| Answered on numbers and times only | `<`, `<=`, `>`, `>=` |
| Parsed, then refused | `LIKE`, `NOT LIKE`, `CONTAINS`, `NOT CONTAINS`, `STARTS WITH`, `ENDS WITH`, `IS IN` |

**The rows above are about a string the model reads as a label.** A string
column with more than 64 distinct values is read as `text` instead, and then
no comparison works on it at all, not even `=`:

```
Operations on a text column are not supported in the foundation model queries
yet.
```

The model does not read a `text` column as a feature either, so a free-text
column is along for the ride whichever way you come at it. Give the model the
label you would have searched for, as its own column.

**A string the model does read as a label can only be matched, never ordered
or searched.** The five in the first row are the whole of it, and all five ask
the same kind of question: is this value one of these. `IS` and `IS NOT` take
a value as readily as `NULL`. Ordering a string is refused, and so is looking
inside one:

```
Operation < between strings is not supported in the foundation model queries.
Operation CONTAINS between strings is not supported in the foundation model
queries.
```

So match a string against the values it actually takes, rather than a prefix,
a substring, or a range of them. `IS IN` is the old spelling of `IN` and says
so:

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
