# Limits

The endpoint enforces several size limits. `materialize()` checks them before
sending, then refuses or samples while naming the limit. Otherwise, a rejected
request returns a 500 or 422 that names neither the limit nor how to meet it.

`Request(context=..., predict=...)` receives exact frames and does not trim
them. It refuses a context above the model's cap instead of discarding named
rows, and reports the count and cap.

`materialize()` keeps the context under each model's cap. On `kumo-tabular`, it
samples to the cap and reports the row count. On `kumo-relational`, it gathers
the context. It refuses `num_context_samples` above the cap. Omitting the
argument gathers and reports the 1,000-row budget; it never samples down to
10,000.

## Sizes

Both endpoints enforce sizes, and most of these are the relational one's.

| Limit | Model | Value | Checked |
|---|---|---|---|
| context rows | relational | 1,000 by default, up to 10,000 with `num_context_samples` | at `materialize()` |
| context rows | tabular | 10,000 | sampled at `materialize()`, refused before sending on a direct `Request` |
| rows scored per call | relational | 10,000, or 200 for link prediction | more are gathered as several calls at `materialize()` |
| entities one question gathers | relational | 100,000, in at most 100 calls | at `materialize()`, and this client's own: the endpoint is never asked for more than one call's worth |
| calls a question asked in parts makes | relational | 200, one per candidate or step | at `materialize()`, as `sdm_client.decompose.MOST_PARTS` |
| request body | relational | 30 MiB per call | at `materialize()` |
| request body | relational, through Databricks model serving | 16 MB per call, the gateway's own decimal limit | before sending, on the wrapped envelope |
| an explanation | relational | exactly one entity | before sending |
| an explanation | relational | only within the 1,000-row context budget | before sending |
| hops | relational | 6 | at `materialize()` |
| tables | relational | 15 | against the graph the request carries, not the schema it was built from |

A relational request for more entities than one call scores is not refused.
`materialize()` gathers it in batches of at most that many, each with a
context of its own, sampled around that batch's rows rather than copied from
the first, and `predict()` sends one call per batch and returns one result.
Its `row_index` counts across the whole request rather than from zero per
call, so `to_pandas()`, which orders by it, gives the indices back in the
order they were asked for. A session scores those batches against the context
it pinned, so the context is uploaded once rather than once per call. `fit()`
itself takes a request of one call, since a session pins one context: fit a
small request and score the batched one against the handle.

**The 10,000 is the deployment's, not this client's.** It is the serving
layer's `TFM_MAX_TEST_ROWS`, whose default it is, so a deployment may hold to
less. Measured at the boundary rather than read from a document: 10,000 rows
to score are answered and 10,001 come back as

```
422 UNSUPPORTED_SHAPE N_test 10001 exceeds limit 10000
```

The earlier `kumo-relational` container holds to 1,000 instead, through a
model-specific admission check of its own, so a request gathered at this
default is refused there on its first call.

Pass `rows_per_call` to `materialize()` to ask in smaller batches than that.
It only ever goes down, since a larger call buys a 422 rather than a bigger
call. Two reasons to reach for it: a graph wide enough that a full batch
outruns the 30 MiB body, and a deployment whose cap is lower than the one
above.

It is not the answer to every 30 MiB refusal. A batch carries its context as
well as its rows, so where the context is itself over the limit, no batch is
small enough: a wide graph at 10,000 context rows was refused at every batch
size tried, and the same graph at the default context was gathered as 15
calls of 200. `num_context_samples`, `num_neighbors` and the columns the graph
carries are what come down in the first case.

**That 100,000 is this client's, not the endpoint's.** The endpoint is never
asked for more than one call holds. Every batch is built before the first is
sent, each with its own context, so the whole population is held in memory at
once, and 100,000 entities is about a gigabyte of it. One `materialize()`
gathers at most that many, in at most 100 calls, and the refusal names
whichever of the two it met. The second is what a small `rows_per_call`
reaches, and what a temporal link prediction reaches at 20,001 entities, since
it scores 200 a call.

Both govern the batches of one question, and neither counts the parts of a
question asked in several. A ranking is one call per candidate and a forecast
one per step, and what bounds those is `sdm_client.decompose.MOST_PARTS`, 200,
which is why a 101-candidate ranking makes 101 calls without meeting the 100
above. What the two ceilings hold for such a question is the population each
part gathers, counted across all of them.

A call that fails ends the request. The answers already paid for are carried
on the error as `details['answered']`, and the rows they cover as
`details['rows_answered']`, so the rest can be asked for again without buying
the whole population twice. A forecast keeps its answered steps the same way,
under `details['answered']`, numbered by step, with how many of them under
`details['parts_answered']`; a ranking keeps only the count, because an order
over some of its candidates is a different answer rather than part of the one
that was asked for.

**Batch size is chosen from the row count, not from the bytes.** The 30 MiB
body is checked per batch at `materialize()`, before anything is sent.
Databricks model serving weighs its own wrapped envelope against a smaller
16 MB, and only at send time, so a wide graph can pass `materialize()` and be
refused on the first call. Ask in smaller calls there, with `rows_per_call`,
unless the context alone is over the gateway's limit, which a single-entity
request can be: then it is `num_context_samples`, `num_neighbors` or the
columns the graph carries, as it is for the 30 MiB body.

**A question asked in parts is not batched.** A ranking is asked as one call
per candidate and a forecast as one per step, and `MOST_PARTS` counts those
calls, so a batched part would send the parts times the batches: a ranking
over 200 candidates and 50,000 entities would pass as 200 calls and make
1,000. More entities than one call scores is refused for both, before
anything is gathered. A ranking could not be batched in any case, since a
candidate is reported unscored for the whole ranking rather than per batch, so
one batch whose context held a single class would drop it from every row.

The tabular cap was measured at the boundary rather than read from a document:
10,000 context rows are answered and 10,001 come back as
`422 UNSUPPORTED_SHAPE N_train 10001 exceeds limit 10000`.

## Aggregation windows open after the anchor and close on their last day

A window written `0, 30, days` covers `(anchor, anchor + 30 days]`. A row at
the anchor itself does not count; one a day later does; one exactly thirty days
later does; one thirty-one days later does not.

This behavior was measured against the engine's labels, not read from the
grammar. The test pinned the context anchor and placed one row at each offset
in turn. Results were read per anchor because the engine samples several;
pooling them answers a different question.

## There is no `run_mode`

The endpoint takes one, and this client derives it. It distinguishes only a
small context budget from a large one, so the four-value preset was a two-way
choice wearing a disguise: `debug` silently gave you `fast`, and `normal`
silently gave you `best`. Set the number you want instead:

```python
query.materialize(indices=[...], num_context_samples=500)
```

**The number is honoured, not rounded up to a bucket.** Asking for 500 gathers
500 labelled examples, although the admitting budget holds 1,000. A count that
only selected a bucket would be as decorative as the two ineffective preset
values.

Omit it on a relational request to use the smaller budget, up to 1,000
examples. This is the endpoint's long-standing default. Omit it on a
single-table `materialize()` with either `indices=` or `predict=` to use every
labelled row, up to the endpoint's 10,000-row limit. Above that, it samples
10,000 rows and reports doing so. Use the argument to request more.

Ask for more than exists and you get what exists; there are only so many
labelled examples, and that is not an error. Ask for more than 10,000 and you
are told the ceiling rather than silently given fewer. Zero, a negative, a
fraction or a string are refused by name.

**Which examples you get** when the count narrows: the first `n` of what was
gathered, and for a temporal question that is **not a sample across time**.

The engine gathers a block of entities per anchor date, newest anchor first,
and narrowing slices from the front. So the narrowed context is the most recent
anchors, in full, and the older ones are absent entirely. Measured on a
two-year history gathered over seventeen monthly anchors:
`num_context_samples=50` returned fifty rows carrying a single anchor date, the
newest; 200 returned the newest four anchors. Neither changed with
`random_seed`. A forecast narrows the same way, each of its steps around its
own anchor.

Entities *are* sampled, within each anchor block, and that part does follow
`random_seed`. It is the anchor dates that are not shuffled, so narrowing
trades history for recency rather than thinning the whole span. Recency is a
defensible default for a temporal model; plan for a recency window rather than
for a representative sample, and gather more context if you need the older
anchors.

## What a deployment serves

`capabilities()` describes **this client** and reports `source='sdk'`. It is
not read from the endpoint.

`models()` *is* read from the endpoint, so it is the endpoint's own answer. The
route it uses is supplied by the NIM runtime rather than by the Universal TFM
contract, so an endpoint that does not serve it gives a clear error rather than
a wrong answer.

There is more than one build of `kumo-relational`. The client declares the
broader build with six task kinds and seven output fields. A narrower build
serves only classification and regression, refusing the rest itself. The
contract has no capabilities route, so the client declares the broader set
and lets the narrower build reject unsupported requests.
