# Limits

The client holds a request to no size of its own.
It sends its best attempt and recovers from what the endpoint says:

| The endpoint answers | The client |
|---|---|
| `413`, Databricks' `400` saying "Request size cannot exceed", an answer past 64 MiB, a `504`, or the call times out | asks the call's rows again in two halves at once, each with the whole context, and keeps that smaller size for the request's later calls; a batch is halved at most 3 times, to an eighth of its rows, and then refused naming the call's rows, bytes and status |
| `503` or `429` | waits as `Retry-After` says and tries again at the same size |
| a connection error, a `500` or a `502` | tries again at the same size with backoff, up to `max_retries` (3 by default), and never splits; through Databricks model serving, a `500` or `502` is tried once more |
| `422`, or a refusal of a call of one row | does not try again, and shows the endpoint's message |

What a call too large or too slow comes to depends on what it carries:

| Call | Too large or too slow |
|---|---|
| a prediction, context and rows to score | its rows are halved, at most 3 times, each smaller call with the whole context; then refused, naming the rows, bytes and status |
| a session's create (`fit()`, or an automatic batch session), the context alone | refused at once: the context is too large for the endpoint; send fewer context rows or columns |
| a session's scoring call, rows to score alone | its rows are halved, at most 3 times |

A temporary failure is tried again at the same size on every call; none is split.
The context is never split.
The context is sampled only to a `num_context_samples` you pass.
`timeout` and `max_retries` are optional settings of the client.

Through Databricks model serving, an oversized call is dropped as an opaque connection error, so `predict()` splits a call whose wrapped envelope is past the endpoint's `max_request_bytes` (16 MB, the gateway's own decimal limit) before sending it.
A call of one row past it is refused, since only a smaller context helps.
There, a deadline spent waiting on a `429`, a `503` or a connection is reported as a `503` and not split; a `504` is a call that timed out.

## Batches

A relational request is gathered in batches of 10,000 rows to score, or 200 for a link prediction: a default, not a cap.
Pass `rows_per_call` to `materialize()` to ask in batches of another size; it is taken as asked.
A single-table request is scored in calls of 10,000 rows too, each with the whole context, and the answers are put back together as one.
Each batch has a context of its own, sampled around that batch's rows rather than copied from the first, and `predict()` sends one call per batch and returns one result.
Its `row_index` counts across the whole request rather than from zero per call, so `to_pandas()`, which orders by it, gives the indices back in the order they were asked for.
A session scores those batches against the context it pinned, so the context is uploaded once rather than once per call.
`fit()` itself takes a request of one call, since a session pins one context: fit a small request and score the batched one against the handle.

Every batch is built before the first is sent, each with its own context, so the whole population is held in memory at once; the client sets no most on the entities or the calls.
A question asked in parts is different: a ranking is one call per candidate and a forecast one per step, bounded by `sdm_client.decompose.MOST_PARTS`, 200, a limit on the wait rather than on the size.
A question asked in parts is not batched by rows, since a batched part would send the parts times the batches.

A call that fails ends the request.
The answers already paid for, the halves of a split call included, are carried on the error as `details['answered']`, and the rows they cover as `details['rows_answered']`, so the rest can be asked for again without buying the whole population twice.
A forecast keeps its answered steps the same way, under `details['answered']`, numbered by step, with how many of them under `details['parts_answered']`; a ranking keeps only the count, because an order over some of its candidates is a different answer rather than part of the one that was asked for.

The relational model reads as many tables and hops as a request carries.
A walk that reaches no new row ends there, so a hop count past the data's depth costs nothing.

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
labelled row.

Ask for more than exists and you get what exists; there are only so many
labelled examples, and that is not an error. Zero, a negative, a fraction or a
string are refused by name.

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

Snowflake samples rows without a warehouse seed.
`random_seed` is optional, and setting it does not make Snowflake's warehouse row sample reproducible.
Repeated predictions can therefore differ even with the same seed.

## What a deployment serves

`capabilities()` describes **this client** and reports `source='sdk'`. It is
not read from the endpoint.

`models()` *is* read from the endpoint, so it is the endpoint's own answer. The
route it uses is supplied by the NIM runtime rather than by the Universal TFM
contract, so an endpoint that does not serve it gives a clear error rather than
a wrong answer. A client opened with `SDMClient.in_process()` has no endpoint,
and lists the models it can run.

There is more than one build of `kumo-relational`. The client declares the
broader build with six task kinds and seven output fields. A narrower build
serves only classification and regression, refusing the rest itself. The
contract has no capabilities route, so the client declares the broader set
and lets the narrower build reject unsupported requests.

## Explanation admission

`EXPLAIN PREDICT` augments the requested outputs; it does not make an empty `outputs=[]` valid.
A ranking with this prefix is refused at materialization before source gathering.
An explanation requested later through `predict(outputs=...)` is checked at prediction admission.
The optional Python model runtime checks its supported outputs before loading weights.
