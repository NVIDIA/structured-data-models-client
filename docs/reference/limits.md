# Limits

The client does not enforce the endpoint's size limits: context rows, rows scored per call, request bytes, tables and hops.
A deployment that raises them is not held back, and a request past them is refused by the endpoint with its own message.

## Defaults

| Setting | Model | Default | Larger values |
|---|---|---|---|
| context rows | relational | 1,000 | as many as `num_context_samples` asks for |
| context rows | tabular | sampled to 10,000, with a warning | as many as `num_context_samples` asks for; a direct `Request` is sent as given |
| rows scored per call | relational | 10,000 | as many as `rows_per_call` asks for |
| hops | relational | 2, or as `num_hops` / `num_neighbors` ask | as many as asked; a walk ends at the first hop that reaches no new row |

A relational request for more entities than one call scores is gathered as several calls, each with a context of its own, and `predict()` sends one call per batch and returns one result.
Its `row_index` counts across the whole request, so `to_pandas()` gives the rows back in the order they were asked for.
A session scores those batches against the context it pinned, so the context is uploaded once.
`fit()` takes a request of one call, since a session pins one context.

The client checks these:

| Check | Value |
|---|---|
| calls a question asked in parts makes (`sdm_client.decompose.MOST_PARTS`) | 200, one per candidate or step |
| `top_k` | at most 20 |
| a response body | at most 64 MiB |
| a call through Databricks model serving | 16 MB on the wrapped envelope, the gateway's own limit, checked before sending |

A call that fails ends the request.
The answers already paid for are carried on the error as `details['answered']`, and the rows they cover as `details['rows_answered']`, so the rest can be asked for again.
A forecast keeps its answered steps the same way, under `details['answered']`, numbered by step, with how many under `details['parts_answered']`.
A ranking keeps only the count, because an order over some of its candidates is a different answer.

**A question asked in parts is not batched.** A ranking is asked as one call per candidate and a forecast as one per step, and `MOST_PARTS` counts those calls.
More entities than one call scores is refused for both, before anything is gathered.

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
labelled row, up to 10,000. Above that, it samples 10,000 rows and reports
doing so. Use the argument to request more.

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
