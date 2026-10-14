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
| request body | relational | 30 MiB per call | at `materialize()` |
| an explanation | relational | exactly one entity | before sending |
| an explanation | relational | only within the 1,000-row context budget | before sending |
| hops | relational | 6 | at `materialize()` |
| tables | relational | 15 | against the graph the request carries, not the schema it was built from |

A relational request for more entities than one call scores is not refused.
`materialize()` gathers it in batches of at most that many, each carrying its
own copy of the context, and `predict()` sends one call per batch and returns
one result. Its `row_index` counts across the whole request, in the order the
indices were given. A session scores those batches against the context it
pinned, so the context is uploaded once rather than once per call.

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
