# Limits

The endpoint enforces several sizes. `materialize()` knows them: it checks
before anything is sent, and refuses or samples while naming the limit — a
rejected request otherwise comes back as a 500 or a 422 that names neither the
limit nor how to get under it.

`Request(context=..., predict=...)` differs in one way: it is handed the frames
to use and does not trim them. Discarding rows that were named row by row would
be the worse of the two answers, so a context past the model's cap is refused
rather than sampled, naming the count and the cap.

`materialize()` keeps the context under the cap, by a different route on each
model. On `kumo-tabular` it samples to the cap and says how many rows it took.
On `kumo-relational` the context is gathered rather than handed over, so
`num_context_samples` past the cap is refused outright and asking for none
gathers the 1,000-row budget and says so; it never samples down to 10,000.

## Sizes

Both endpoints enforce sizes, and most of these are the relational one's.

| Limit | Model | Value | Checked |
|---|---|---|---|
| context rows | relational | 1,000 by default, up to 10,000 with `num_context_samples` | at `materialize()` |
| context rows | tabular | 10,000 | sampled at `materialize()`, refused before sending on a direct `Request` |
| rows scored per call | relational | 1,000, or 200 for link prediction | at `materialize()` |
| an explanation | relational | exactly one entity | before sending |
| an explanation | relational | only within the 1,000-row context budget | before sending |
| hops | relational | 6 | at `materialize()` |
| tables | relational | 15 | against the graph the request carries, not the schema it was built from |

The tabular cap was measured at the boundary rather than read from a document:
10,000 context rows are answered and 10,001 come back as
`422 UNSUPPORTED_SHAPE N_train 10001 exceeds limit 10000`.

## Aggregation windows open after the anchor and close on their last day

A window written `0, 30, days` covers `(anchor, anchor + 30 days]`. A row at the
anchor itself does not count; one a day later does; one exactly thirty days
later does; one thirty-one days later does not.

Measured against the engine's own labels rather than read off the grammar, with
the context anchor pinned and a single row placed at each offset in turn — and
read per anchor, because the engine samples several and pooling them answers a
different question.

## There is no `run_mode`

The endpoint takes one, and this client derives it. It distinguishes only a
small context budget from a large one, so the four-value preset was a two-way
choice wearing a disguise: `debug` silently gave you `fast`, and `normal`
silently gave you `best`. Set the number you want instead:

```python
query.materialize(indices=[...], num_context_samples=500)
```

**The number is honoured, not rounded up to a bucket.** Ask for 500 and 500
labelled examples are gathered, even though the budget that admits them holds
1,000 — a count that only chose a bucket would be as decorative as the two
preset values that did nothing.

Omit it on a relational request and you get the smaller budget — up to 1,000
examples — which is the default the endpoint has always had. Omit it on a
single-table `materialize()`, either `indices=` or `predict=`, and you get every
labelled row, up to the 10,000 the endpoint accepts; past that a sample of
10,000 is taken and you are told. Ask for more and say so; that is what the
argument is for.

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
`random_seed`. A forecast narrows the same way, each of its steps around its own
anchor.

Entities *are* sampled, within each anchor block, and that part does follow
`random_seed`. It is the anchor dates that are not shuffled, so narrowing
trades history for recency rather than thinning the whole span. Recency is a
defensible default for a temporal model; plan for a recency window rather than
for a representative sample, and gather more context if you need the older
anchors.

## What a deployment serves

`capabilities()` describes **this client**, and says so — `source='sdk'`. It is
not read from the endpoint.

`models()` *is* read from the endpoint, so it is the endpoint's own answer. The
route it uses is supplied by the NIM runtime rather than by the Universal TFM
contract, so an endpoint that does not serve it gives a clear error rather than
a wrong answer.

There is more than one build of `kumo-relational`. The client declares the
broader one — six task kinds, seven output fields. A narrower build serves
only classification and regression, and refuses the rest itself. The contract has no capabilities route to ask with, so declaring the
broader set and letting the narrower build say no is the lesser of two wrongs.
