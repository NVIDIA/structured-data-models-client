# Limits

The endpoint enforces several sizes. Where this client can tell in advance, it
refuses first and names the limit — a rejected request otherwise comes back as
a 500 or a 422 that names neither the limit nor how to get under it.

## Sizes

| Limit | Value | Checked |
|---|---|---|
| context rows | 1,000 when explaining, otherwise 10,000 | at `materialize()` |
| rows scored per call | 1,000, or 200 for link prediction | at `materialize()` |
| an explanation | exactly one entity | before sending |
| an explanation | only within the 1,000-row context budget | before sending |
| hops | 6 | at `materialize()` |
| tables | 15 | at `materialize()` |

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

Omit it and you get the smaller budget — up to 1,000 examples — which is the
default the endpoint has always had. Ask for more and say so; that is what the
argument is for.

Ask for more than exists and you get what exists; there are only so many
labelled examples, and that is not an error. Ask for more than 10,000 and you
are told the ceiling rather than silently given fewer. Zero, a negative, a
fraction or a string are refused by name.

**Which examples you get** when the count narrows: the first `n` of what was
gathered. For most tasks the candidates are shuffled first, so that is a
sample. For forecasting the engine orders context chronologically before
slicing, so a narrowed forecast keeps the **oldest** `n` examples, not the most
recent. That is the endpoint's own long-standing behaviour when it trims to its
ceiling, and it is inherited rather than chosen — worth knowing before you
narrow a forecast.

## What a deployment serves

`capabilities()` describes **this client**, and says so — `source='sdk'`. It is
not read from the endpoint.

`models()` *is* read from the endpoint, so it is the endpoint's own answer. The
route it uses is supplied by the NIM runtime rather than by the Universal TFM
contract, so an endpoint that does not serve it gives a clear error rather than
a wrong answer.

There is more than one build of `kumo-relational`. The client declares the
broader one — six task kinds, seven output fields. A `structured-data-models`
backed build serves only classification and regression, and refuses the rest
itself. The contract has no capabilities route to ask with, so declaring the
broader set and letting the narrower build say no is the lesser of two wrongs.
