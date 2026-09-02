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
small context budget from a large one, so a four-value preset was a two-way
choice wearing a disguise. Set the number you want:

```python
query.materialize(indices=[...], num_context_samples=5000)
```

Ask for more than 10,000 and you are told the ceiling rather than being
silently given fewer.

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
