# Limits

The endpoint enforces several size limits.
`materialize()` checks them before sending, then refuses, samples or splits while naming the limit.
Otherwise, a rejected request returns a 500 or 422 that names neither the limit nor how to meet it.

`Request(context=..., predict=...)` receives exact frames and does not trim them.
It refuses a context above the model's cap instead of discarding named rows, and reports the count and cap.

`materialize()` keeps the context under each model's cap.
On `kumo-tabular`, it samples to the cap and reports the row count.
On `kumo-relational`, it gathers the context.
It refuses `num_context_samples` above the cap.
Omitting the argument gathers and reports the 1,000-row budget; it never samples down to 10,000.

## Sizes

Both endpoints enforce sizes, and most of these are the relational one's.

| Limit | Model | Value | Checked |
|---|---|---|---|
| context rows | relational | 1,000 by default, up to 10,000 with `num_context_samples` | at `materialize()` |
| context rows | tabular | 10,000 | sampled at `materialize()`, refused before sending on a direct `Request` |
| rows scored per call | relational | 10,000, or 200 for link prediction | more are gathered as several calls at `materialize()` |
| entities one question gathers | relational | 100,000, in at most 100 calls | at `materialize()`, and this client's own: the endpoint is never asked for more than one call's worth |
| calls a question asked in parts makes | relational | 200, one per candidate or step | at `materialize()`, as `sdm_client.decompose.MOST_PARTS` |
| request body | tabular or relational | 30 MiB per call | a call past it is split into smaller calls when it is predicted; a context past it on its own is refused at `materialize()` |
| request body | tabular or relational, through Databricks model serving | 16 MB per call, the gateway's own decimal limit | on the wrapped envelope, split as above |
| gradient explanations | relational and tabular | endpoint-dependent; no separate client row/context limit | endpoint validates support |

The relational model reads as many tables and hops as a request carries.
This client sets no most on either: a schema may describe any number of tables, and `num_hops` or `num_neighbors` may ask for any number of hops of at least one.
A walk that reaches no new row ends there, so a hop count past the data's depth costs nothing.

A client opened with `SDMClient.in_process()` is held to this table as well, since this client checks it.
A NIM also has limits of its own, which each deployment sets, such as the longest string a cell may hold or the most distinct values a column may have.
Those protect a shared server, and are not applied when the model runs in your own process.

## Settings

The context rows, the rows scored per call and the request body are the serving layer's defaults (`TFM_MAX_TRAIN_ROWS`, `TFM_MAX_TEST_ROWS` and `TFM_MAX_PAYLOAD_BYTES`), and a deployment may be configured to take more or less.
Each is a setting of this client with that default:

| Setting | Environment variable | Default | What it holds |
|---|---|---|---|
| `max_relational_context_rows` | `SDM_CLIENT_MAX_RELATIONAL_CONTEXT_ROWS` | 10,000 | the most context rows a relational request carries; `num_context_samples` asks for at most this many |
| `max_relational_predict_rows` | `SDM_CLIENT_MAX_RELATIONAL_PREDICT_ROWS` | 10,000 | the most rows one relational call scores; more are asked in several calls |
| `max_temporal_link_predict_rows` | `SDM_CLIENT_MAX_TEMPORAL_LINK_PREDICT_ROWS` | 200 | the most rows one temporal link prediction call scores |
| `max_tabular_context_rows` | `SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS` | 10,000 | the most context rows a single-table request carries; `materialize()` samples down to it |
| `max_request_bytes` | `SDM_CLIENT_MAX_REQUEST_BYTES` | 31,457,280 (30 MiB) | the most bytes one call's body takes |

Set one with an argument or with its environment variable.
An argument comes before the environment variable, which comes before the default.
In Python, pass `limits=` to the client and to `materialize()`:

```python
from sdm_client import Limits, SDMClient

limits = Limits(max_tabular_context_rows=200_000, max_request_bytes=512 * 1024 * 1024)
client = SDMClient(url, limits=limits)
request = query.materialize(indices=keys, limits=limits)
result = client.predict(request, model=KumoTabular())
```

`client.limits.resolved()` reports what each setting comes to.
In Java, pass `Limits.builder()...build()` to `ClientOptions.builder().limits(...)`, `DatabricksServing.limits(...)` and `MaterializeOptions.builder().limits(...)`.
Both read the same environment variables, and an environment variable that is not a whole number of at least 1 is refused with code `INVALID_CONFIGURATION`, naming it.
A refusal at one of these limits names the setting and its environment variable.

Raise a setting only to what the deployment is configured to take.
This client then holds requests to the raised limit, and the endpoint answers them; a setting above the deployment's limit only moves the refusal to the endpoint.

## Batches

A relational request for more entities than one call scores is not refused.
`materialize()` gathers it in batches of at most that many, each with a context of its own, sampled around that batch's rows rather than copied from the first, and `predict()` sends one call per batch and returns one result.
Its `row_index` counts across the whole request rather than from zero per call, so `to_pandas()`, which orders by it, gives the indices back in the order they were asked for.
A session scores those batches against the context it pinned, so the context is uploaded once rather than once per call.
`fit()` itself takes a request of one call, since a session pins one context: fit a small request and score the batched one against the handle.

**The 10,000 is the deployment's, not this client's.**
It is the serving layer's `TFM_MAX_TEST_ROWS`, whose default it is, so a deployment may hold to more or less.
Measured at the boundary rather than read from a document: 10,000 rows to score are answered and 10,001 come back as

```
422 UNSUPPORTED_SHAPE N_test 10001 exceeds limit 10000
```

The earlier `kumo-relational` container holds to 1,000 instead, through a model-specific admission check of its own, so a request gathered at this default is refused there on its first call.

Pass `rows_per_call` to `materialize()` to ask in smaller batches than that.
It only ever goes down, since a larger call buys a 422 rather than a bigger call.
Reach for it for a deployment whose cap is lower than the one above, or set `max_relational_predict_rows` to that cap.

**A call past the request body's limit is split by its bytes as well as its rows.**
When a call's body would pass `max_request_bytes`, or a Databricks serving endpoint's 16 MB once wrapped, `predict()` sends its rows as several smaller calls, halving until each fits, and returns one result numbered as before.
Every one of those calls carries the batch's whole context, which is not split.
Where the context alone, with one row to score, is past the limit, no call is small enough, and the request is refused with the context's size and the limit: a wide graph at 10,000 context rows can be.
`num_context_samples`, `num_neighbors` and the columns the graph carries are what come down then, or the limit goes up where the deployment takes more.
`materialize()` makes the same check before anything is sent.
A single-table request is sent as one call and is not split: past the limit, it is refused with its size, and the rows scored at once or the context come down.

**That 100,000 is this client's, not the endpoint's.**
The endpoint is never asked for more than one call holds.
Every batch is built before the first is sent, each with its own context, so the whole population is held in memory at once, and 100,000 entities is about a gigabyte of it.
One `materialize()` gathers at most that many, in at most 100 calls, and the refusal names whichever of the two it met.
The second is what a small `rows_per_call` reaches, and what a temporal link prediction reaches at 20,001 entities, since it scores 200 a call.

Both govern the batches of one question, and neither counts the parts of a question asked in several.
A ranking is one call per candidate and a forecast one per step, and what bounds those is `sdm_client.decompose.MOST_PARTS`, 200, which is why a 101-candidate ranking makes 101 calls without meeting the 100 above.
What the two ceilings hold for such a question is the population each part gathers, counted across all of them.

A call that fails ends the request.
The answers already paid for are carried on the error as `details['answered']`, and the rows they cover as `details['rows_answered']`, so the rest can be asked for again without buying the whole population twice.
A forecast keeps its answered steps the same way, under `details['answered']`, numbered by step, with how many of them under `details['parts_answered']`; a ranking keeps only the count, because an order over some of its candidates is a different answer rather than part of the one that was asked for.

**A question asked in parts is not batched by rows.**
A ranking is asked as one call per candidate and a forecast as one per step, and `MOST_PARTS` counts those calls, so a batched part would send the parts times the batches: a ranking over 200 candidates and 50,000 entities would pass as 200 calls and make 1,000.
More entities than one call scores is refused for both, before anything is gathered.
A ranking could not be batched in any case, since a candidate is reported unscored for the whole ranking rather than per batch, so one batch whose context held a single class would drop it from every row.
A part past the request body's limit is still split by its bytes, each smaller call carrying the part's one context.

The tabular cap was measured at the boundary rather than read from a document: 10,000 context rows are answered and 10,001 come back as `422 UNSUPPORTED_SHAPE N_train 10001 exceeds limit 10000`.

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
