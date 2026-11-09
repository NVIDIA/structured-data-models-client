# Running kumo-relational in this process

Nothing deployed to point at, or tables that must not leave the machine: the
same client runs the model here instead, with the same calls. This is how a
notebook asks with no NIM at all.

It takes two extras, the engine that gathers a request and the models that
answer one:

```
pip install 'structured-data-models-client[relational,in-process]'
```

The second brings structured-data-models and PyTorch, and needs Python 3.11
or later. Without it, `SDMClient.in_process()` raises `MissingExtraError`
naming that command; on Python 3.10 the error names the Python it needs
instead. The first prediction fetches the published weights,
`nvidia/Kumo-Relational` on Hugging Face, into the Hugging Face cache, and
later ones read them from there. That is both task checkpoints, about 240 MB,
once. No Hugging Face account or token is needed; a warning about
unauthenticated requests is about rate limits, not access. The model runs on
a GPU when there is one, else on the CPU.

Open the client with `SDMClient.in_process()` in place of a URL. Everything
else is [the whole interface](../SKILL.md): the schema, the query and
`materialize()` are unchanged, and gathering still happens there, in this
client. What moves is the model run, and `client.url` says where it answers
from.

```python
import pandas as pd
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    SDMClient,
    Schema,
)

users = pd.DataFrame({'user_id': range(12), 'segment': ['a', 'b'] * 6})
orders = pd.DataFrame(
    {
        'order_id': range(120),
        'user_id': [i % 12 for i in range(120)],
        'item': [i % 4 for i in range(120)],
        'price': [10.0 + i for i in range(120)],
        'ordered_at': pd.date_range('2026-01-01', periods=120, freq='D'),
    }
)
items = pd.DataFrame({'item': range(4), 'kind': list('abcd')})

schema = Schema.from_data(
    {'users': users, 'orders': orders, 'items': items},
    primary_keys={'users': 'user_id', 'orders': 'order_id', 'items': 'item'},
    time_columns={'orders': 'ordered_at'},
    relationships=[
        Relationship('orders', 'user_id', 'users', 'user_id'),
        Relationship('orders', 'item', 'items', 'item'),
    ],
)
request = PredictiveQuery(
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
).materialize(indices=[1, 2, 3], model=KumoRelational())

with SDMClient.in_process() as client:
    print(client.url, client.models())
    answer = client.predict(request, model=KumoRelational())
print(list(answer.to_pandas().columns), len(answer.to_pandas()), 'rows')

named = answer.to_pandas().merge(users, left_on='id', right_on='user_id')
print(list(named.columns))
```

**Verified** in this process, on a CPU:

```
in-process ['kumo-tabular', 'kumo-relational']
['row_index', 'id', 'prediction'] 3 rows
['row_index', 'id', 'prediction', 'user_id', 'segment']
```

The frame has the shape a NIM gives: `row_index`, the entity's `id` and the
`prediction`. `id` is the key the question was asked for, and merging on it
is how the answers go back onto those entities: never line them up by row
order. The numbers are not shown because they depend on where it ran, which
is the first difference below.

## An explanation, here

The shipping deployment refuses `explanation` on every task. This process
serves it, with the same `result.explanations` a deployment that serves it
gives, keyed by the entity:

```python
with SDMClient.in_process() as client:
    explained = client.predict(
        request, model=KumoRelational(), outputs=['prediction', 'explanation']
    )
print(sorted(explained.explanations.gradients))
```

**Verified** in this process, on a CPU:

```
[1, 2, 3]
```

Reading one entry is in [the whole interface](../SKILL.md), under
explanations.

## What differs from a NIM

- **Where it ran decides the numbers.** On a GPU, with the published weights,
  the answers are a NIM's exactly. On a CPU the model computes at full
  precision and draws its random numbers there, so a prediction can differ
  from a GPU-hosted NIM's by more than rounding. The draw is seeded, so two
  runs on one machine print the same numbers; it is the device that moves
  them, not the run. Compare against a NIM on the same device, or compare
  shapes rather than numbers, as above.
- **A deployment's limits are not applied.** This client's own still are:
  every limit `materialize()` checks is checked as it would be for a NIM. A
  deployment also protects a shared server with limits of its own, such as
  the longest string a cell may hold, and nothing applies those here.
- **A refusal has no HTTP status.** It is a `StructuredDataError` carrying the
  code a NIM would give, and never a `NimRequestError`, since nothing was
  sent. Inside a session the code is the reason's own, where a NIM files the
  refusal under `SESSION_CREATE_VALIDATION_FAILED`. Catching
  `StructuredDataError` catches it as before.
- **A session is held in this process's memory** until `close()`, or until
  the client closes, with no expiry. `fit()` works as it does against a NIM,
  and the `with` block above is what lets go of what it holds.
- **`client.models()` lists what it can run**, rather than what an endpoint
  advertises: both models when it loads them itself, or the ones passed in.

## A machine that cannot reach Hugging Face

The weights are read from the Hugging Face cache, and `HF_HOME` says where
that is. Fetch them once where the Hub is reachable, copy that cache to the
machine that cannot reach it, point `HF_HOME` at the copy and set
`HF_HUB_OFFLINE=1`: the model loads from it with no network at all. With
nothing there, an offline run refuses with `MODEL_NOT_READY` naming
`LocalEntryNotFoundError`, rather than waiting on a request that cannot
arrive.

## Choosing the device, or passing a loaded model

`SDMClient.in_process(device='cuda')` places the models this client loads. A
model already loaded with structured-data-models is used as it is:
`SDMClient.in_process(loaded)`, where `loaded` is a
`sdm.models.KumoRelational`, shares its weights with every context the client
fits, and refuses `device=` beside it, since that model is already somewhere.
The quickstart's section on a model in this process, in
[`docs/get-started/quickstart.md`](../../../docs/get-started/quickstart.md),
shows it.
