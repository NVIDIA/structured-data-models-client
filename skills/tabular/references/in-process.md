# Running kumo-tabular in this process

Nothing deployed to point at, or rows that must not leave the machine: the
same client runs the model here instead, with the same calls. This is how a
notebook asks with no NIM at all.

It is an extra:

```
pip install 'structured-data-models-client[in-process]'
```

No published release carries `in_process()` yet, so until one does, that
command installs a client without it. Install from a checkout instead:

```
git clone https://github.com/NVIDIA/structured-data-models-client
cd structured-data-models-client
pip install -e './packages/structured-data-models-client[in-process]'
```

A pip warning that the client `does not provide the extra 'in-process'`, or
an `AttributeError` saying `SDMClient` has no attribute `in_process`, is an
installed client older than the feature, whichever command installed it.

The extra brings structured-data-models and PyTorch, and needs Python 3.11 or
later. Without it, `SDMClient.in_process()` raises `MissingExtraError` naming
that command; on Python 3.10 the error names the Python it needs instead. The
first prediction fetches the published weights, `nvidia/Kumo-Tabular` on
Hugging Face, into the Hugging Face cache, and later ones read them from
there. That is both task checkpoints of the large model, about 1.7 GB, once;
`size='small'`, below, is about 220 MB. No Hugging Face account or token is
needed; a warning about unauthenticated requests is about rate limits, not
access. The model runs on a CUDA GPU when there is one, else on the CPU: the
check is CUDA's alone, so on Apple silicon it runs on the CPU.

Open the client with `SDMClient.in_process()` in place of a URL. Everything
else is [the whole interface](../SKILL.md): the request, the model handle and
the result are the same objects, and `client.url` says where it answers from.

```python
import pandas as pd
from sdm_client import Request, SDMClient, KumoTabular

labelled = pd.DataFrame(
    {
        'age': [31.0, 44.5, 28.0, 51.0, 39.0],
        'spend': [120.0, 430.0, 90.0, 510.0, 260.0],
        'churn': [0, 1, 0, 1, 0],
    }
)
to_score = pd.DataFrame({'age': [33.0, 47.0], 'spend': [150.0, 480.0]})
request = Request(
    context=labelled, predict=to_score, target='churn', task='classification'
)

with SDMClient.in_process() as client:
    print(client.url, client.models())
    answer = client.predict(
        request, model=KumoTabular(), outputs=['prediction', 'probabilities']
    )
print(list(answer.to_pandas().columns), len(answer.to_pandas()), 'rows')

scored = (
    to_score.reset_index(drop=True)
    .reset_index(names='row_index')
    .merge(answer.to_pandas(), on='row_index')
)
print(list(scored.columns))
```

**Verified** in this process, on a CPU:

```
in-process ['kumo-tabular', 'kumo-relational']
['row_index', 'prediction', 'probability_0', 'probability_1'] 2 rows
['row_index', 'age', 'spend', 'prediction', 'probability_0', 'probability_1']
```

The frame has the shape a NIM gives: `row_index`, `prediction` and a
probability column per class. `row_index` is the *position* in the frame
passed as `predict`, not that frame's index label, so drop the index before
making it a column: a frame that was filtered or sorted carries labels that
are no longer its positions, and merging on those quietly pairs a row with
another row's answer, or drops it. Merge on `row_index`, never line the two
frames up by row order. The numbers are not shown because they depend
on where it ran, which is the first difference below.

## What differs from a NIM

- **Where it ran decides the numbers.** On a GPU, with the published weights,
  the answers are a NIM's exactly. On a CPU the model computes at full
  precision and draws its random numbers there, so a probability can differ
  from a GPU-hosted NIM's by more than rounding: in the quickstart's
  comparison a class probability moved by 0.16. The draw is seeded, so two
  runs on one machine print the same numbers; it is the device that moves
  them, not the run. Compare against a NIM on the same device, or compare
  shapes rather than numbers, as above.
- **A deployment's limits are not applied.** This client's own still are: a
  context above 10,000 rows is refused as it would be for a NIM. A deployment
  also protects a shared server with limits of its own, such as the longest
  string a cell may hold, and nothing applies those here. A request is also
  read for what the model needs rather than against every rule of the request
  contract, so a name longer than the contract's 1,024 characters is answered
  here and refused by a NIM with `VALIDATION_FAILED`.
- **A refusal has no HTTP status.** It is a `StructuredDataError` carrying the
  code a NIM would give, `UNSUPPORTED_OUTPUT` for `quantiles` on a
  classification, and never a `NimRequestError`, since nothing was sent.
  Catching `StructuredDataError` catches it as before.
- **A session is held in this process's memory** until `close()`, or until
  the client closes, with no expiry. `fit()` works as it does against a NIM,
  and the `with` block above is what lets go of what it holds. Closing also
  lets go of the models the client loaded, and the next client reads them
  from the cache into memory again, so a notebook that opens a `with` block
  per cell pays that load per cell: open one client, use it across the
  cells, and `close()` it at the end.
- **`client.models()` lists what it can run**, rather than what an endpoint
  advertises: both models when it loads them itself, or the ones passed in.
  Listing `kumo-relational` is not a report that the `relational` extra is
  installed; building a relational request still needs it.

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
`SDMClient.in_process(loaded)`, where `loaded` is a `sdm.models.KumoTabular`,
shares its weights with every context the client fits, and refuses `device=`
beside it, since that model is already somewhere. Loading it yourself is also
how to pick a size: the client loads `'large'`, and `'medium'` and `'small'`
are published beside it.

```python
import sdm

small = sdm.models.KumoTabular(size='small')
with SDMClient.in_process(small) as client:
    print(client.models())
    smaller = client.predict(request, model=KumoTabular())
print(list(smaller.to_pandas().columns))
```

**Verified** in this process, on a CPU:

```
['kumo-tabular']
['row_index', 'prediction', 'probability_0', 'probability_1']
```

A model loaded that way arrives in evaluation mode and loaded for both tasks,
which is what this client asks of it. One put back into training mode is
refused with `INVALID_CONFIGURATION` naming `.eval()`, and one loaded for a
single `task=` is refused with `UNSUPPORTED_TASK` for the other; a NIM's
model is never either.
