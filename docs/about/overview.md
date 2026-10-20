# Overview

Structured-data foundation models predict **in context**. You give the model
labelled rows and the rows you want scored, and it answers in one pass. There
is no training loop, no hyperparameter search, and no per-dataset artifact.

This client builds the request, sends it to a NIM, and gives you back a
`PredictionResult`, which `to_pandas()` turns into a DataFrame. Inference runs
in the NIM, so the client downloads no weights and needs no GPU. A client
opened with `SDMClient.in_process()` runs the model in your own process
instead, with the same calls; that one loads the weights, and uses a GPU when
there is one.

## The one idea

**The model is an argument, not a method name.**

```python
result = client.predict(request, model=KumoTabular())
result = client.predict(request, model=KumoRelational())
```

One call uses either model. Because the models read requests differently, each
request is gathered for one of them. `materialize(model=...)` selects the model
and defaults to the one implied by the schema's shape. A model rejects an
unsupported request *before* anything is sent. Its supported requests are
declared as data instead of discovered from a 422.

## Two models today

| Model | For | Needs |
|---|---|---|
| `kumo-tabular` | one table | nothing (pure Python, runs anywhere) |
| `kumo-relational` | several related tables, joined as a graph | `structured-data-models-client[relational]`, a native build |

## When not to use it

- **You want to train a model.** These models do not train. If you need a model
  fitted to your data, this is the wrong tool.
- **Your data cannot leave your environment.** Run a NIM in your environment
  and point the client at it, or run the model in your own process with
  `SDMClient.in_process()`.
