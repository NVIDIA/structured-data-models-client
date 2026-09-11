# Overview

Structured-data foundation models predict **in context**. You give the model
labelled rows and the rows you want scored, and it answers in one pass. There
is no training loop, no hyperparameter search, and no per-dataset artifact.

This client builds the request, sends it to a NIM, and gives you back a
DataFrame. Inference runs in the NIM, so the client downloads no weights and
needs no GPU.

## The one idea

**The model is an argument, not a method name.**

```python
result = client.predict(request, model=KumoTabular())
result = client.predict(request, model=KumoRelational())
```

One call answers with either model. The request itself is gathered for one of
them -- the two read it differently -- so `materialize(model=...)` says which,
and defaults to what the schema's shape implies. A model that cannot answer a
request says so *before* anything is sent, because what a model may be asked is
declared as data rather than discovered from a 422.

## Two models today

| Model | For | Needs |
|---|---|---|
| `kumo-tabular` | one table | nothing — pure Python, runs anywhere |
| `kumo-relational` | several related tables, joined as a graph | `sdm-client[relational]`, a native build |

## When not to use it

- **You want to train a model.** These models do not train. If you need a model
  fitted to your data, this is the wrong tool.
- **Your data cannot leave your environment.** It can, but only to a NIM you
  run yourself. Run one and point the client at it.
- **You want inference in-process.** Everything here is an HTTP call.
