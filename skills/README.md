# Agent skills for the Structured Data Models client

Two skills, one per model, because the two models answer different questions
and almost nothing transfers between them.

Start at one of these two. Each is enough on its own to ask a first question.

| Skill | Model | Load it when |
|---|---|---|
| [`tabular/SKILL.md`](tabular/SKILL.md) | `kumo-tabular` | The question is about rows of one table |
| [`relational/SKILL.md`](relational/SKILL.md) | `kumo-relational` | The question is about an entity and what it is linked to |

An agent answering a question about a single table does not need the
relational material, and the four task kinds that make the relational model
what it is — the two shapes of classification, forecasting and link prediction
— are not on the tabular one at all.

Each skill names the references it needs. They stand on their own, so one can
be loaded without the skill it belongs to:

| Reference | |
|---|---|
| [`relational/pql.md`](relational/pql.md) | Writing the predictive query |
| [`tabular/references/pql.md`](tabular/references/pql.md) | The same language on one table, and what it refuses |
| [`relational/references/warehouses.md`](relational/references/warehouses.md) | Tables that stay in a warehouse, and metric views |
| [`relational/references/beyond-one-answer.md`](relational/references/beyond-one-answer.md) | Rankings, forecasts and explanations |

## What these models are

Neither model is trained on your data. Both are in-context learners: a request
carries labelled rows as context and unlabelled rows to score, and the model
reads both in one forward pass. No weights are fitted, and nothing you send
changes the model. There is no training job and no model registry.

A session is the one thing that is held: `fit()` pins a context in the
endpoint's memory so later calls can reuse it, and it stays there until
`close()` or until it expires. That is a cache of what you sent, not anything
learned from it.

That is what decides the shape of every example here: there is nothing to
train, so each skill is about describing the data and asking the question.

## Provenance

Every claim in these files was read from the shipped code, or run.

Output marked Verified was produced by running the example above it against
a live endpoint and pasting the result back. Read those blocks for their shape
— the columns, how they are keyed, how many rows come back. The numbers in them
are one deployment's answers on made-up data, not anything this client
promises, and a different build will print different ones.

[`../tests/test_skills.py`](../tests/test_skills.py) reads the tables out of
these files and asks the client whether they are still true: both capability
tables against `capabilities()`, the relational limits against the shipped
constants, the task each query resolves to, the aggregations that are answered
and the ones that are refused, and the names every example imports from this project -- `sdm_client`,
`sdm_connectors` and its `databricks` module, which for a fragment is the
only check its imports get. Run with
`-m live_nim` and two endpoint URLs, it also re-runs every example that can be
run and compares what it printed to the block pasted beneath it. An example
that reads a name no example before it binds is a fragment -- a Databricks
session, a Snowflake one -- and those are checked as code but executed
nowhere, so none of them is allowed to show output.

Where a skill says the client allows something the endpoint refuses, that gap
is the point and is stated as such. If a claim here contradicts the code, the
code is right and the skill is stale: say so rather than working around it.

## What is not here

- Training, fine-tuning, checkpoints, model registries. The models do not train.
- `predictive_query`, the parser package behind `PredictiveQuery`. It is
  internal, as [`docs/reference/stability.md`](../docs/reference/stability.md)
  says. Write the query as a string.
- Deployment, authentication and endpoint operation. A skill assumes a URL.
