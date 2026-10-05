# Agent skills for the Structured Data Models client

Two skills, one per model, because the two models answer different questions
and almost nothing transfers between them.

| Skill | Model | Use it when |
|---|---|---|
| [`tabular/SKILL.md`](tabular/SKILL.md) | `kumo-tabular` | The question is about rows of one table |
| [`relational/SKILL.md`](relational/SKILL.md) | `kumo-relational` | The question is about an entity and what it is linked to |
| [`relational/pql.md`](relational/pql.md) | | Writing the predictive query itself |

Load one. An agent answering a question about a single table does not need the
relational material, and the relational model's six task kinds do not exist on
the tabular one.

## What these models are

Neither model is trained on your data. Both are in-context learners: a request
carries labelled rows as context and unlabelled rows to score, the model reads
both in one forward pass, and nothing is fitted or stored. There is no training
job, no checkpoint and no model registry. A question is asked and answered.

That is what decides the shape of every example here: there is nothing to
train, so each skill is about describing the data and asking the question.

## Provenance

Every claim in these files was read from the shipped code, or run.

Output marked **Verified** was produced by running the example above it against
a live endpoint and pasting the result back. Everything else that can be
checked without an endpoint is checked by
[`../tests/test_skills.py`](../tests/test_skills.py), which reads the tables
out of these files and asks the client whether they are still true: the task
each query resolves to, the aggregations that are answered and the ones that
are refused, and the names every example imports.

Where a skill says the client allows something the endpoint refuses, that gap
is the point and is stated as such. If a claim here contradicts the code, the
code is right and the skill is stale: say so rather than working around it.

## What is not here

- Training, fine-tuning, checkpoints, model registries. The models do not train.
- `predictive_query`, the parser package behind `PredictiveQuery`. It is
  internal, as [`docs/reference/stability.md`](../docs/reference/stability.md)
  says. Write the query as a string.
- Deployment, authentication and endpoint operation. A skill assumes a URL.
