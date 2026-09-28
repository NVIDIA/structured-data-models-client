# sdm-client — architecture plan

**What it is:** one Python client for NVIDIA's structured-data foundation model NIMs.
**Status:** built. All seven phases below are implemented and tested end to end against live NIMs.

> This began as the plan and is kept as the record of *why* the design is what it is, including the
> places reality corrected it. Where a section describes an interface, the shipped one is
> authoritative — see the [quickstart](../get-started/quickstart.md) and
> [prediction output](../reference/prediction-output.md).

---

## 0. One inconsistency in the contract, which nothing shipping depends on

**The contract will not let a new model exist.**

`PredictionRequest.model` and `SessionCreateRequest.model` are a **closed enum** of exactly
`kumo-tabular` and `kumo-relational`. So `model="some-future-model"` — which the source doc asks for
as an escape hatch — cannot be sent at all. The response's `model` field *is* an open string
(`maxLength: 1024`), so the contract contradicts itself.

Verified on `structured-data-api@main`, current as of today.

**This does not affect anything shipping.** `kumo-tabular` and `kumo-relational` are both in the
enum, so both work today and neither needs a contract change. What the enum forecloses is the
source doc's `model="some-future-model"` escape hatch, which is deferred along with the models it
would have served (§11). An unrecognised id is refused client-side with a message naming the
contract change it would need, rather than being sent and rejected.

If a third model is ever wanted, a small MR on `structured-data-api` opens the field. Until then
this is a note, not a task.

> **Model ids are fine.** An earlier draft claimed the NIMs still served `nemotron-*`. That was a
> stale local checkout. The serving side declares
> `ModelId = Literal["kumo-tabular", "kumo-relational"]`, and the tabular NIM's adapter registration,
> `/v1/models` entry and `model_manifest.yaml` all say `kumo-tabular`. All three places agree.
> **Re-clone before copying anything (§8).**

---

## 1. The idea in one page

Today you pick a model by calling a different method. Tomorrow the model is an argument.

```python
from sdm_client import (
    SDMClient,
    Schema,
    PredictiveQuery,
    KumoRelational,
    KumoTabular,
)

schema = Schema.from_data({'users': users_df, 'orders': orders_df})
query = PredictiveQuery(
    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
)
request = query.materialize(indices=[1, 2, 3])

with SDMClient(url='http://localhost:8000') as client:
    result = client.predict(request, model=KumoRelational())
```

A one-table question can be put to either model. The two read a request differently, though, so it
is **gathered** for one of them — say which when it is not the default:

```python
schema = Schema.from_data({'users': users_df})
question = PredictiveQuery('PREDICT users.churn FOR EACH users.user_id', schema)

client.predict(question.materialize(indices=[1, 2]), model=KumoTabular())  # ✅ the default
client.predict(
    question.materialize(indices=[1, 2], model=KumoRelational()),  # ✅ gathered as a graph
    model=KumoRelational(),
)
```

Gather and send for the same model. A request built for one is refused by the other rather than
reshaped, because the relational gather samples a neighbourhood the tabular envelope has nowhere to
put.

A relational aggregation is relational-only — the `TABULAR` profile rejects related tables, and says so
before anything is sent. `TABULAR` also refuses target with a time window, a
`WHERE` filter on the entity, an `ASSUMING` clause, and a target that is not a column of the table.

Two ways to build the one `Request`:

```
PredictiveQuery(text, schema).materialize(...)  ─┐
                                                 ├─►  Request  ─►  client.predict(request, model=…)
Request(context=…, predict=…, target=…)         ─┘
```

Four rules the whole design follows:

| Rule | Consequence |
|---|---|
| The model is an argument, never a method | `client.predict(req, model=…)`. No `client.relational()`. |
| One `Request` for every model | Tabular is the one-table case, not a second system. |
| One return type | `predict()` always returns a `PredictionResult`. Never a DataFrame-or-Explanation union. |
| A model declares itself as data | Adding a model is a declaration plus a small adapter. |

---

## 2. Packages

Five distributions in one repo, released together on one version.

| Distribution | Import | Kind | Holds |
|---|---|---|---|
| `sdm-client` | `sdm_client` | pure python | The public API: client, `Schema`, `Request`, `PredictiveQuery`, model handles, adapters, transport, sessions, results. |
| `sdm-api` | `sdm_api` | pure python | The wire only: OpenAPI-generated bindings and the types the NIM speaks. |
| `sdm-pql` | `predictive_query` | pure python | The language: grammar, parser, validator, AST, model profiles, and the schema types PQL speaks. |
| `sdm-connectors` | `sdm_connectors` | pure python | Reading source tables: SQLite, DuckDB, PostgreSQL, Snowflake, Databricks, S3. |
| `sdm-relational-engine` | `sdm_relational_engine` | **native** | The relational model's heavy client-side work: graph building, samplers, the C++ neighbor sampler. |

```
sdm-pql  ◄── sdm-api  ◄── sdm-client ──► sdm-connectors
   ▲                          │
   └────── sdm-relational-engine ◄┘   (only via the [relational] extra)
```

**Only `sdm-relational-engine` is native.** Everything else installs anywhere, so `kumo-tabular`
works on a Mac; `kumo-relational` needs Linux (x86-64 or aarch64).

```bash
pip install sdm-client                 # tabular, PQL, transport — everywhere
pip install sdm-client[relational]     # adds the native relational engine
pip install sdm-client[snowflake]      # one backend (also duckdb/sqlite/postgres/databricks/s3)
pip install sdm-client[all]            # relational + every backend
```

`[explain]` and `[relbench]` stay outside `[all]` and must be asked for by name — `[explain]` posts
row data to a third-party LLM, so installing it is a deliberate act.

### Why `sdm-pql` is separate

The language is a distribution of its own because two packages need it and neither needs the
other. `sdm-client` resolves a query against a schema; `sdm-relational-engine` reads the AST it
produces. Putting the parser in either one would make the other depend on a driver or a client it
has no use for.

It carries the parser, the AST, the validators, **and** the schema types (`Dtype`, `Stype`,
`SchemaDefinition`, `TableDefinition`), so those types have exactly one owner and nothing has to
translate between two spellings of the same enum.

The tree is first-party source of this repository, maintained here under Apache-2.0. It was
briefly carried as a vendored copy of a separate internal repository; that repository is being
retired, so there is no upstream to track, no manifest to re-record, and no dependency swap
pending. It imports as `predictive_query`, which is a name inherited from that period rather than
one this layout would choose today.

### Who owns which types

| Type | Owner | Why |
|---|---|---|
| `Dtype`, `Stype`, `TaskType`, `ProblemType` | `sdm-pql` | PQL speaks them and will be an external dependency. One owner, no translation layer. |
| `SchemaDefinition`, `TableDefinition`, `Column`, AST | `sdm-pql` | Same. The relational samplers *read* an AST, they never parse one. |
| `Profile`, `Capabilities` | `sdm-pql` | What a model may be asked, as data. §6. |
| Wire request/response, `InferenceSpec`, `OutputSpec` | `sdm-api` | The Universal TFM envelope. PQL knows nothing about HTTP. |
| The DataFrame→wire encoder (`wire/`) | `sdm-client` | **Corrected during phase 2.** §8 first put this in `sdm-api`, but `wire/dtypes.py` imports numpy and pandas, which would cost `sdm-api` its zero-dependency property — the reason the split exists. `sdm-api` keeps the vocabulary; encoding a frame is client-side work. |
| `Schema`, `Request`, `PredictionResult` | `sdm-client` | The public surface. |

---

## 3. `Schema` — the rename

`Graph` becomes **`Schema`**. Reviewers were right that calling a one-table dataset a "graph" is
confusing, and `Schema` is the word the wire contract already uses.

```python
schema = Schema.from_data({'users': df})  # one table
schema = Schema.from_data({'users': u, 'orders': o})  # many

schema['users'].primary_key = 'user_id'
schema['users']['churn'].stype = Stype.categorical
```

Everything is inferred — dtypes, semantic types, primary keys, time columns, relationships. You
override only when inference is wrong. **You never write a schema by hand.**

`PredictiveQuery` also accepts raw tables, so a one-table user need not meet `Schema` at all:

```python
PredictiveQuery('PREDICT users.churn FOR EACH users.user_id', {'users': df})
```

**Composite keys.** A warehouse row identified by `(customer, region)` is folded into one derived
column, because the wire carries a single `instance_id` per instance. `Schema` does the fold. Float
columns are refused, because a join must agree exactly on both sides. Precisely: *entity identity*
becomes scalar after folding — the wire's `TableSchema.primary_key` is still an array where a related
table's occurrence identity needs one, e.g. `[instance_id, order_id]`.

---

## 4. `Request` — one type

`Request` is the wire envelope minus the two things that are `predict()` arguments. It is not a new
abstraction; the Universal TFM contract already unified tabular and relational.

| Field | Meaning |
|---|---|
| `task` | what is being asked — kind, target, entity |
| `schema` | the tables and how they relate |
| `context` | rows where the answer is known |
| `predict` | rows being asked about |
| `inference` | gathering knobs (§7) |

Tabular is the case where `schema` has one table and no relationships. That is the whole difference.

`Request` is **singular**. A run that exceeds the server's per-call cap is split into several round
trips by the adapter — batching is transport, and does not belong in a type the user reads.

**The direct builder is the single-table one.** A relational request is
gathered, not assembled from frames: the neighbourhood around each entity is
sampled and folded onto the instance table, which is what `materialize()` does
and what the relational model is sent. So `related_tables=` is refused there
rather than building a request no path can send.

```python
Request(context=ctx_df, predict=rows_df, target='churn', task='classification')

PredictiveQuery(
    'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
).materialize(indices=[1, 2, 3], model=KumoRelational())
```

### The task kind is stated, never guessed

The source doc's last blocker: *"a 2 class target becomes multiclass, not binary."* Inferring the
kind from a column's cardinality is a guess, and it is wrong exactly when a binary target looks
categorical. What the two models actually accept settles it, and they differ:

| | Task kinds the NIM accepts |
|---|---|
| `kumo-tabular` | `classification`, `regression` |
| `kumo-relational` | `classification`, `binary_classification`, `multiclass_classification`, `regression`, `forecasting`, `temporal_link_prediction` |

So there are two different answers, and neither is "guess from cardinality":

- **Tabular has nothing to decide *between* classifications.** The NIM takes only the coarse
  `classification`, and the model resolves binary vs multiclass itself. The client sends
  `classification` and never narrows it. The blocker does not exist on this path.
- **It does have to decide classification against regression, and it got that wrong.** A query
  names a target; it does not say what is being asked of it. The derivation read the target's
  semantic type, integers are numerical, and so `PREDICT users.churn` over a 0/1 column was
  gathered as a *regression* — answered with continuous values near 0.5 that read like
  probabilities and are not. Zero-and-one is the most common way there is to write a binary
  label, so that exact case now derives `classification` and says so. The values decide, not
  their storage — a 0/1 column with one missing value arrives as `float64` and is still a label.
  Only that case: three whole numbers may be a count, and a fractional pair is not a label, so
  both are left as regressions.
- **The caller has the last word where there is a decision to make.** `task=` on the direct
  builder, or `materialize(task=…)` on the single-table PQL path, is authoritative and overrides
  everything above it. Stating it also silences the warning, which is the point of the warning.
  A relational query has no such decision — the engine resolves the query and states the kind in
  the payload — so passing `task=` with one is refused rather than quietly dropped.

PQL's own `TaskType` already has `BINARY_CLASSIFICATION` and `MULTICLASS_CLASSIFICATION`, but a query
resolves only to the coarse `ProblemType` (`CLASSIFY`/`RANK`/`FORECAST`) — the narrowing happens
downstream from the target's cardinality, which is exactly where the source doc's bug lives. **The
caller's explicit `task=` wins over that derivation, and when the derivation is the only signal and
the target has exactly two classes the client warns**, naming both kinds rather than silently picking
multiclass.

### Which rows get scored — the one question the source doc left open

A query says *what* to ask. It never says *which rows* to ask about. That is always an explicit argument.

| Builder | Rows to score | Available |
|---|---|---|
| `Request(context=…, predict=…)` | `predict` — any rows, including rows the model has never seen | v1 |
| `query.materialize(indices=[…])` | existing rows, looked up by entity key | v1, tabular and relational |
| `query.materialize(predict=rows_df)` | supplied rows, which need not exist in the tables | phase 7, tabular |

`materialize()` with neither argument raises, naming both. **There is no all-rows default** — it would
silently score a whole table.

**The model a request is gathered for is an argument, not a consequence of the schema's shape.**
`materialize(model=…)` takes the same handle or id that `predict()` does. Left unsaid, the shape
decides, and the shape that counts is the one `Request.is_relational` reports: a second table or a
relationship sends it relationally, so a lone table that points at itself is gathered as the graph
it is. Everything else is tabular, and `model='kumo-relational'` gathers it as a graph of one node. Without that argument the two models could disagree
with no way to reconcile them: a one-table request was always built tabular, so the relational
model was refused it and nothing the caller passed could say otherwise. Gathering and sending
disagree loudly now: each adapter refuses a request gathered for the other before it is sent.

**Tabular PQL has no key to name, and that is decided, not open.** `FOR EACH` today must name a
primary key, and a tabular frame has none. In phase 7 `Schema` synthesizes a request-local row id for
a keyless table — a real column, positional, never sent as data and never surfacing in the result,
which carries `row_index`. `FOR EACH users.<synthetic>` then validates unchanged, so the PQL validator
is **not** relaxed and no query that is illegal today becomes legal. This is the concrete form of what review
asked for here.

On the question review raised here: "unseen" means a row that is not in the tables the schema was
built from. `FOR EACH` enumerates rows that *are* there, so PQL text alone can never produce them. The
rows are an input, not a derivation — which is exactly why the direct builder carries tabular in v1.

---

## 5. `predict`, `fit`, and one result type

### fit is a session

The contract already has it: `POST /v1/sessions` pins context and runs no inference;
`POST /v1/sessions/{id}/predictions` scores against it. That is `fit` and `predict`.

```python
fitted = client.fit(request, model=KumoRelational())  # POST /v1/sessions
fitted.predict(more_rows)  # reuses the pinned context
fitted.close()  # DELETE /v1/sessions/{id}
```

`client.predict()` still works statelessly in one call, and falls back to the stateless route if a NIM
does not serve sessions.

### One return type

The old design returned a DataFrame, or an `Explanation` if you asked to explain. That union was
undocumented. Now:

```python
result = client.predict(
    request, model=KumoRelational(), outputs=['prediction', 'explanation']
)

result.to_pandas()  # always a DataFrame
result.explanation  # None unless you asked for it
```

Explanations are an **output field**, not a method. `explain=True` remains sugar for
`outputs=[…, "explanation"]`.

The DataFrame's shape depends on the **task**, not the model. Verified against a live endpoint:
classification and multiclass both give **one row per entity**, with multiclass carrying a
probability per class as columns rather than a row per class. Ranking also gives one row per
entity, holding its `k` candidates as a list, because expanding them would leave `row_index` no
longer identifying a row; forecasting gives one row per entity per step. An earlier draft said
multiclass gave a row per class, and another said a ranking gave `k` rows; neither does.

**Every row carries `row_index`. It also carries the entity key when the model has one** — tabular
requests may have no key at all. Join on the key when present, otherwise on `row_index`.
**Never join on row order.**

### What each model can produce

The contract defines seven output fields. **This table is read from the servers**, not from the source
doc — the tabular adapter and the relational NIM's universal adapter each declare their own
capabilities, and this mirrors what they answer:

| Output | `kumo-relational` | `kumo-tabular` |
|---|---|---|
| `prediction` | ✅ | ✅ |
| `probabilities` | ✅ | ✅ |
| `quantiles` | ✅ | ✅ |
| `scores` | ✅ | ❌ |
| `rankings` | ✅ | ❌ |
| `embeddings` | ✅ | ❌ |
| `explanation` | ✅ | ❌ |

> **The source doc's matrix is wrong here.** It lists `scores` as tabular-only and not supported by
> the relational model. The relational adapter advertises `scores` and returns it; the tabular adapter
> advertises only `prediction`, `probabilities`, `quantiles`. We follow the servers. Phase 6 pins this
> against the live NIMs, and `ModelSpec.runtime` is corrected to whatever they answer.

Asking a model for a field it does not produce fails client-side, naming what it does produce.

> **A second relational build serves less.** The table above is the shipping
> kumoml-backed NIM. A `structured-data-models`-backed build of the same model id serves
> only classification and regression, and produces only `prediction`, `probabilities` and
> `quantiles`. The client declares the broader one, because the contract has no
> capabilities route to ask with — `/v1/models` returns an id and nothing else — and
> refusing locally what one deployment can answer would be worse than letting the other
> say no. Discovered by running against both.

---

## 6. Adding a model

Three pieces. Two are cheap; one is a contract change we do not own.

**1. The contract must accommodate the model — and this is bigger than an id.** `PredictionRequest.model`
is a closed enum, so at minimum a new model needs an MR on `structured-data-api` and a regeneration of
`sdm-api`. Until §0 is fixed, an unrecognised id fails **client-side with a message naming that MR**
rather than being silently sent and rejected.

But a genuinely new *kind* of model needs more than a name. Anomaly detection is the honest test:

| What it needs | Does the contract have it? |
|---|---|
| a task kind `anomaly_detection` | ❌ `TaskSpec.kind` has no such value |
| a request with **no label target** | ❌ `TaskSpec.target` is required |
| score direction, threshold, calibration | ❌ nothing standardises them |
| a place to put the score | ✅ `OutputSpec.fields` already has `scores` |

So **"add a model" is a contract change plus a profile plus an adapter, and for a new task shape the
contract change is the large part.** Anything else would push the meaning into open `metadata` dicts,
and a unified client whose semantics live in untyped dicts is a facade over per-model behaviour, not
a common interface. Timeseries is the cheaper case — `forecasting` already exists as a task kind — which
is why it is the sensible third model, not anomaly.

This is the seam's real cost, stated up front so nobody discovers it at model three.

**2. A profile** says what the model may be asked. `sdm-pql` already ships `RELATIONAL`, `TABULAR`,
`TIMESERIES`, `ANOMALY`. A capability may only *reject*: the language accepts a fixed set of queries
and each model accepts a subset, so a new model can never widen the language.

```python
TABULAR = Profile(
    mode=INFERENCE,
    capabilities=Capabilities(
        model='a tabular model',
        relationships=False,
        multiple_tables=False,
        temporal=False,
        entity_filters=False,
        counterfactuals=False,
        derived_targets=False,
        link_prediction=False,
        forecasting=False,
    ),
)
```

**3. An adapter** does what a profile cannot describe: shape the wire request, normalize the response,
choose a session strategy, release resources.

```python
class ModelAdapter(Protocol):
    spec: ModelSpec  # model id + profile + runtime facts

    def default_output(self, request) -> OutputSpec: ...
    def prediction_plan(
        self, request, output
    ) -> Iterable[PredictionRequest]: ...
    def session_plan(self, request) -> SessionCreateRequest: ...
    def normalize(self, response) -> PredictionResult: ...
    def close(self) -> None: ...
```

The adapter does not own **query-shape** validation — the profile does. Everything else is validated
by shared client code combining the profile with `ModelSpec.runtime`: which output fields exist,
whether sessions are served, and the resource limits in §7. A profile cannot know whether *this
deployment* has a regressor checkpoint; that is a runtime fact and lives beside it, never duplicated
into it.

> **Cost, stated plainly.** The predecessor client deleted its registry on purpose. We are
> re-opening it, and that buys back an adapter protocol to keep stable and conformance tests to
> maintain. That is the price of serving more than one model, and it is the price the doc asks for.

A model with no heavy client-side work (tabular) is an adapter alone. A model with heavy work
(relational) also gets its own package behind an extra, so base installs stay light.

### What the endpoint says vs what the client believes

The old SDK reported its own registry, so it claimed a model the NIM did not serve.

| Call | Answer |
|---|---|
| `client.models()` | Read from the endpoint's `/v1/models`. Endpoint truth. |
| `client.capabilities(model)` | Local: what this client can build and send. Tagged `source="sdk"`. |

`/v1/models` is a NIM runtime route and is **not** one of the contract's four paths, so a NIM without
it degrades to a clear error rather than a wrong answer.

---

## 7. Knobs: `run_mode` is gone

`run_mode` was a preset over two numbers, and it leaked a server detail into the user's hands.

We checked the server. It distinguishes only two context budgets, the smaller one
reached by `fast`; the client's four-tier `debug/fast/normal/best` ladder was a
client-side fiction. Both numbers are in [limits](../reference/limits.md).

So the user sets the real numbers, and the client derives the wire value:

```python
query.materialize(indices=[1, 2, 3], num_context_samples=500, num_neighbors=[8, 8])
```

**And the number is honoured, not just used to pick a bucket.** The endpoint's
budget is a ceiling, not a request; gathering 1,000 because that is the bucket that
admits 500 would make the number as decorative as the two preset values that did
nothing, which is the defect this replaced. The count narrows the gathered
examples where they are built, before anything is encoded.

The client **pre-flights the server's limits before sending**, so you get a readable error instead of
a 500, naming the limit and how to get under it.

Every value and where each is checked lives in
[limits](../reference/limits.md), and only there. This page had its own copy of
those tables and they went stale twice: a limit that moved was corrected in the
reference and not here, so the two pages disagreed about what the client does.

---

## 8. What we copy

Most of this client is not new. An earlier internal NVIDIA relational client, at 1.0.0, is post-QA,
hardened and live-tested, and a separate internal client carried the tabular surface. We copy both
rather than rewrite them.

What came from where, and what changed on the way:

| From | To | Change |
|---|---|---|
| The predecessor client's connectors package | `sdm-connectors` | names only |
| ⟂ its relational engine, `rfm/` | `sdm-relational-engine` | `Graph`→`Schema`; PQL types from `predictive_query` |
| ⟂ its C++ neighbour sampler, `csrc/neighbor_sampler.cpp` | `sdm-relational-engine` | none |
| ⟂ its relational adapter | `sdm_client/adapters/relational.py` | returns the unified `Request`/`PredictionResult` |
| ⟂ its transport | `sdm_client/core/transport.py` | none |
| The earlier tabular client's adapter | `sdm_client/adapters/tabular.py` | model id renamed to `kumo-tabular` |
| ⟂ its wire types | `sdm-client` | none (see §2: not `sdm-api`, which stays dependency-free) |
| A separate internal Predictive Query Language repository, now retired | `sdm-pql` | none; the tree is maintained here from this point on |

Two hard-won behaviours to carry over deliberately: every redirect hop is closed rather than drained,
and credentials in a URL are stripped everywhere the URL is rendered.

---

## 9. Phases

Each phase is independently shippable and tested.

| # | Phase | Ships | Done when |
|---|---|---|---|
| 1 | **Foundation** | Repo, CI, `sdm-api` + `sdm-pql`, SPDX headers, provenance records | `predictive_query` imports clean; generated bindings match the contract hash |
| 2 | **Client + tabular** | `SDMClient`, `Schema`, `Request`, `PredictionResult`, transport, sessions, tabular adapter | a direct `Request` predicts against the tabular NIM |
| 3 | **Connectors** | `sdm-connectors` | every backend reads a table |
| 4 | **Relational** | `sdm-relational-engine`, relational adapter, PQL materialization | `PredictiveQuery(...).materialize(indices=…)` predicts against the relational NIM |
| 5 | **fit + explain** | Session handle, explanation outputs, pre-flight limits | `fit()`/`predict()` reuse one context; every §7 limit fails locally |
| 6 | **Docs + E2E** | Docs, notebook, live-NIM harness for both models | both models pass live E2E |
| 7 | **PQL for tabular** | `materialize(predict=…)` emits a one-table request | a tabular question runs from PQL text |

Phases 2 and 3 are independent. Phase 4 needs 1–3. Nothing is blocked on the contract: both
shipping models are already in its enum (§0).

---

## 10. This gets open-sourced

It ships internal first, but public release is the destination, not a maybe. The predecessor client
reached public by **retrofit** — a full audit plus an OSRB pass after the code existed. We are not
repeating that. Every rule below is cheap on day one and expensive later.

**Written OSS-clean from commit 1, and now carrying the rest:**

| Held from commit 1 | Added for the release |
|---|---|
| SPDX header on every source file | `LICENSE` and `NOTICE` at the root, and in every distribution's metadata |
| A provenance record for every vendored tree, CI-enforced | `GOVERNANCE.md`, `MAINTAINERS.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md` |
| Provenance for the generated OpenAPI bindings: spec path, ref, SHA-256, generator version, and a test that they match | Public issue policy, and the repository URL every package publishes to, checked by the hygiene script |
| An automated scan for internal URLs, ticket ids, usernames, engineer TODOs and codenames in shipped packages and docs | — |

Two things in the right-hand column are still open, and neither is a
nice-to-have. Public CI and publishing provenance land when the repository
does. A generated **SBOM** covering direct, optional, transitive, build, native
and connector dependencies is not written yet, and a hand-maintained dependency
table is not one; treat it as a gate on the release.

License is **Apache-2.0** — it matches the predecessor client, the source of most of what we
copy, so nothing we carry across needs relicensing.

### Vendored code is the risk, and we already know how it breaks

We vendor one tree verbatim and rename aggressively across the repo (`Graph`→`Schema`,
`kumo_*`→`sdm_*`). That exact combination has already caused a licensing defect in this family of
repos: a repo-wide rename silently rewrote two identifiers **inside** the minified `mermaid.min.js`,
turning a vendored artifact into a modified one while the LICENSE still described it as upstream's
bytes. Six bytes in a 3.3 MB file. No test caught it, and it invalidated the component enumeration
OSRB had already validated.

So, as a rule, not a suggestion:

1. Every vendored tree has a **digest pinned beside it and asserted in CI** — for `mermaid.min.js`,
   a single SHA-256 over the bundle.
2. A test asserts the rename **did not inject target-repo identifiers** into a vendored tree — no
   `sdm_`, `Schema`, or `sdm-client` strings appearing where upstream has none. The assertion is
   "unchanged from upstream", not "contains no forbidden word".

> This rule applied to `predictive_query` while that tree was a verbatim copy of a separate
> repository. That repository is being retired, so the tree is first-party source of this one now:
> there is no upstream to diverge from, and a manifest pinned to a baseline nobody can fetch would
> assert nothing. It is linted and header-checked like any other first-party tree instead. `mermaid`
> is the only tree the rule still governs.
3. The rename tool **excludes vendored paths and `assets/` by configuration, before it runs** — then
   the hash test verifies it afterwards. Exclusion prevents the churn; the hash only detects it.
4. Bumping a vendored bundle means **re-enumerating its components from the new release's manifest** —
   a version bump silently moves sub-component versions too.

The predecessor client already ships this test; we copy it.

### What ships that we did not write

| Component | License | Where | Note |
|---|---|---|---|
| ANTLR-generated parser | Apache-2.0 | `sdm-pql` | Generated by ANTLR from `PQLGrammar.g4` in the same tree; the grammar and everything around it are first-party |
| ANTLR Python runtime | BSD-3-Clause | `sdm-pql` dependency | Required by the PQL parser |
| `mermaid.min.js` (vendored) | MIT, and a bundle — its components are enumerated in `LICENSE` | `sdm-relational-engine` assets | **Kept.** `visualize()` draws offline, and the notebook environments this client is used from — Snowflake and Databricks — block external scripts, so a CDN-fetched diagram would not draw at all. Upstream bytes, SHA-256 pinned, digest asserted by a test. `to_mermaid()` returns the source for anyone who would rather render it elsewhere. |
| pandas, numpy, requests, pydantic | BSD / Apache-2.0 | dependencies | Resolved at install, not vendored |

OSRB needs the provenance records and the inventory above. The existing ticket was approved against
the *old* repo, so a new repo is a new submission regardless — writing these from commit 1 makes that
a form-filling exercise rather than an audit.

---

## 11. Deferred, on purpose

| Thing | Why | What would change it |
|---|---|---|
| Timeseries, anomaly, Tesseract models | Not asked for yet | A NIM serves one — profiles already exist, plus a contract MR (§6) |
| Warehouse push-down | The wire requires embedded data | The contract gains source references |
| Public OSS release | Internal first, written OSS-clean from commit 1 (§10) | A release decision; the retrofit is form-filling, not an audit |
| Arrow IPC | The wire is JSON-arrays-only everywhere | The contract activates it |

---

## 12. Open questions for the human

None outstanding. Two earlier ones are settled:

- **The client is `SDMClient`.**
- **The contract's closed `model` enum is not a task.** Both shipping models are in it. It only
  forecloses the escape hatch for a third, which is deferred along with the models it would serve
  (§11). If one is ever wanted, a small `structured-data-api` MR opens the field.
