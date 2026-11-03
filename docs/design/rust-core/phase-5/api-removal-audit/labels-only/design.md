<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compute known answers without calling a model

**Approved for contract-first implementation: add a labels operation to Python/Java `PredictiveQuery`.**
This produces historical ground truth for evaluation, debugging a prediction target or preparing a labeled dataset.
It does not train or fit a model.

## What is missing?

Illustrative question: “For each customer, how much did they spend in the 30 days after January 1?”
With historical orders, the engine can calculate the actual answer directly.

```text
Query + schema + historical anchor + row limit
                    ↓
Rust: choose eligible entities and evaluate target from data
                    ↓
ENTITY | ANCHOR_TIMESTAMP | TARGET
       Python DataFrame / Java Arrow table
```

| Question | Answer |
|---|---|
| Old support | `KumoRelational.get_train_table(query, size, anchor_time, random_seed, max_iterations)` |
| Intended use | Obtain known target values without neighborhood gathering or model inference |
| Documentation | Method docstring and direct tests in `test_rfm.py`; no exact-API example found in tracked Markdown/notebooks |
| New client today | Materialized context contains some labels, but these are context samples tied to a prediction request |
| Cost of leaving it out | No public bounded label-extraction workflow; users duplicate target SQL/aggregation or misuse sampled context |

## Proposed API and boundary

Python: `query.labels(size=..., anchor_time=..., random_seed=..., max_iterations=10)` returns a DataFrame.
Java: `query.labels(schema, options)` returns an owned Arrow table that callers close.
The schema is explicit in Java, consistent with its existing query API.
No endpoint, credentials for model serving, `SDMClient`, or fitted session is needed; warehouse data access still requires connector credentials.

Rust owns eligibility, anchor selection, label aggregation and bounded entity selection because these must mean exactly the same thing as prediction targets.
Hosts only adapt options, execute existing provider requests, and expose the Arrow result.
Add a production operation using `sdm-engine` labeling and existing materialization/provider machinery; do not implement target arithmetic in Python/Java.

## Contract and checks

- Zero training/context rows, at most `size` output labels, and bounded work corresponding to `max_iterations * size` candidate trials.
- Preserve explicit timestamps and entity-time anchors; default historical anchors must account for the full forecast horizon, not accidentally include unknown future outcomes.
- Preserve `ENTITY`, `ANCHOR_TIMESTAMP`, `TARGET` output meaning and key/target types.
- Return typed empty output when no labels satisfy the target filter; retain errors for invalid schemas, times and unsupported query shapes.
- Cover scalar classification/regression, temporal aggregation, forecast and link targets through existing target semantics; refuse any unimplemented shape explicitly.
- Never call a model or gather neighboring feature tables merely to return labels.

The existing test-only labels mode is not the public solution: it has different controls and an explicit link-label restriction.
Reuse its underlying engine pieces, not its entry point.
Shared fixtures must compare Python/Java values to independently calculated answers, for memory and provider paths, including empty outcomes and temporal boundaries.
Warehouse label queries are required; preserve existing prediction sampling SQL, with regression checks demonstrating that separation.
GPU testing is unnecessary for this operation.
The cited old tests establish scalar/temporal label behavior; forecast/link output layouts still need explicit fixtures before claiming restoration parity for those shapes.

## Test standard: parity or an explained improvement

Use existing public-API tests and small shared fixtures, not a new migration comparison framework.
Be thorough about behaviors, edge cases and connector differences; use focused old-client comparisons where they resolve a real semantic question.
Before implementation, settle signatures, ownership and error contracts with coordinator review; finish with independent code review and affected tests.

For each supported target shape, calculate expected labels independently on tiny data, then compare old Python, new Python and Java.
Use all eligible entities for exact comparisons; test bounded/random subsets separately for size, membership, seed reproducibility and trial limits.
Cover classification/regression, temporal windows, entity-time anchors, default forecast-horizon adjustment, link/forecast layouts, nulls and empty outcomes.
Run the same fixtures through memory, SQLite/DuckDB and supported warehouse providers, retaining explicit connector limitations.
Assert no model transport calls or neighborhood feature gathering occur; inspect provider calls and preserve existing prediction-sampling behavior.
An old-client discrepancy is investigated against the query definition and expected labels, then kept as a documented corrected regression, not copied automatically.
These are public end-to-end data-to-label tests; no GPU endpoint is involved because this API never calls a model.

## Verified code anchors

Old source at `12955d397`: engine `rfm/rfm.py:1801` under `packages/structured-data-models-relational-engine/src/sdm_relational_engine/`.
Old tests: engine `tests/test_rfm.py:87,694,705,748` cover labels, empty results and invalid entity timing.
Current seams: `rust/crates/sdm-engine/src/labels.rs`, `sdm-core/src/materialize.rs`, `sdm-core/src/materialize/frames.rs` and the test-only dispatch in `sdm-core/src/test_ops.rs`.


## Reviewed implementation contract

Python `query.labels(size, *, anchor_time=None, random_seed=42, max_iterations=10)` returns a detached DataFrame.
Java `query.labels(schema, LabelsOptions)` returns an owned Arrow Table; callers close it.
Size is a nonnegative integer, iterations a positive integer, and their product is checked before allocation or provider sampling.
A zero size returns typed empty output without an entity-sampling request; schema metadata may still be needed.
Scalar targets retain scalar values; link targets return a list of identifiers per entity.

A forecast retains the old one-entity layout: for `FORECAST 3 TIMEFRAMES`, output has up to three rows for one selected entity at successive anchors, capped by `size`.
An explicit entity in PQL is honored; otherwise seeded selection chooses one eligible entity.
For example, daily anchors January 1, 2 and 3 produce three scalar targets for that entity, not three target columns or a vector target.
The default anchor accounts for the full horizon even when the row limit is smaller.
Static entity-time anchors are supported; temporal entity-time anchors receive a structured refusal instead of the old assertion failure.
No eligible entity remains an error, while eligible entities whose target filter yields no labels return typed empty output.

The production operation uses a separate orchestration module sharing the existing engine's target, time, eligibility and provider helpers.
It does not use the test-only labels operation, construct a model request, or gather a neighborhood.

Target validity matches the shared engine: SUM/COUNT produce zero for empty windows; AVG/MIN/MAX and link targets omit examples with no target observations.
The independent fixture includes a lower-bound event that must be excluded and an upper-bound event that must be included.
The old pandas executor confirms `(start, end]` windows and omission of empty LIST_DISTINCT groups.

Scalar output uses the existing target conversion, including its value-dependent Decimal classification rule: represent colliding numeric classes as distinct strings.
A zero-row result uses that conversion on an empty column; it cannot detect observed Decimal class collisions without reading rows.
