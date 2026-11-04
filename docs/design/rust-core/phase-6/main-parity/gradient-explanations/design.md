<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Explain every prediction

**Gap:** main can explain each prediction; Phase 6 still exposes mainly a single raw explanation and rejects some supported requests.
This is the largest remaining product change.

## Approved engine and language boundaries

Main advanced to `847a1e9e3`: PR #62 makes `EXPLAIN PREDICT` request explanations, including forecast and fitted-session handling.
Rust must interpret that flag before decomposition, combine requested outputs and preserve question identity independently of the output prefix.
Do not implement separate Python/Java interpretations of PQL; retain main's refusal for explanations of client-assembled candidate rankings.

The model computes gradients; Rust interprets them, ties them to the exact input rows and assembles results.
The language-neutral engine result must preserve row position, typed entity ID, forecast step, explained outputs and per-table row/key/hop/gradient records.
Python dictionaries/pandas and Java views/Arrow are adapters over that contract, not competing decoders.
Malformed model output must fail consistently in Rust; old opaque explanation formats remain available without pretending they are validated gradients.
No extra warehouse reads or Python callbacks are needed for decoding.

PR #63 is still open, now at inspected head `f240fe448`; its `.gradients`/`.columns` container is not final.
If approved, column-cohort calculations and exact context/holdout/session reuse belong in the Rust engine because they define explanation semantics over the prediction request.
Tensor execution remains in the model service or optional Python model runtime; plotting, pandas/Arrow conversion and explicitly configured LLM prose remain language-side presentation/integration.

## Applicability (source-audited)

| Dimension | Scope |
|---|---|
| Models/tasks | Tabular classification/regression; Relational classification/regression and forecast result assembly. Do not infer working temporal-link/ranking gradients merely because the Relational model advertises both that task and the explanation output. Their combination needs explicit backend evidence. |
| Connectors | Source-independent once a request exists: in-memory, SQLite, DuckDB, PostgreSQL/Lakebase, Databricks SQL/Spark and Snowflake; file/S3 data after loading into supported frames. Existing semantic-view restrictions still apply. |
| Sampling | Both Relational in-memory and push-down-prepared requests benefit. Decoding uses the exact sampled payload; it changes neither neighbor selection nor SQL. Tabular uses in-memory rows, including rows read from a database first. |
| Languages/runtime | Python and Java endpoint clients; optional local model execution is Python-only. Local model task support must remain narrower where its reader supports only classification/regression; forecast decomposition is distinct from native forecasting support. |

This restores a shared result capability, not universal endpoint support or new connector/task combinations.
Main's local `in_process/{tabular,relational}.py` explicitly lists supported tasks; its model-wide output declarations alone are insufficient proof for every task/output pair.

## What is it for?

For a predicted customer outcome, show how model outputs respond to numerical features in that customer's row and related orders/events.
These are gradients in the model's preprocessed numerical space, not causal explanations or general feature-importance scores.

| Question | Finding |
|---|---|
| Did main support it? | Yes, `1c5e2e71d`: multi-row Tabular/Relational, batches, forecasts, fitted sessions and optional in-process execution. Earlier main had narrower explanations. |
| Was it documented? | Yes: `docs/reference/prediction-output.md`, both model skills, relational explanation examples and QA probes. No `explanations` example was found in notebooks at that commit. |
| What is missing here? | Tabular explanation admission, removal of obsolete relational one-row/fast-mode restrictions, decoding related rows, and a collection covering every result row. Existing Phase 6 `ExplanationView` is not that collection. |

## Proposed flow and ownership

```text
Python / Java: predict(outputs including explanation)
    → Rust: validate request and retain its row identities
    → host: call endpoint (or Python's optional local model)
    → Rust: decode gradients against THAT call's sampled rows
            merge batches / forecast steps without losing identities
    → Python / Java: expose explanation collection and table view
```

| Layer | Contract |
|---|---|
| `sdm-request`, coordinated by `sdm-core` | Validate/decode `sdm_gradient_v1`; preserve typed keys, output labels, row positions, forecast steps and relationship hop distances. Keep each result tied to its own request payload. |
| Python / Java API | Matching collection semantics with idiomatic lookup and pandas/Arrow presentation. Preserve the existing singular view and summarizer during the transition; do not copy main's already-expired “removed in 2.0” deadline. |
| Python in-process adapter | Execute gradients and convert model tensors into the response format. Rust does not gain a Python/PyTorch callback dependency. Java local Python-model execution is out of scope. |

For duplicate or missing entity IDs, use global result-row position, not a key that silently overwrites another explanation.
Forecast identity also includes the step.
Do not reread warehouses to decode results or resample data after prediction.
Share immutable request data where possible; avoid cloning every sampled table solely for explanation presentation.

## PR #63 affects this API

At inspected head `f240fe448`, the open PR changes `.explanations` to a container with `.gradients` and `.columns`.
The latter summarizes labelled context by column/cohort, computed once per prediction or after fit holdout, then reused by that session.
**Keep this pending, not approved scope.**
If merged, its context/holdout/forecast semantics belong alongside Rust request/session preparation; pandas/Arrow formatting stays in the hosts.
Review the final algorithm and memory lifetime before designing that addition; do not implement the mutable PR description as a contract.

## Prove it works, and improve on main

- Shared Rust cases: multiple rows, duplicate IDs, wide keys, forecasts, per-batch related rows, disconnected/multihop graphs and unknown legacy formats.
- Reject malformed shapes, negative/out-of-range row indices, nonfinite gradients and unknown table references with structured errors; do not copy permissive Python indexing accidents.
- Python and Java public tests: same explained rows and output order, collection/table views, fitted prediction and partial batch failures.
- One supported model/local-runtime E2E per model: classification and regression gradients explain the reported class order and mean/median/quantile outputs. Endpoint support must be verified, not inferred from client capability.

Evidence: main's `explanations/{decoder,converter,explanations}.py` and explanation tests; current `rust/crates/sdm-request/src/predict.rs`, Python `result.py`, Java `PredictionResult.java`.
