<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 9: bring the two new main commits into Rust

**Status: complete and integrated into `rust-core-phase-9`.**
See [verification](verification.md) for results and limits.
Audited `rust-core` at `0b3afe8fa` against `main` at `4edabbf9c`.
Phase 8 is already integrated into `rust-core`.
At that audit baseline, their common ancestor was `847a1e9e3`; the two main commits below are now integrated.

## What changed

| Main commit | What users gain | What Phase 9 must do |
|---|---|---|
| `506e37024`, Jana, [#63](https://github.com/NVIDIA/structured-data-models-client/pull/63): column explanations | See which context columns are associated with the target, including cohorts such as numerical ranges and categories. | Port the analysis and its lifecycle into the Rust engine; expose it through Python and Java. |
| `4edabbf9c`, Adi, [#48](https://github.com/NVIDIA/structured-data-models-client/pull/48): Databricks Tabular quickstart | AdventureWorks classification/regression examples, endpoint setup, and direct-input/PQL usage. | Retain the notebook and deployment checks; verify examples against our client; retain clearer 16 MB request-limit guidance. |

The notebook uses existing APIs, including a single-table `Schema.from_data()`.
It does not introduce a new sampling algorithm or require Tabular `Schema.from_source()` support.

## Why the integration required review

A non-mutating trial merge reported **23 conflicts**, across runtime code, documentation, tests and the public API snapshot.
Main changed Python explanation code that the migration replaced with Rust and a different Python wrapper.
Git even suggests moving new explanation files into `in_process/`; that inferred rename is incorrect.
Resolving text conflicts alone would risk restoring the old Python engine or losing the new behavior.
Review automatically merged runtime files as well as conflicted files, and regenerate snapshots rather than merging their contents by hand.

## Column explanations: implemented boundary

These are descriptive statistics about labelled context, not causal explanations or model gradients.
Current Rust-backed results expose gradients directly as `result.explanations`.
Main now exposes `result.explanations.gradients` and `result.explanations.columns`, with columns convertible to a pandas table.
Adopt that explicit container: Python `result.explanations.gradients/columns`; Java `result.explanations().gradients()/columns()` using existing Java absence conventions.
This changes the current direct-gradient accessor, including Java's return type: update callers, docs and generated API snapshots together, and explicitly record the compatibility change.
Do not make the new container impersonate a gradient dictionary or duplicate statistical behavior in a compatibility wrapper.

```text
Python / Java: user inputs and connection handles
                  |
Rust: existing context preparation and sampling
                  |
Rust: column statistics on the correct labelled context
                  |
Existing inference transport -> Rust result decoding
                  |
Python / Java: result.explanations
                  +-- gradients: model explanation
                  +-- columns: context statistics
```

| Owner | Responsibility |
|---|---|
| Rust `sdm-request` | Column analysis beside existing gradient interpretation: cohorts, relational hops, scores and deterministic ordering. No new crate. |
| Rust `sdm-core` | Select the correct context view and preserve analysis through fit, prediction, forecasting and result merging. |
| Rust `sdm-contract` | Typed analysis output and versioned operation/result attachments; explicit missing-value and ordering rules. |
| Python / Java | Public containers, dataframe/Arrow presentation, transport and retention of immutable analysis alongside existing fitted-session state. No statistical implementation or sorting callbacks. |

Keep the existing host-owned fitted-session metadata arrangement; do not introduce a Rust global session registry.
This does not remove existing native graph ownership: `frames.rs` already keeps graph state with its native owner.
Compute from already available sampled context, without changing sampling SQL or adding warehouse queries.
Retain the small analysis result for fitted sessions, not an additional full copy of the context.

### What crosses the language boundary

Rust owns these rules because they interpret task semantics, context eligibility and explanation results, not merely because two languages need them.
Python/Java still run the existing I/O loop; Rust decides eligibility, holdout, scoring and merge semantics.
The engine boundary is prediction semantics: which evidence belongs to an explanation and what that explanation means.
Generic container construction, dataframe formatting, display labels, warning emission and notebook deployment helpers remain in the hosts, even when that means small Python/Java implementations.
Do not create native operations for trivial setters, getters or formatting helpers.
Validate engine-produced explanation data within its existing operation; do not add an engine round trip solely to construct a public wrapper.

| Boundary | Contract |
|---|---|
| Context into analysis | Reuse existing typed task/schema metadata and Arrow/native attachments. Rust selects eligible rows, keys, hops and forecast identity. No pandas objects, Java collections of data rows, callbacks, or new JSON copy of the full context. |
| Analysis out of Rust | A versioned descriptor plus typed rows, using existing attachment conventions. Define field types, nullable values, class-label representation and row ordering once; hosts only render them. |
| Fit to later prediction | Host retains the immutable analysis attachment and diagnostic with that fitted instance; Rust associates it with the accepted fit context. No recomputation on scoring rows or reuse across unrelated fits. Release retained resources on close. |
| Result merging | Internal batches share one request-level analysis. Rust validates selection; never sum scores or silently take the first analysis from unrelated requests. Specify explicit supplied-analysis behavior for public `PredictionResult.merge`, including main's `columns` argument. |
| Failures | Distinguish not requested, ready-but-empty, unsupported, and analysis failure internally. Hosts format Rust diagnostics as language-native warnings; public absence behavior follows main. Invalid protocol data remains an error, not a swallowed analysis warning. |

Python `ColumnExplanations(payload)` and Java construction accept raw analysis input and invoke the Rust analysis operation once.
Wrapping an already computed result performs no native call; host views only format rows.
Java exposes equivalent typed data and its existing Arrow presentation conventions, without copying Python-specific dictionary APIs.
Presentation must not regroup, rank or recalculate scores.
Do not promise zero-copy: numerical analysis needs working memory, but avoid repeated host/native conversion and retain only the resulting analysis for reuse.

Existing `session.rs`, `frame_session.rs`, `result.rs` and contract version 71 carry this lifecycle.
Materialized request bytes retain a versioned `client_analysis_delta`: only leakage-removed rows, their original positions and schema binding.
Rust validates and restores exact context order for analysis, including after request save/load.
The delta is excluded from model payloads and byte limits; no second host dataframe or warehouse query is required.
Fit deliberately uses its actual post-holdout context and never restores removed labels.
Ordinary batches share one analysis; forecasts combine step contexts with distinct instance identities before computing cohorts.
Failed statistics warn without breaking inference; corrupt state and conflicting public merge inputs remain errors.

## Behaviors that must survive the port

| Case | Required behavior from main |
|---|---|
| One-shot prediction | Compute once per request when explanations are requested, before batching; attach the same analysis to the merged result. |
| Relational context | Use labelled context before inference leakage trimming; preserve entity/hop identity. |
| Fit then predict | Compute after holdout on the pinned fit context, even if fit did not request explanations; later scoring may request them. |
| Forecast | Combine labelled context across steps using `(step, instance_id)` identity; handle related tables absent from some steps. |
| Empty or partially failed prediction | Preserve the analysis through result assembly; do not silently use only the first successful batch. |
| Analysis failure | Prediction remains usable; surface the analysis warning when explanations are requested. |

Match the intentional numerical rules: regression targets or up to 30 classification classes; numerical quantile cohorts; categorical/multicategorical groups; anchored timestamp cohorts; relational counts and hops.
Preserve missing/rare-category handling, sample-size correction in scores, normalization and stable ties.
Output columns are `table, hop, column, stype, cohort, class, share, mean, score`.
Ranking explanation support does not expand as part of this port.
Both models and both languages should use the same engine behavior wherever the existing request supplies supported labelled context, independent of its connector.

## Implementation and verification

1. **Lock the API and context contract first.** Trace current materialize/fit/forecast operations to identify where pre-trim and post-holdout views exist.
   Specify result containers, attachment lifetime, warning behavior and Java equivalents; review these before implementation.
2. **Port, then reconcile main.** Implement analysis and lifecycle in Rust with thin host wrappers, merge the audited main commits on `rust-core-phase-9`, and adapt the notebook/docs/tests.
   Reject legacy decoder/statistics paths and the inferred `in_process` moves; preserve existing warehouse sampling.
3. **Validate the changed behavior.** Use hand-calculated Rust fixtures for scores, missing/constant values, category limits, timestamp anchors and ties, with selected main comparisons as supporting evidence.
   Test Python/Java result parity, both models and classification/regression, late EXPLAIN after fit, holdout, forecast identity, batching and failures.
   Exercise in-memory and warehouse-derived context through public APIs; assert unchanged sampling queries and use bounded CPU endpoint tests for result integration.
   Test notebook deployment decisions with a mocked SDK and its prediction examples against the candidate client; ordinary CI must not deploy a billed GPU endpoint.

Update the existing compatibility manifest, per-symbol behavior links, generated API/wire snapshots and user documentation.
Reuse maintained tests and CI machinery; do not rebuild migration comparison tooling or repeat unrelated performance work.
Finish with main as an ancestor of Phase 9, no restored legacy runtime routes, and passing affected Rust/Python/Java and repository contract gates.
Record exact tests and artifact revisions; Phase 8 evidence does not automatically validate this new functionality.
CI parallelization improvements remain deferred until after merging `rust-core` into `main`, as previously agreed.
