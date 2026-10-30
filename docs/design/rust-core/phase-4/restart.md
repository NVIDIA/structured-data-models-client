<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 restart: tabular, then relational

The restart base is the completed Phase 3 commit `38071e0fb3e5f89c9ca5b74c72bec55ba32c6190`.
The first delivery target is the complete supported tabular client journey.
Relational in-memory graph construction and sampling follow tabular.
Existing fit, fitted prediction, close, transports and Python public objects already exist and are reused.
Moving their remaining semantics into the core is migration underneath that API, not a new warehouse feature or a prerequisite to one-shot tabular prediction.

## Concrete causes being corrected

1. The initial tabular selection composition sorted source positions under a RangeIndex restriction, while the general contract requires sorting actual sampled index labels.
   A later correction supplied candidate labels, but duplicate labels remained unfinished.
   The missing contract was the composition of source identity, candidate positions, sample positions and index labels, not the random draw itself.
2. Scalar and SIMD index sorting kernels were developed without a production profile selector or a connected public sampled-context consumer.
   Component permutations did not prove the final model context order.
3. Python assembled selection policy through separate key, population, labelled-row and context binding objects.
   Ordinary typed inputs should execute that policy through one core composition, with pure selection methods internal to Rust.
4. The same unfinished topic absorbed constructor, task, envelope, response-owner and unrelated frame/warehouse observation corrections.
   A first tabular call does not require a relational graph, continuing neighbor RNG, SQL seed-binding comparison or warehouse publication framework.
5. A new branch and worktree were created for many review corrections.
   One tabular topic now retains correction commits and exact-source review history until its connected increments land.

The old relational CSC builder sorts adjacency stably.
Its neighbor draw is a separate MT19937 modulo/without-replacement algorithm.
Tabular sampled-context index sorting can have unstable ties and is a different obligation.
These algorithms must not be conflated or forced through one universal sampler.

## Ownership and public boundaries

```text
Direct Request --------------------------+
                                         v
Tabular PQL -> select context/score rows -> model-neutral request
                                         |
                                         v
                            tabular bind / encode / decode
                                         |
                        host transport and native presentation
```

Python owns native objects, conversion at actually reached consumption points, I/O and presentation.
Rust owns generic request validity, schema meaning, model policy, selection, encoded model layout and response attribution.
The public Request remains model-neutral, retains its original frame references and can be inspected or reused with a later model.
A stored description must never override later observed edits to its public task, schema or frames.

| Boundary | Input and output | Owning component |
| --- | --- | --- |
| Generic shape validation | Ordered frame headers, row counts and actual task/target/argument facts produce a checked generic shape or the existing diagnostic. | `sdm-request::direct`, composed by `sdm-core`. |
| Schema and feature validation | Checked shape, held schema and reached column/value facts produce a direct description preserving task, target, identity and ordered feature meaning. | Existing `sdm-schema` policies plus `sdm-request::direct`. |
| Model preflight | Current generic request capabilities, model and default/explicit outputs produce a bound call or the existing refusal. | Shared `sdm-request::predict` policy. |
| Tabular encoding | Bound call and reached current context/scoring columns produce one ordered model schema, body and reply-attribution layout. | `sdm-request::tabular` and existing `sdm-wire`. |
| PQL row selection | Typed source/scoring entities, target validity, source row identities and requested count produce scoring rows and final context rows. | `sdm-engine::tabular`, composed by `sdm-core`. |
| Response | Raw reply plus its exact encoded-call layout produce attributed result rows and presentation ownership selections. | Existing wire decoder and shared request/result code; host retains actual native owners. |

These are domain interfaces, not a serialized program counter or a generic native-action interpreter.
Shape checks must remain callable before schema or feature conversion that the reference has not reached.
The consumed native observations and substantive exception order are specified with each connected input family before freezing its concrete fields.
The first implementation must not add all possible native stages in anticipation of future inputs.

## Direct preparation and prediction contract

Direct construction performs generic checks immediately, before a model is known.
Preserve the existing related-table, task, target-container, empty-context, target-presence, known-target and scoring-column guards in their observed order.
Use the existing held-schema vocabulary for adopted or inferred schema and subsequent feature compatibility.
Do not force inference or task-option values through JSON before their source consumption point.

At prediction, preserve model/output/shape/task/capacity checks before the empty-scoring no-send result.
An explicit empty output selection is different from default outputs.
An oversized direct context is refused at the model boundary; it is never silently sampled.
A generic task accepted by Request may legitimately be refused by a selected model later.

The model builder computes one wire dtype for each column across both halves and uses that same target dtype in the task and model schema.
Preserve column order, exact numeric width, null versus present NaN and typed-empty versus untyped values.
Native conversion facts must describe actual storage or completed observations, not host-selected semantic types.
Column-sized metadata, data and identities use existing byte/Arrow attachments rather than an unbounded control envelope.

The existing Predict input is extended with a closed direct-input alternative while preserving existing gathered state/held requests.
Mixed or incomplete input alternatives are refused.
Bindings call `sdm-host`; `sdm-core` composes the domain methods and existing model-call run.
No new crate, callback framework, warehouse provider or direct binding-to-policy dependency is introduced.

Keep live native frame/dtype owners in the host at their reference lifetimes.
After model bytes are emitted, response attribution uses the row/key layout bound to those bytes, not a later reread of mutated caller frames.
Validate missing/out-of-range/duplicate response rows, recognized wrong keys and unanswered rows in source order.
Reuse the existing native parsing/conversion boundary where required and preserve selected public field ownership and aliases.
Do not invoke the whole legacy adapter normalization path as a substitute for core policy.

## PQL context selection contract

The indices path expands every requested key, including repetitions, to all matching source positions and preserves request order.
It excludes the selected source positions and unknown-target rows from context.
The supplied-frame path excludes matching scoring entity keys, then unknown-target rows; it is not positional exclusion.
Task inference consumes the final selected context target unless the caller supplied an explicit task.
Both paths construct the same model-neutral request consumed by direct prediction.

Sampling is conditional:

1. Validate the requested count and derive the cap.
2. If candidates already fit, return them in candidate order without sorting or consuming index labels.
3. Otherwise apply the reference's fresh legacy RandomState seed-42 sample to candidate positions.
4. Sort that sampled subset using its actual index semantics and map back to original source rows.

`IndexFacts` separates row identity from labels and represents the consumed ordinary index family, values, missingness and structural data.
Categorical facts include category order and codes; MultiIndex facts include levels, codes and order structure.
Floating facts preserve required bits and native missingness.
The sort profile is closed and established from the actual supported producer; it is not an arbitrary CPU guess or silent fallback.
A globally precomputed permutation or stable source-row tiebreaker is not equivalent to the reference's selected-subset sorting.
Pure sort methods remain library components consumed by this operation, not separate deliveries without a caller.

The existing approved exception for custom sorting-method overrides remains precise.
It does not automatically cover custom objects stored inside an otherwise ordinary Index.
If a family requires inherently native comparison effects that conflict with the no-native-sorting-callback boundary, produce a public counterexample and a concrete owner decision before claiming that family implemented.
No broader callback, behavior change or Python fallback is approved by this plan.

## Delivery and acceptance

The tabular target includes direct calls, PQL indices and supplied scoring rows, supported native/index families, fitted use, file-fed calls, result inspection and applicable live acceptance.
A numeric prototype or an isolated sorting kernel does not complete that target.

1. Connect and land direct construction, tabular model binding, encoded request and response through both hosts.
   Use one small ordinary-family pilot to verify the composition, then extend the same interfaces across the supported direct families.
   A partial development route must remain explicitly partial and preserve existing defaults outside its admitted domain.
2. Connect both PQL entrypoints through that same request builder and close filtering, sample-before-sort and the supported index/runtime matrix.
   Complete or explicitly resolve each accepted family's ordinary, empty, null, duplicate, mixed, mutation and substantive-failure cases.
3. Reuse the existing fitted lifecycle and transport with the same tabular layout and pinned/scoring sections, then close tabular file/result/state and live evidence.
   Finish tabular before starting fresh relational graph/sampling implementation.

Each connected increment requires independent exact-source architecture and compatibility reviews before checks, then the required package checks and immediate landing.
Generated contracts and fingerprints use their canonical writers and receive completed-source review.
Do not inherit execution credit from archived candidates.
Keep one integration branch, one tabular topic and at most one genuinely independent companion.
No branch or worktree is created merely because a review round changed.

For every claimed tabular family, record the frozen-reference public case, actual current-source/native producer, selected source rows where applicable, raw model request/reply, result and decisive failure/ownership controls.
Maintain one finite acceptance table and one execution record per immutable candidate.
Preserve existing Phase 3 regressions, separate Python suite collection, heavy guard, Cargo jobs=2, nightly CI and phase-end acceptance.
General pandas warning differences remain governed by the owner's existing direction and must not become a new delivery campaign.
The final default switch and full Phase 4 exit requirements remain separate from an increment's landing.
