<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# File-by-file merge disposition

**All incoming files are resolved, verified and integrated into `rust-core` in merge commit `711968587`.**
See [verification](verification.md) for passing checks and limits.
Baseline: main `847a1e9e3`, Rust `d184fb269`, common ancestor `12955d397`.
`C/` means `packages/structured-data-models-client/src/sdm_client/`; `T/` means that package's `tests/` directory.
“Absorbed” means Git proposes no change to that file, not that its behavior needs no testing.
Every retained public assertion must map to a current test or a reviewed difference during implementation.

| File | Git outcome | Resolution |
|---|---|---|
| `CHANGELOG.md` | Conflict | Combine actual feature entries; do not restore retired package claims. |
| `docs/about/architecture.md` | Conflict | Keep Rust engine/host ownership; add explanation flow. |
| `docs/reference/limits.md` | Conflict | Keep current limits; remove obsolete explanation restrictions. |
| `docs/reference/prediction-output.md` | Conflict | Document collections, restored manual construction and retained raw view. |
| `C/__init__.py` | Conflict | One GradientExplanations export from the approved Rust-backed view. |
| `C/adapters/relational.py` | Conflict | Retain host error translation; do not restore planning/answered/decoder policy. |
| `C/adapters/tabular.py` | Conflict | Retain host adapter duties; do not restore Python result decoding. |
| `C/client.py` | Conflict | Keep Rust dispatch; map EXPLAIN/session/forecast semantics to core tests. |
| `C/decompose.py` | Conflict | Keep Rust decomposition; port early ranking rejection into core. |
| `C/explanations/__init__.py` | Clean | Exclude this incoming package: it shadows the current module. |
| `C/explanations/converter.py` | Clean | Already ported as in_process/gradients.py; do not duplicate. |
| `C/explanations/decoder.py` | Clean | Do not ship the Python decoder; cover its valid contracts in Rust. |
| `C/explanations/explanations.py` | Clean | Retain idiomatic current view; port missing manual-result semantics through Rust. |
| `C/in_process/answers.py` | Absorbed | Keep current tensor/answer conversion; verify real-model tests. |
| `C/in_process/backend.py` | Conflict | Port no-load preflight; retain raw response handoff to Rust, not adapter.answered. |
| `C/in_process/relational.py` | Absorbed | Keep current runtime capability changes; verify supported tasks. |
| `C/in_process/tabular.py` | Absorbed | Keep current runtime capability changes; verify supported tasks. |
| `C/models.py` | Absorbed | Verify Python advertised outputs still match Rust capabilities. |
| `C/result.py` | Conflict | Restore explanations constructor and manual merging; preserve shared Rust policy. |
| `C/wire/answers.py` | Clean | Do not adopt duplicate gradient class-label policy; retain required existing host helpers. |
| `T/conftest.py` | Clean | Keep useful public fixtures; hand_built_result exposes missing constructor contract. |
| `T/test_a_model_runs_in_this_process.py` | Clean | Retain no-load rejection and supported runtime assertions. |
| `T/test_envelope_and_result.py` | Conflict | Retain result/forecast observations through current API and Rust results. |
| `T/test_every_backend_answers_alike.py` | Clean | Remove universal explanation refusal; assert backend-specific capabilities. |
| `T/test_in_process_explanations.py` | Clean | Keep tensor conversion tests; replace Python decoder/private adapter dependencies. |
| `T/test_limits_and_absent_outputs.py` | Conflict | Reconcile legitimate limits and absence diagnostics, not old one-row restrictions. |
| `T/test_limits_and_sessions.py` | Conflict | Verify Rust session ownership and output admission. |
| `T/test_live_nim.py` | Clean | Retain live contracts; require an explicitly capable endpoint. |
| `T/test_live_relational.py` | Clean | Retain useful live assertions; inspect identities and generated-key behavior. |
| `T/test_relational_explanations.py` | Clean | Port manual-result, early-ranking and forecast assertions; replace _gather patches. |
| `T/test_request_and_dispatch.py` | Conflict | Verify current native dispatch/admission; do not reinstate adapter planning. |
| `T/test_tabular_explanations.py` | Clean | Port empty-output/public collection cases; remove direct Python decoder dependence. |
| `packages/structured-data-models-pql/src/predictive_query/query.py` | Conflict | Keep deleted; EXPLAIN parsing belongs to sdm-pql/core. |
| `qa/README.md` | Clean | Keep applicable probe instructions with actual endpoint capabilities. |
| `qa/probe_relational.py` | Clean | Keep public probes; verify they call installed Rust-backed client. |
| `qa/probe_tabular.py` | Clean | Keep public probes; verify new output/view assumptions. |
| `skills/relational/SKILL.md` | Conflict | Reconcile documented contracts and maintained skill assertions. |
| `skills/relational/pql.md` | Clean | Keep EXPLAIN examples consistent with Rust admission. |
| `skills/relational/references/beyond-one-answer.md` | Conflict | Retain forecast/ranking distinctions and endpoint capability limits. |
| `skills/tabular/SKILL.md` | Conflict | Reconcile Tabular explanation support and constructor/result API. |
| `skills/tabular/references/pql.md` | Clean | Retain valid EXPLAIN examples; test actual native route. |
| `tests/golden/public_api.json` | Conflict | Regenerate from the resolved public API; never hand-merge. |
| `tests/test_skills.py` | Clean | Retain useful assertions; do not assume every endpoint serves gradients. |

## Operation evidence to assert

Record operation names/actions, not full payload copies.
Select expectations by the actual host/input route; a Python object path and Java Arrow path need not use identical messages.

| Route | Current core operation anchors |
|---|---|
| Prepared Relational prediction | `predict` |
| Python direct response | `decode_direct_response`; permitted native-object assembly uses `layout_native_direct` |
| Frame/warehouse session | `frame_session` fit/predict/close actions |
| Python direct Tabular session | `prepare_tabular_fit`, `accept_tabular_session`, `prepare_tabular_score`, `send_tabular_session`, `close_tabular_session` as applicable |
| Java Arrow Tabular | `tabular_arrow` predict/fit/score/close actions |
| Result merging | Python `plan_result_merge`; Java `merge_results` |

Check these against `sdm-contract/src/ops.rs`, host dispatch and the actual journey; rejected calls must not perform later I/O.
A token native call followed by Python decoding must still fail the forbidden-path checks.

## Behavior anchors at the audited baseline

- Manual construction: main `T/test_relational_explanations.py::test_hand_built_*`; current `C/result.py::__init__` lacks the argument.
- Empty outputs: main `T/test_tabular_explanations.py::test_empty_outputs_still_refused`; current Rust `pql::explanation_outputs` appends before validation.
- No-load rejection: main `T/test_a_model_runs_in_this_process.py::test_an_output_the_model_does_not_produce_is_refused_before_it_loads`; current backend calls `_serving` first.
- Early ranking refusal: main `T/test_relational_explanations.py::test_ranking_refused_before_gathering`; current `T/test_gradient_explanations.py` only checks rejection at prediction.
- Generated-key concern: `sdm-request/src/payload.rs` creates keys; `sdm-request/src/gradients.rs` exposes primary-key values. Reproduce before fixing.
- Existing forecast warning: `sdm-core/src/predict.rs` calls `steps_unexplained`; add EXPLAIN/default-output coverage rather than restoring Python orchestration.

## Documentation resolution

The seven conflicting documentation files retain the Rust architecture and incorporate main's explanation contracts: collection keys, per-call decoding, related rows, hop views, forecast steps, optional local execution and removal of obsolete explanation limits.
`CHANGELOG.md`, `docs/about/architecture.md`, `docs/reference/{limits,prediction-output}.md`, both model skills and `skills/relational/references/beyond-one-answer.md` were resolved against both parents.
The clean PQL reference edits were retained with a correction: EXPLAIN adds an output rather than replacing ordinary outputs, and an explicit empty list stays invalid.
Main's raw-view deprecation was intentionally omitted; the approved raw compatibility view remains supported.
Claims that every NIM serves explanations were qualified by endpoint capability, and synthetic keys are distinguished from genuine source keys rather than hidden by name prefix.
No verified example output was edited or invented.
These resolutions still require the final implementation and skills checks; they are not completion evidence by themselves.

## Implemented source and test resolution

All 43 rows above were inspected, including clean merges and absorbed changes.
The source resolutions follow the table: no competing `explanations/` package, retired PQL distribution, Python decoder, adapter planning, or Python forecast decomposition was restored.
The four reproduced gaps now use the existing engine boundary: Rust output admission and early ranking rejection; Python optional-runtime preflight; Python manual containers with Rust identity and merge planning.
An actual sampled keyless table also reproduced exposed synthetic keys.
Its provenance is now explicit internal state, preserved across persistence and sessions and removed before model HTTP requests; genuine similarly named source keys remain visible.
Contract 70 describes that state and the typed manual-entry ownership records, rather than passing arbitrary Python identities through JSON.

| Incoming assertions | Maintained coverage / reviewed difference |
|---|---|
| `test_tabular_explanations.py` | Public predict/session/EXPLAIN assertions retained; decoder-dependent assertions now send mocked endpoint replies through real Rust decoding. Missing IDs, duplicate IDs, gradients and flat views retained. Empty views keep their columns. |
| `test_relational_explanations.py` | Related rows, keys, hops, output labels including booleans, split requests and all four manual-construction groups retained on current public requests. Added native datetime/bytes/tuple identities, existing collection metadata and nonconsecutive row positions. |
| Old `_gather`/decomposition monkeypatches | Replaced by actual schema materialization in `test_gradient_explanations.py` and `test_gradient_key_provenance.py`; forecast step identity, default EXPLAIN missing-step warnings and pre-read ranking refusal are asserted. One-class refusal identifies the source question rather than wire `TARGET`. |
| `test_in_process_explanations.py` | Tensor conversion and real local-runtime assertions retained; removed duplicate Python decoder tests. Runtime regression quantile test now captures the actual public request and decodes through Rust. |
| Limits/session/request/result tests | Retained current Rust dispatch and valid incoming observations. Removed obsolete one-row/1k-context explanation restrictions; explicit empty outputs remain invalid. Raw explanation compatibility remains supported. |
| `conftest.py`, backend/live tests, skills tests | Useful incoming public fixtures and capability assertions retained. Optional-runtime absence hides only optional `sdm`/Torch, not required PyArrow. Live skips are not counted as endpoint evidence. |
| QA probes | Retained as public probes; permissive refusal acceptance is not used as proof that gradient prediction succeeded. |

The public repeated-result merge also exposed mismatched frame and gradient positions.
Rust now preserves unique positions and resolves collisions consistently in both views, without mutating inputs; ambiguous duplicate positions within one explained input are rejected.
Ordinary unexplained merges retain their existing behavior.
This is a correctness improvement over both parents, covered by nested merges and both language bindings.
The public snapshot and contract schemas were regenerated with their maintained generators.
