<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4: what landed

**Completed and pushed to `rust-core-phase-4` at `8eafd69e5`.**
Java landed first; Relational and the final integration followed.
This completed the agreed Java Tabular plus Python/shared-Rust Relational delivery, not Java in-memory Relational support.

## Delivered

| Area | What users get |
|---|---|
| Java Tabular | Arrow Table input, direct prediction, single-table PQL, and `fit → predict → close` sessions |
| Python Relational | Existing API, with Rust preparing examples, labels and related rows, then supporting prediction and sessions |
| Default behavior | Implemented in-memory and Tabular paths now use Rust by default |
| Warehouses | Existing SQL push-down sampling and provider behavior preserved |

`materialize()` prepares model input; `fit()` uploads reusable examples to a supporting server, rather than training a model.

## Important decisions

| Decision | Reason |
|---|---|
| Rust owns sampling and model-data rules; Python/Java handle input conversion, HTTP and returned objects | Shared behavior across languages, without sampling callbacks into Python |
| Retain graph and random state together; snapshot inputs | Reuse preparation safely and prevent later caller mutations from changing requests |
| Java exposes one Arrow Table input and copies it into owned storage | Simple API with safe lifetime; this is **not zero-copy** |
| Keep original related-table row IDs instead of legacy synthetic IDs | Preserve source identity; comparisons confirmed equivalent sampled rows and answers |
| Preserve meaningful behavior, not incidental old bugs or warning wording | Correctness takes priority over mechanically copying the old implementation |

## Tests and reviews

- **Python integration:** 7,970 passed, 137 skipped; all 13 failures were fixed and passed together on rerun.
  Final installed-package schema/release checks: 103 passed, one skipped.
- **Engine/Rust:** 2,573 legacy-engine tests passed; 190 targeted Rust tests passed, covering sampling, temporal labels, row identity, payloads and sessions.
- **Java:** 258 tests, zero failures, 13 gated skips; eight CI configuration checks passed.
- **Live comparisons:** old/new scalar and temporal-link predictions matched, including fit, repeated scoring and close; Java also matched frozen Python.
- **Quality:** independent architecture/compatibility reviews, lint, formatting, types, contract compatibility and generated-file checks passed.

The full Python suite was not repeated after repairs; affected tests were rerun.
Live comparisons preceded the final diagnostic/advisory fixes, which were checked locally.

## Anything you need to decide?

The bounded implementation is delivered, but the broader Phase 4 scope needs reconciliation as listed below.
Other points worth knowing:

1. **Performance tradeoff:** one 60,000-row workload improved repeat preparation from 22.7 to 3.7 ms, but first use rose from 137.9 to 149.4 ms and peak memory from 136.9 to 150.7 MiB.
   Broader optimization is follow-up work, not a demonstrated universal speedup.
2. **Coverage limit:** nine defensive Rust error branches lack measured direct branch coverage; nearby tests are not claimed as exact hits.
3. **Separate extension:** direct Python Arrow input was outside this delivery; Java Arrow support does not imply that Python API exists.
4. **Java Relational gap:** Java's in-memory `Schema.fromData()` accepts exactly one table and routes through Tabular operations.
   Java warehouse Relational support exists, but exposing the new multi-table Rust sampler through Java still requires API/JNI integration and end-to-end tests.

Design details: [Java Tabular](java-tabular.md) and [in-memory Relational](relational-in-memory.md).

## Is Java Relational the last remaining item?

**Not an unconditional yes.**
The restart explicitly preserved warehouse paths, while the original migration plan assigned some of their remaining work to Phase 4.

| Item | Verified status |
|---|---|
| Java in-memory Relational | Missing: multi-table input, connection to the Rust sampler, prediction/session integration and tests |
| Python warehouse sessions | Public API exists, but warehouse requests still use Python session preparation/handling; only Tabular and in-memory Relational select the new Rust session operations |
| Warehouse temporal-link targets | The warehouse `LIST_DISTINCT` refusal remains; the new frame path supports it |
| Phase-wide acceptance | Local tests and selected live journeys passed; these do not establish every original platform/performance gate, and changes are on `rust-core-phase-4`, not yet integrated into `rust-core` |

The warehouse items are preserved behavior under the approved restart, not newly introduced regressions.
They need an explicit recorded deferral or completion before claiming the original full Phase 4 migration is finished.
Java warehouse sessions are also absent: Java `fit()` currently accepts only in-memory Tabular requests.
That is a separate Java scope extension, not part of the original Python-only Phase 4 promise.

Source checks: [Java input](../../../../java/src/main/java/com/nvidia/sdm/Schema.java), [Java sessions](../../../../java/src/main/java/com/nvidia/sdm/SdmClient.java), [Python session dispatch](../../../../packages/structured-data-models-client/src/sdm_client/client.py), [warehouse/frame materialization](../../../../rust/crates/sdm-core/src/materialize.rs), and [original phase exits](../migration-plan.md).
Old-engine removal and release packaging remain Phase 5; direct Python Arrow input remains a separate extension.

Correction after tracing provider declarations: ordinary warehouse text timestamps are supported.
The surviving `text_time_rules` refusal belongs to an unused/custom-provider mode, not a general warehouse gap.
See the [remaining-work design](remaining-work-design.md) for the proposal and audit sources.
