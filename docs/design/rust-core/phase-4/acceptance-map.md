<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 acceptance map

**Feature delivery is verified; the original slice/switch gates are not fully proved by the available completion evidence.**
This maps [migration-plan.md sections 6–8](../migration-plan.md#6-testing) to existing results as of `7449c19c0`.
It neither waives an original requirement nor starts new implementation/testing.
**Phase 5 audit update:** [earlier Phase 3 final receipts and accepted performance costs](../phase-5/verification.md) provide substantial prior proof.
The remaining question is applicability to changed Phase 4/5 code, not whether those original campaigns ever ran.
“Partial” means evidence exists for a narrower scope or earlier build; “unproven” means this reconciliation did not establish a passing gate.

## Original Phase 4 feature exits

| Requirement | Existing evidence | Assessment |
|---|---|---|
| Tabular requests no longer receive the migration refusal | Python/Java direct/PQL and session cases in both parity matrices | Verified within the agreed API/type scope |
| Client-defined text-time parsing no longer receives `text_time_rules` | `2d4021249`; `materialize.rs` handles `TextTimes::ClientRules`; provider-contract tests | Implemented with focused validation |
| Foreign-key `LIST_DISTINCT` temporal links | Warehouse/in-memory link and session comparisons; item identity correction | Verified for applicable matrix cases |
| Fallback optional operations, slice byte accounting and collation | `2d4021249`; `sdm-core/tests/fallback.rs`; manifest `provider-contract` | Implemented with focused validation |
| Large Arrow values | Actual >2 GiB omitted-binary prediction for both languages and frozen Python; large-text refusal tests | Verified bounded behavior, not unlimited feature support |
| No old computational policy in migrated paths | `query.py::_resolve`, shared frame edits and permanent `tests/parity/routes.py` tripwires | Verified for tested default journeys; old code is not deleted |

## Original test layers

| Layer | Evidence found | What it does not yet establish |
|---|---|---|
| L0: boundaries/contracts | Independent reviews, contract-lock/generation checks and strict Clippy in scoped deliveries | One final-candidate result covering every original dependency/unsafe/contract guard and version decision |
| L1: unit/property/fuzz | Rust/Python/Java regression suites and mismatch regressions passed on recorded builds | A consolidated final-candidate unit/property/fuzz gate |
| L2: full contract corpus | 582 historical parity comparisons, 210 type comparisons, API checks and comparator negative controls | The original full checkpoint-owned corpus/ratchet, all observation fields and approved difference register on one candidate |
| L3: randomized differential testing | Completed Phase 3 generated/held-out campaign with recorded owner dispositions; targeted Phase 4 comparisons | Applicability of the completed Phase 3 generated/held-out campaign to new Phase 4 paths; see the Phase 5 tracker |
| L4: SQL/provider kit | Provider regressions and real connector journeys; shared warehouse policy retained | Full SQL-golden/provider-conformance gate on the final candidate |
| L5: conversion/runtime matrix | Earlier six x86 Tabular PQL lanes, two Arm lanes, AVX512 evidence; later Java JDBC and 210 type comparisons | All declared runtime/driver conversion cells on the final candidate; selective carry-forward evidence stays selective |
| L6: live parity | Real source/model comparisons and identified session bridges across supported inputs | Entire final-build matrix plus the original Databricks job/runtime sweep; bridge tests do not establish public serving-session support |
| L7: performance | Phase 3 warehouse measurements and owner-accepted costs; narrow Phase 4 ordering/session benchmarks | Applicability of existing measurements and approved costs to later changes, plus new/missing measurements; prior accepted costs are not reopened |
| L8: packaging/safety | Clean installed wheel/JAR artifacts, ownership/lifecycle tests and earlier platform checks | Full declared OS/architecture × JDK17/21 matrix and sanitizer/crash/leak acceptance; full platform gate belongs before epic merge |

The new permanent parity harness is valuable additional coverage, but no reviewed blanket replacement of L0–L8 was established here.
In particular, passing predictions cannot certify performance, randomized coverage or platform safety.

## What blocks which milestone?

| Milestone | Current status |
|---|---|
| Requested feature/parity initiatives | Complete within their recorded scope; [current audit](coverage-audit.md) reconciles the old findings |
| Original phase-branch landing gate | Scoped checks passed; the prescribed full CI/ratchet evidence is not consolidated here |
| Original slice gate / merge into `rust-core` | **Unproven:** full L2/L3/checkpoint evidence and difference accounting; required `main` reconciliation/reference rerun and final fresh review also need recording |
| Original Python switch gate | **Unproven:** slice closure, final live/runtime sweep and L7; current Rust routing alone does not prove this gate |
| Dedicated scheduled live CI | **Explicitly owner-deferred:** credentials and runner setup; [CI status](parity-ci.md) |
| Epic merge to `main` / release | Later scope: final-build live/runtime checks, full L8, existing suites and Phase 5 old-engine removal |

The documentation reconciliation stopped before integration as requested.
The owner subsequently authorized the merge into `rust-core`, completed at `2b9e6ea88`; the merge tree matches the Phase 4 tip.
That authorization did not relabel unproven gates as passed.

## Evidence locations and provenance

- [Historical matrix](parity-matrix.md): 536 original cells on `7f7c19f95`, then 46 Java Snowflake cells on `3ec35527d`.
- [Type matrix](type-parity-matrix.md): 210 cells on product artifact `285c5c053`; [results and limits](type-parity-status.md).
- [Permanent manifest](../../../../tests/parity/manifest.json): API mappings, provider/large-value gates and scenario definitions; declarations alone are not execution evidence.
- Local orchestration evidence under `phase-4/`: `coverage-audit/historical582/report.json`, `coverage-audit/result-items-api56.json`, and `type-parity/wire-comparison-complete.json`.
- Earlier scoped evidence: `restart-20261030T014909Z/pql-final-completion-audit.md` and `final-static-gates-r1.md`; their exact commits and runtime limits remain controlling.
- Narrow benchmark records in the same restart directory: `session-benchmark-{core,frozen}-final.json` and `pql-pure-benchmark-{core,old}.json`; these do not certify final-build L7 limits.

The final type report leaves historical cells outside its selected scope; the separate historical report proves those 582 cells.
Combining their counts is historical accounting, not a single-build release certificate.
