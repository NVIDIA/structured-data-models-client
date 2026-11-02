<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 5 verification tracker

**Audit and reuse first; run only missing or invalidated checks, then required final-candidate acceptance.**
The Phase 4 merge alone changed no product bytes and does not invalidate its results.
Phase 5 removal/packaging changes need an explicit impact assessment before reusing evidence.

| Requirement | Existing proof | Remaining action / status |
|---|---|---|
| Formal contracts/differential | Phase 3 final `ff86cd1c` reviews: 600 generated local sources, 20 SQL and 20 Snowflake cases, 40 held-out cases and contract/provider checks, accepted under recorded owner dispositions | **Applicability audit:** map new or changed Phase 4/5 paths; retain approved differences and rerun affected/missing checks |
| Runtime coverage | Phase 3 CI run `40473118676`, attempt 2: 81 successes, two intentional skips; Phase 4 PQL: 6,490 x86 checks, 780 per Arm lane, AVX512 584 per lane | **Applicability audit:** reconcile declared platforms/runtime versions with changed bindings, dependencies and final packaging |
| Warehouse/model runtime | Phase 3: 21 profiles, 42 executions, 255 questions, 114 positive pairs/7,485 typed result rows and 141 matched endpoint refusals | **Applicability audit:** preserve producer/model IDs; these are prior measured results, not a universal current-build pass |
| Live parity | 582 historical comparisons plus 210 type/raw-reader comparisons; both languages compared directly to frozen Python | **Reuse candidate:** retain unchanged cells under the recorded review rule; rerun routes affected by runtime removal |
| Performance | Actual Phase 3 measurements and accepted costs; narrower Phase 4 ordering/session measurements | **Reconcile first:** apply exact accepted dispositions, benchmark new/affected paths; no blanket 10% target reset |
| Engine-free packaging | Existing clean wheel/JAR checks still include the pre-removal dependency layout | **Must run after changes:** clean install without old engine/Python PQL, public API and frozen pickle compatibility, supported native platforms |
| Historical refusal wording census | Frozen receipts, difference registry and original comparison tests preserved in the effort 2 archive | **Effort 3 applicability/reconciliation:** permanent template/public-refusal tests do not prove the final candidate matches every historical census probe |
| Final acceptance | Prior CI and historical artifacts are retained | **Must close:** required final-source CI and final-artifact gates, with approved evidence reuse explicitly recorded |

## Existing decisions are not new failures

Phase 3 acceptance preserved the raw held-out NO-GO findings (33 IDs / 82 legs); reviewed dispositions left one residual against a threshold of four.
It did not turn every raw observation green.

The archived Phase 3 decisions accept the measured Databricks ~2× latency, Snowflake 21–32% latency cost, native library ~16 MB versus 9.6 MB, and five small SQLite overhead cases.
Separate retained owner answers accept two additional exact SQLite timing rows and omit CPU timing only for one Spark divide-by-zero case, retaining correctness.
These are scoped dispositions, not acceptance of future regressions.
Keep raw failures visible beside their approvals.

The reuse decision requires two independent reviewers to show later changes cannot affect retained warehouse/model results, rerun affected cases, preserve original identities and require final green full CI.
Dedicated live-CI infrastructure is deferred; this does not erase the final-source checks or require waiting to run authorized local live tests.

## Authoritative receipts

Historical Phase 0–4 documents are preserved with SHA-256 checksums under `phase-5/effort-2/deletion-first-historical-designs/docs/design/rust-core/` in the same external archive.

Paths below are relative to `~/.local/share/sdm-orchestration/`; preserve them during disk cleanup.

| Evidence | Location |
|---|---|
| Phase 3 final acceptance and CI | `phase-3/final-ff86-source-acceptance.private.json`; `phase-3/final-ff86cd1c-full-ci-20261027T082304Z/final-CI-green-receipt.private.json` |
| Independent Phase 3 GO reviews | `codex-reports/T3.10-final-ff86-exit-broad-r1.md`; `codex-reports/T3.10-ff86-final-Phase3-focused-GO.md` |
| Additional timing/reuse decisions | `phase-3/owner-answer-20261027-additional-sqlite-costs.primary.json`; `phase-3/owner-answers-2026-10-26-reuse-and-spark-cpu.relay.json` |
| Phase 4 live results | `phase-4/coverage-audit/historical582/report.json`; `phase-4/type-parity/wire-comparison-complete.json` |
| Phase 4 runtime/benchmark scope | `phase-4/restart-20261030T014909Z/pql-final-completion-audit.md`; `session-benchmark-environment.json` and adjacent benchmark JSON files |

The session benchmark uses `9bda54ef1` and a constant-response local transport; it excludes network/model latency.
The older Phase 3 validation tracker is superseded by its final receipts where they disagree.
No new tests were run for this planning audit.
