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
| Distribution release | Python CI/release config covers five targets; Java packaging covers four; no Java publication in `publish.yml` | **Effort 3:** complete the [distribution matrix](effort-3/design.md), publish Python/Java artifacts and verify clean consumer installs; configured builds are not successful release evidence |
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

## Effort 3 work in progress

- Owner scope: five targets for Python and Java; Windows ARM64 deferred because required Arrow binaries are unavailable upstream.
- Distribution preparation committed as `d13d50c9d`: Windows x86-64 Java build/loading, complete JAR contents, version-derived artifact paths and JAR/POM source/hash checks.
- Independent review fixes verified: actual DLL fixture packaging, actual JAR license/notice contents and generated attribution including JNI.
- Focused distribution checks: 69 CI/release/attribution tests passed; final Java CI/artifact suite has 10 passing tests, including corrupted-artifact rejection.
- Main reconciliation is underway: local model execution stays behind Rust request/result handling; old Python adapters and engine remain removed.
- New local model suite: 82 passed, no failures; CPU random-weight models, not live-endpoint evidence.
- Shared Rust fix: transmitted diagnostic queries respect the server's 1024-character limit; original request text is preserved, including Unicode (focused Rust regression passed).
- API ledger passes with 159 Python and 307 Java symbols; Python-only model runtime applicability is explicit.
- Broad source suite: 6,154 passed and 269 skipped; four stale test expectations were corrected and one Java check rerun with the required JDK, with all five passing.
- Independent reconciliation review approved the code boundaries; required local-model API identities now run through the existing outcome checker (5 passed, no skips).
- Two independent reuse reviews are retained at `phase-5/effort-3/evidence-review-a.md` and `evidence-review-b.md` in the external archive.
  Both support retaining unchanged sampling/SQL proof, while requiring current checks for changed Python dispatch, numeric batching, DuckDB reads, Snowflake connections and long Unicode queries.
  The 582-cell report spans multiple builds; it and the 210-cell type report are historical evidence, not one final-build certificate.
- Remaining: actual platform CI, bounded affected live checks, original acceptance/performance reconciliation, release version, Java publication destination/access and publication verification.

Current candidate `418520620` is running in full CI `41490937592`.
Its clean installed wheel passed five memory prediction/session cells, 24 DuckDB cells (including expected refusals), and eight Snowflake/semantic-view cells.
Eleven additional live option/long-query checks passed; nine invalid-option cases exposed the early-rejection gap described in the Effort 3 design.
The wide 5,000-column public journey passed in 14.38 seconds process wall time with 202,016 KiB peak RSS; this is a current measurement, not a complete performance gate.
Commit `932fabaf9` records an independently reviewed non-applicable ZIP advisory in an excluded test-build code path; local dependency checks pass.

The final candidate is not yet accepted, built across all targets or published.
