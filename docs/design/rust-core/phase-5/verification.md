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
| Distribution release | Python and Java build configuration covers five targets; Java JAR/POM release assets are configured, but repository publication is unresolved | **Effort 3:** complete the [distribution matrix](effort-3/design.md), publish Python/Java artifacts and verify clean consumer installs; configured builds are not successful release evidence |
| Final acceptance | Prior CI and historical artifacts are retained | **Must close:** required final-source CI and final-artifact gates, with approved evidence reuse explicitly recorded |

## Original acceptance requirements: concrete closure work

Audited against archived `migration-plan.md` §6–8 and `phase-4/acceptance-map.md`; the old labels below identify requirements, not extra projects.

| Gate | What remains to close it |
|---|---|
| L0: boundaries | Final CI dependency/unsafe scans, Clippy, contract lock and dependency checks; retain reviewed contract-version decisions |
| L1: algorithms/decoders | Final Rust suites, including retained PQL, contract and YAML/view fuzz tests; decoder fuzz alone does not establish algorithm parity |
| L2: public contracts | Final maintained API/outcome ledger and refusal tests; explicitly reconcile historical observation fields and approved differences rather than claiming the old corpus was replayed |
| L3: randomized cases | Carry forward reviewed Phase 3 generated/held-out results for unchanged algorithms; identify new Phase 4 paths not covered by that campaign |
| L4: SQL/providers | Final SQLite/DuckDB/Postgres golden and provider jobs; previous Postgres jobs used retired selectors, now repaired |
| L5: conversions | Final runtime/driver jobs, including repaired DBR16.4 Arrow conversion and unresolved NumPy platform ordering |
| L6: live predictions | Retain identified historical cells; finish affected installed-client journeys, including ranking/forecast, and reconcile the Databricks runtime sweep separately |
| L7: performance | Preserve exact approved historical costs; measure changed numeric preparation and session paths, with baseline identities and memory scope |
| L8: artifacts/safety | Final five-platform wheel/JAR acceptance, minimum-system-library checks and explicit sanitizer/crash/leak evidence; Java loading alone cannot establish these |

At source `cc44e8dd8`, the retained contract-decoder property suite (6 tests) and core PQL/view fuzz suites (1 + 5 tests) passed with the pinned toolchain and warm build.
Logs: external `phase-5/effort-3/current-contract-fuzz.log` and `current-core-fuzz.log`.
Current CI contains boundary guards and bounded decoder fuzz tests, but this audit found no sanitizer job in the workflow.
Locate applicable historical safety evidence or execute the missing check; absence from CI does not prove a product defect or waive the requirement.
Do not rebuild retired migration tooling merely to preserve its command names.

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
- Main reconciliation is committed: local model execution stays behind Rust request/result handling; old Python adapters and engine remain removed.
- New local model suite: 82 passed, no failures; CPU random-weight models, not live-endpoint evidence.
- Shared Rust fix: transmitted diagnostic queries respect the server's 1024-character limit; original request text is preserved, including Unicode (focused Rust regression passed).
- API ledger passes with 159 Python and 307 Java symbols; Python-only model runtime applicability is explicit.
- Broad source suite: 6,154 passed and 269 skipped; four stale test expectations were corrected and one Java check rerun with the required JDK, with all five passing.
- Independent reconciliation review approved the code boundaries; required local-model API identities now run through the existing outcome checker (5 passed, no skips).
- Two independent reuse reviews are retained at `phase-5/effort-3/evidence-review-a.md` and `evidence-review-b.md` in the external archive.
  Both support retaining unchanged sampling/SQL proof, while requiring current checks for changed Python dispatch, numeric batching, DuckDB reads, Snowflake connections and long Unicode queries.
  The 582-cell report spans multiple builds; it and the 210-cell type report are historical evidence, not one final-build certificate.
- Remaining: actual platform CI, bounded affected live checks, original acceptance/performance reconciliation, release version, Java publication destination/access and publication verification.

Full CI `41490937592` completed with failures on source `418520620`; current fixes are committed in `3b443c78a` and still require final CI.
The failed run nevertheless verified packaged Java loading and public Arrow `Schema.fromData` on all five required platforms (JDK 21 everywhere; JDK 17 additionally on Linux x86-64 and Windows x86-64).
Its logs report contract 56, so they do not certify the current contract 57 candidate.
Python wheel builds passed on both Linux and macOS architectures; Windows failed, and installed/runtime acceptance was not green.
Job metadata, artifact identities and seven Java load logs are retained at `phase-5/effort-3/ci-41490937592-platform-audit/` in the external archive.
Its clean installed wheel passed five memory prediction/session cells, 24 DuckDB cells (including expected refusals), and eight Snowflake/semantic-view cells.
The early-rejection gap is fixed in shared Rust admission (contract 57): all 20 live option/long-query checks now pass, including assertions that invalid options never reach the transport.
Related local checks passed: 145 Python tests, two Rust admission tests, 22 Rust prediction/session tests, and the Java prediction/fit/session admission regression.
The Python custom-transport path still preserves exotic non-JSON values; this is not a claim of universal option normalization.
The wide 5,000-column public journey passed in 14.38 seconds process wall time with 202,016 KiB peak RSS; this is a current measurement, not a complete performance gate.
Commit `932fabaf9` records an independently reviewed non-applicable ZIP advisory in an excluded test-build code path; local dependency checks pass.

CI exposed additional compatibility and test-configuration gaps.
The DBR16.4 Arrow string-storage mismatch is reproduced and fixed, with null preservation verified in its exact environment.
Stale wheel-smoke package references, Windows log decoding, missing timezone data, retired CI selectors and obsolete test expectations are repaired; 447 related checks passed and the final stale assertion passed separately.
NumPy producer admission remains incomplete across supported Python/platform combinations; the bounded ordering proposal in the design awaits an owner decision.

The final candidate is not yet accepted, built across all targets or published.
