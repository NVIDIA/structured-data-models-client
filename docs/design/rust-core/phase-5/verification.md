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
| Distribution preparation | Five Python/Java targets; Windows ARM64 deferred; RC version selected; initial Java destination is GitHub Releases | **Effort 3:** verify the [distribution matrix](effort-3/design.md) and clean local installs; publication is explicitly on hold |
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
| L5: conversions | Final runtime/driver jobs, including repaired DBR16.4 Arrow conversion and approved portable Rust ordering |
| L6: live predictions | Retain identified historical cells; finish affected installed-client journeys, including ranking/forecast, and reconcile the Databricks runtime sweep separately |
| L7: performance | Preserve exact approved historical costs; measure changed numeric preparation and session paths, with baseline identities and memory scope |
| L8: artifacts/safety | Final five-platform wheel/JAR acceptance, minimum-system-library checks and explicit sanitizer/crash/leak evidence; Java loading alone cannot establish these |

At source `cc44e8dd8`, the retained contract-decoder property suite (6 tests) and core PQL/view fuzz suites (1 + 5 tests) passed with the pinned toolchain and warm build.
Logs: external `phase-5/effort-3/current-contract-fuzz.log` and `current-core-fuzz.log`.
The deferred host sanitizer check is now complete locally: 128 tests across 18 executables passed with AddressSanitizer and Linux leak detection, no ignored tests or sanitizer diagnostics.
It used `nightly-2026-07-15`, rebuilt the standard library, and tested unchanged Rust source from `59addf5cc`; command, lockfile/log hashes and exit status are recorded in external `phase-5/effort-3/host-asan-state.json`.
The newer nightly failed to compile before tests; both attempt logs are retained.
This closes the scoped host check, not all L8 requirements or sanitizer coverage of Python/JVM adapters.
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
- Final CI and artifact verification passed; the owner accepted the measured RC performance cost.
  Final independent evidence review accepted RC preparation; publication is explicitly on hold.

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
Owner-approved stable Rust ordering now removes NumPy producer admission from the supported Python path; affected local, installed live and final platform CI checks passed.

The Java load matrix now requires JDK 17 and 21 on every required platform; 15 workflow/artifact tests passed, including an exact matrix-completeness check.
Current Python socket retry/timeout and credential-redaction tests passed (61 tests at `05e9a5d63`); external log: `phase-5/effort-3/current-transport-safety.log`.
The independent native-safety review confirms that the deferred sanitizer requirement concerns `sdm-host`; Python/JVM ownership and crash tests are complementary, not sanitizer coverage of those runtimes.

The final candidate is built and CI-validated across all five approved targets; publication remains on hold.

Owner release direction: prepare a 2.0.0 RC; do not publish or tag.
Java will initially use GitHub Releases after separate authorization; Maven Central is a follow-up.
The sorting audit corrected an earlier claim: no-replacement sampling preserves unique source indexes; the approved change concerns existing ties and certain numeric object labels.

Current session comparison (release wheel `d17422bb9`, same Python/NumPy/pandas/Arrow as archived `9bda54ef1`): three alternating process runs, each with five measured repetitions, found median fit 27.4 ms versus 23.9 ms and scoring 7.3 ms versus 6.4 ms (about +15%).
Peak process RSS was 134,772 versus 133,704 KiB; this is a process high-water measure, not isolated native allocation.
The earlier build explicitly forced Rust routing; its default route is Python, so the initial default-route comparison is not a Rust baseline.
Evidence: external `phase-5/effort-3/session-benchmark-interleaved-core.json` and `session-benchmark-current-environment.json`.
This local constant-response benchmark excludes network/model time.
On 2026-11-02 the owner accepted this measured cost for the RC, while explicitly noting it is undesirable.
Follow-up: reduce repeated host discovery and investigate combining schema validation/result handling calls without dropping validation or moving shared policy back into Python.
The increase from 291 to 357 Rust calls explains added work, but the timing contribution of each operation has not been isolated.

RC artifact check at `64a164bd0`: API/connectors wheels and source archives built as `2.0.0rc1`; the client wheel normalizes Rust `2.0.0-rc.1` to the same Python version.
All three wheels installed into a fresh external Python 3.12 environment; dependency checks, native SHA/contract 57 checks and absence of retired engine/PQL imports passed.
The installed client completed six fit/close cycles and 30 predictions using a local constant-response transport, not a live model endpoint.
Evidence: external `phase-5/effort-3/rc-installed-identity.json`, `rc-pure-python/artifacts.json` and `rc-session-smoke.json`.
This earlier local Linux x86-64 wheel targets glibc 2.34; current manylinux2014 and other platform artifacts are verified separately below.

Earlier RC CI `41501690988` tested `43d94e9bc` with locked Rust dependencies; it was superseded and is not counted as passing evidence.
The installed `64a164bd0` RC wheel passed live in-memory Relational ranking and forecasting against private-bridge model version 28, followed by the selected Python contract checks.
Evidence: external `phase-5/effort-3/rc-live-ranking-forecast.jsonl` and `rc-live-ranking-forecast-report.json`.
These two cases close the affected Python journey rerun; they are not a Java comparison or a complete live/runtime matrix.

## Latest RC corrections

Commit `79769ad00` implements the approved portable Tabular ordering.
It passed 407 Python index tests, targeted Rust tests, Clippy and two live Tabular journeys; independent review found no remaining blocker.
Commit `1661f3388` updates RC-version snapshots and connector assertions; 101 focused checks passed, with two optional skips.
CI run `41501690988` tests earlier source `43d94e9bc`, so its ordering, lint and RC-expectation failures do not test these fixes.
The Snowflake subprocess-probe failure was traced to the runner’s system-tool symlink and repaired without changing client behavior; subsequent CI Python 3.11–3.14 suites passed.
Earlier shipping artifacts identified source `2e3823fcc`; their checks are retained as history.
Current artifact identity and remaining acceptance are recorded below; the owner has accepted the measured session-preparation overhead for this RC.

## Retained observation coverage

The live matrix is complemented by these maintained contracts; final CI must execute them.

| Observation | Retained tests |
|---|---|
| SQL and typed parameters | `contract/tests/test_t22_sqlite.py`; Rust `databricks_goldens.rs` and `t328_key_text.rs` |
| Provider paging, failures and negative controls | `contract/tests/test_t25_provider_kit.py`, with live local database services |
| Warnings and refusal details | `test_invalid_escape_warning_on_the_core.py`, `test_t310_metric_time_warning.py`, `test_tabular_direct_core.py`, `test_t136_refusal_codes.py` |
| HTTP errors, causes, retries and timeouts | `test_transport_socket.py` and `test_transport_hardening.py` |
| Pickles and versioned state | `test_t214b_pickles.py`, `test_schema_pickles.py`, `test_t214b_host.py` |
| New in-memory semantics | 12 generated Tabular cases and eight generated Relational temporal-join cases; independent row/order/aggregate oracles |

The two generated suites passed all 38 tests, including their fixed examples.
Incidental pandas warnings and owner-approved stable ordering remain explicit accepted differences; this is not a replay of every historical error string.
Exact test-function mapping is retained in external `effort-3/evidence-observation-map.md`.

The historical managed Databricks sweep is genuine Jobs evidence: 21 paired profiles and 255 questions, with pinned artifacts and terminal run receipts.
Its reviewed consumer SHA-256 is `cfabf1853ccc01d722da21d0f9261d7a970d95cbd6f3566ad9a8ee2b380e28ce`.
Reuse unchanged warehouse-query outcomes; the affected DBR16.4 nullable-string path passed the managed check described below.
Local CI dependency recreation is not a managed-job execution.

## Managed DBR16.4 closure

Run `704276691546809`, task `997203890765698`, completed successfully with installed RC source `483df0c54` and a matching native hash.
Raw Spark reads and Arrow-backed strings each retained all 12 nulls and completed prediction plus fit/predict/close against Tabular model version 2.
The runtime used NumPy 1.26.4, pandas 2.2.2 and Arrow 15.0.2.
The first attempt failed before product execution because the notebook incorrectly constrained pandas to the runtime's bundled 1.5.3; the corrected install follows the client's documented minimum.
Receipt: external `effort-3/dbr164/receipt-r2.json`, with raw terminal state and notebook output alongside it.
This is a fresh x86 managed-runtime check; the historical Arm outcome is retained with separate current native-platform checks, not described as freshly rerun.

Corrected full CI is run `41510270371` on source `2e3823fcc`.
Its predecessor exposed test/configuration issues now independently reviewed and fixed; it is not a final passing run.

Run `41510270371` exposed two further test-only issues, corrected in `f26f51aa9`: Python 3.10 nanosecond parsing in the evidence comparator, and two external-source checks incorrectly required by the installed API gate.
The affected Python 3.10 tests pass; the full local comparator suite passed 49 tests and the corrected current-source API gate passed all 400 cases without skips.
The first local API attempt used the older installed 1.0.0 package and correctly failed its version snapshot; the current-source rerun passes.
Independent review approved the corrections; external canonical-source checks remain in the maintainer suite, not claimed by installed API evidence.
That earlier CI run was superseded; current corrected CI is recorded below.

Artifact inventory is now complete locally: five native wheels, API/connectors wheels and source archives, and JAR/POM, with hashes and build-source identities.
Both source archives rebuilt wheels successfully; final CI acceptance is separate.
Scoped host ASan reuse was independently approved in external `effort-3/host-asan-reuse-disposition.json`: unchanged host/binding sources and external dependencies, with only workspace version changes in Cargo.lock.
This retains the earlier memory-safety scope and does not certify the new sorting algorithms or final binaries.

## Current candidate acceptance

Final candidate `d494b2e76` passed [CI run 41524166634](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41524166634).
All five Python wheels and the Java JAR/POM have verified hashes and source identities; the JAR contains all five native libraries and legal notices.
The 11-file inventory is external `effort-3/ci-41524166634-artifacts/artifact-inventory.json`.
API/connector source archives retain their original `29950f0d9` identity, with unchanged package directories and successful wheel rebuild evidence.
The installed client suite passed 4,324 tests, with 250 skips and 87 deselections; the separate required API gate passed all 400 cases without skips.
Both macOS wheel consumers passed 143 public API cases and optional connector-driver imports.
Two independent reviews found no additional implementation or artifact blocker.
Final CI completed successfully with 81 successful jobs, including all ten JDK 17/21 platform loading checks and Java behavior/provider suites; the targeted-only PQL job was intentionally skipped.
Final independent evidence review accepted RC preparation for `d494b2e76`, with no remaining acceptance gap in the approved scope.
Publication, tags and merge remain on hold by owner instruction.

## Subsequent deleted-API audit

The [API removal audit](api-removal-audit/audit.md) found legacy-engine capabilities absent from the maintained facade, including visualization and diagnostics.
The CI and distribution evidence above remains valid for its tested scope, but does not establish full legacy capability parity.
These gaps require explicit disposition and targeted coverage before merge readiness is reasserted.
