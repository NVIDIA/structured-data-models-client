<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effort 3 readiness

**Final CI and artifact validation passed; publication and merge remain on hold.**

**Subsequent API audit found capability gaps outside the tested facade.**
See [deleted API audit](../api-removal-audit/audit.md) before treating this report as overall merge readiness.

| Area | Current evidence / remaining work |
|---|---|
| Integration | Candidate includes `main` at `12955d397` and `rust-core` at `2b9e6ea88`; verified against refreshed remote refs |
| Release scope | Python/Java `2.0.0rc1`, Rust `2.0.0-rc.1`; Linux x86/Arm, macOS Intel/Arm, Windows x86; Windows Arm deferred by owner |
| Python artifact | All five shipping wheels verified against source/native hashes; clean Linux manylinux2014 install verifies contract 57 and RC versions; retired runtimes absent |
| Java artifact | JAR/POM hashes verified at `d494b2e76`; all five native libraries and legal notices present; all ten JDK 17/21 platform loading checks passed |
| Final CI | [Run 41524166634](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41524166634), source `d494b2e76`; installed client suite: 4,324 passed, 250 skipped, 87 deselected; all 400 required API cases passed without skips; 81 CI jobs passed, including Java consumer checks; the targeted-only PQL job was intentionally skipped |
| Ordering improvement | Approved stable ties and exact numeric values; 407 focused Python tests, Rust checks, independent review and two installed live Tabular journeys passed |
| Generated semantics | 12 Tabular and eight Relational cases added to existing suites; all 38 tests including fixed examples passed; independent oracle review approved |
| Managed runtime | DBR16.4 run `704276691546809` passed: raw Spark and Arrow-backed strings preserved 12 nulls each and completed prediction plus fit/predict/close against Tabular version 2 |
| Historical coverage | Retained tests and genuine 21-profile Databricks sweep mapped in [verification.md](../verification.md); unchanged warehouse paths do not need wholesale replay |
| Performance | Ordering benchmark improved at 10,000 context rows in the measured cases; session preparation is about 15% slower than the earlier Rust build, with similar memory; owner accepted this measured cost for the RC; optimization remains a follow-up |
| Publication | Explicitly on hold; no release/tag/upload to package indexes; Java destination will initially be GitHub Releases |

The current CI shipping artifacts identify source `d494b2e76`; the successful managed Databricks run used local artifact `483df0c54`, with identical runtime code.
Detailed receipts and review findings remain under the external `phase-5/effort-3/` evidence directory.
Final CI, five-platform artifact verification and independent evidence review are complete.

CI corrections are confined to tests/configuration: Windows line-ending hashing, a symlink-aware system-tool guard, a pandas-independent numeric oracle, and installation of the Databricks serving test extra.
Earlier local verification passed 14 ledger checks, three guard checks, 319 ordering checks on the affected dependency stack, and 402 API cases before the external-source checks were separated.
The first managed-runtime attempt failed during installation because the validation notebook incorrectly pinned bundled pandas 1.5.3; the retry uses documented compatible versions.

The managed run used NumPy 1.26.4, pandas 2.2.2 and Arrow 15.0.2, with the installed native hash checked against the RC artifact.
Independent review approved the CI-only corrections without finding weakened gates.

Two further test-only corrections were independently reviewed: Python 3.10 timestamp comparisons preserve nanoseconds outside the datetime parser, and the installed API gate excludes two external-source provenance checks.
Those source checks remain in the maintainer suite; installed API success does not prove comparison against the external canonical specification.

Static inspection confirms JNI glibc requirements of at most 2.17, macOS deployment targets of 10.12 Intel / 11.0 Arm and system-only macOS dependencies; Windows imports the Visual C++ and Universal CRT runtimes.
This is binary inspection, not execution on those oldest operating systems; JDK and connector dependency requirements also apply.
Receipt: external `final-ci-artifacts/java-native-dependency-inspection.json`.

The complete local inventory contains five client wheels, two pure-Python wheels, two source archives, one JAR and its POM; hashes and source identities are in external `ci-41524166634-artifacts/artifact-inventory.json`.
The two source archives retain their original `29950f0d9` build identity; both package directories are unchanged through the final candidate.
Both API/connector source archives contain legal notices and successfully rebuild wheels.
Independent review approved reuse of the 128-test host sanitizer result for unchanged host memory-safety code and dependencies, without extending it to new ordering behavior or other platforms.

The Python native-gate invocation selected no tests and accidentally triggered whole-repository pytest collection.
It is removed, with an early-error regression preventing empty selections from running pytest; the declared native provider/type gates remain enforced in Rust CI.
All 78 focused checks and independent review passed; final corrected CI also passed.
