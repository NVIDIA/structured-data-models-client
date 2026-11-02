<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effort 3 readiness

**Validation is still in progress; not ready to merge or publish.**

| Area | Current evidence / remaining work |
|---|---|
| Integration | Candidate includes `main` at `12955d397` and `rust-core` at `2b9e6ea88`; verified against refreshed remote refs |
| Release scope | Python/Java `2.0.0rc1`, Rust `2.0.0-rc.1`; Linux x86/Arm, macOS Intel/Arm, Windows x86; Windows Arm deferred by owner |
| Python artifact | Current CI shipping wheels verified for Linux x86/Arm, Windows x86 and macOS Arm; clean Linux manylinux2014 install verifies native hash, contract 57 and RC versions; retired runtimes absent |
| Final CI | [Run 41510270371](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41510270371), source `2e3823fcc`; still running; Python 3.10 comparator and installed API source-checkout selection need corrected validation |
| Ordering improvement | Approved stable ties and exact numeric values; 407 focused Python tests, Rust checks, independent review and two installed live Tabular journeys passed |
| Generated semantics | 12 Tabular and eight Relational cases added to existing suites; all 38 tests including fixed examples passed; independent oracle review approved |
| Managed runtime | DBR16.4 run `704276691546809` passed: raw Spark and Arrow-backed strings preserved 12 nulls each and completed prediction plus fit/predict/close against Tabular version 2 |
| Historical coverage | Retained tests and genuine 21-profile Databricks sweep mapped in [verification.md](../verification.md); unchanged warehouse paths do not need wholesale replay |
| Performance | Ordering benchmark improved at 10,000 context rows in the measured cases; session preparation is about 15% slower than the earlier Rust build, with similar memory; owner disposition pending |
| Publication | Explicitly on hold; no release/tag/upload to package indexes; Java destination will initially be GitHub Releases |

The current clean installed shipping artifact identifies source `2e3823fcc`; the successful managed Databricks run used local artifact `483df0c54`, with identical runtime code.
Detailed receipts and review findings remain under the external `phase-5/effort-3/` evidence directory.
Final readiness requires successful affected checks, five-platform artifact verification, the performance decision and final evidence review.

CI corrections are confined to tests/configuration: Windows line-ending hashing, a symlink-aware system-tool guard, a pandas-independent numeric oracle, and installation of the Databricks serving test extra.
Local verification passed 14 ledger checks, three guard checks, 319 ordering checks on the affected dependency stack, and all 402 required API cases without skips.
The first managed-runtime attempt failed during installation because the validation notebook incorrectly pinned bundled pandas 1.5.3; the retry uses documented compatible versions.

The managed run used NumPy 1.26.4, pandas 2.2.2 and Arrow 15.0.2, with the installed native hash checked against the RC artifact.
Independent review approved the CI-only corrections without finding weakened gates.

Two further test-only corrections were independently reviewed: Python 3.10 timestamp comparisons preserve nanoseconds outside the datetime parser, and the installed API gate excludes two external-source provenance checks.
Those source checks remain in the maintainer suite; installed API success does not prove comparison against the external canonical specification.
