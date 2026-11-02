<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effort 3 readiness

**Validation is still in progress; not ready to merge or publish.**

| Area | Current evidence / remaining work |
|---|---|
| Integration | Candidate includes `main` at `12955d397` and `rust-core` at `2b9e6ea88`; verified against refreshed remote refs |
| Release scope | Python/Java `2.0.0rc1`, Rust `2.0.0-rc.1`; Linux x86/Arm, macOS Intel/Arm, Windows x86; Windows Arm deferred by owner |
| Python artifact | Clean Linux x86 install verified against native hash, contract 57 and all three RC package versions; retired runtimes absent; this local glibc 2.34 wheel is not shipping manylinux2014 proof |
| Final CI | [Run 41506491000](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41506491000), source `ba74849b0`; found test-infrastructure issues now fixed locally; corrected CI run pending |
| Ordering improvement | Approved stable ties and exact numeric values; 407 focused Python tests, Rust checks, independent review and two installed live Tabular journeys passed |
| Generated semantics | 12 Tabular and eight Relational cases added to existing suites; all 38 tests including fixed examples passed; independent oracle review approved |
| Managed runtime | DBR16.4 retry `704276691546809` is active, checking Spark/null strings and Arrow-backed strings through predict and sessions |
| Historical coverage | Retained tests and genuine 21-profile Databricks sweep mapped in [verification.md](../verification.md); unchanged warehouse paths do not need wholesale replay |
| Performance | Ordering benchmark improved at 10,000 context rows in the measured cases; session preparation is about 15% slower than the earlier Rust build, with similar memory; owner disposition pending |
| Publication | Explicitly on hold; no release/tag/upload to package indexes; Java destination will initially be GitHub Releases |

The current installed native artifact identifies source `483df0c54`; subsequent commits add tests and documentation, not runtime code.
Detailed receipts and review findings remain under the external `phase-5/effort-3/` evidence directory.
Final readiness requires successful affected checks, five-platform artifact verification, the performance decision and final evidence review.

CI corrections are confined to tests/configuration: Windows line-ending hashing, a symlink-aware system-tool guard, a pandas-independent numeric oracle, and installation of the Databricks serving test extra.
Local verification passed 14 ledger checks, three guard checks, 319 ordering checks on the affected dependency stack, and all 402 required API cases without skips.
The first managed-runtime attempt failed during installation because the validation notebook incorrectly pinned bundled pandas 1.5.3; the retry uses documented compatible versions.
