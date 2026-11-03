<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First four restorations

Each feature has its own branch/worktree, public Python/Java contract review, focused tests, implementation and independent code review.
The coordinator merges reviewed commits into `rust-core-phase-6` and reconciles shared API inventories after integration.
Use at most two heavy jobs, Cargo jobs=2 and the existing heavy guard; do not install competing native builds into a shared environment.

| Feature | Branch | Status |
|---|---|---|
| Visualization | `rust-core-phase-6-visualization` | Reviewed and merged; 13 Python / 3 Java tests passed; wheel/JAR assets verified |
| Sanitization | `rust-core-phase-6-sanitization` | Reviewed and merged; 321 Rust / 74 Python tests and Java relational tests passed |
| RelBench | `rust-core-phase-6-relbench` | Reviewed and merged; 58 Python / 4 Java tests passed; real dataset matches across hosts |
| Reports/explanations | `rust-core-phase-6-reports` | Reviewed and merged; shared fixtures and contract 59 verified |

Three workers can run alongside the coordinator; the fourth starts when a slot frees.
No feature is complete on implementation alone: contracts, affected tests and independent review must pass.
Remaining audit capabilities stay in the Phase 6 scope and are not implicitly completed by these four merges.

Sanitization introduced contract 58.
Reports adds contract 59 for lazy attribution aggregation.
Final integration passed 157 Python tests against the installed wheel and 33 Java tests.
Wheel/JAR assets and packaged native loading passed.
See [completion evidence](completion.md) for scope, reviews and remaining audit work.


## Next three restorations and stronger coverage

| Work | Branch | Status |
|---|---|---|
| Inference and safe edits | `rust-core-phase-6-inference` | Implementation checkpoint; constructor provenance review and native validation await contract 60 |
| Time-series preparation | `rust-core-phase-6-timeseries` | Contract approved; implementation in progress |
| Labels-only extraction | `rust-core-phase-6-labels` | Contract approved; implementation in progress |
| Thorough testing of the first four | `rust-core-phase-6-coverage` | Worker active; auditing and strengthening public-workflow coverage |

The three designs under `phase-5/api-removal-audit/{inference-edits,timeseries,labels-only}/design.md` define the scope and testing expectations.
Use existing test infrastructure, focused old-client comparisons and independently calculated expected results; do not rebuild migration instrumentation.
The coverage worker audits actual tests for the first four features, closes meaningful missing public-workflow/security/ownership/error coverage, reproduces and fixes real bugs, and records remaining limits honestly.
Prioritize real execution over mocks where it exercises the feature; use model endpoints only for workflows that actually predict.
All changes receive independent review and merge into Phase 6; no publication or merge into `rust-core`/`main`.
The coordinator serializes contract-version allocation and warm-target ownership to avoid conflicting native builds.

Current allocation: time-series preparation owns contract 60 and the warm native target.
Inference/labels may develop isolated logic and tests, but must integrate the preceding contract before generating their own version.

Planned subsequent contracts: inference 61, labels 62, each based on the preceding integrated contract.
