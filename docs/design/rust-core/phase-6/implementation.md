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
| Inference and safe edits | `rust-core-phase-6-inference` | Reviewed and merged; 95 affected Python / 12 Java tests and live three-row prediction passed |
| Time-series preparation | `rust-core-phase-6-timeseries` | Reviewed and merged; Python/Java fixtures and live four-row forecast passed |
| Labels-only extraction | `rust-core-phase-6-labels` | Reviewed and merged; 21 Python / 3 Java tests and live Databricks/Snowflake labels passed |
| Thorough testing of the first four | `rust-core-phase-6-coverage` | Reviewed and merged; composed workflows, corruption/ownership/report edge cases; 22 Python / 15 Java tests passed |

The three designs under `phase-5/api-removal-audit/{inference-edits,timeseries,labels-only}/design.md` define the scope and testing expectations.
Use existing test infrastructure, focused old-client comparisons and independently calculated expected results; do not rebuild migration instrumentation.
The coverage worker audits actual tests for the first four features, closes meaningful missing public-workflow/security/ownership/error coverage, reproduces and fixes real bugs, and records remaining limits honestly.
Prioritize real execution over mocks where it exercises the feature; use model endpoints only for workflows that actually predict.
All changes receive independent review and merge into Phase 6; no publication or merge into `rust-core`/`main`.
The coordinator serializes contract-version allocation and warm-target ownership to avoid conflicting native builds.

All three contracts are integrated through contract 62; the warm native target is released.
Inference/labels may develop isolated logic and tests, but must integrate the preceding contract before generating their own version.

See [restoration completion](restoration-completion.md) for combined production-artifact checks and remaining audit scope.

Contract lock policy retains merged entries and only the newest unmerged version via `scripts/rust_core/contract_lock.py --prune-working`; schema history files remain unchanged.

[Original-feature coverage audit](coverage-audit.md) records the focused additions and their limits.
