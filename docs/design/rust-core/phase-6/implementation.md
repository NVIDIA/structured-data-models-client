<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First four restorations

Each feature has its own branch/worktree, public Python/Java contract review, focused tests, implementation and independent code review.
The coordinator merges reviewed commits into `rust-core-phase-6` and reconciles shared API inventories after integration.
Use at most two heavy jobs, Cargo jobs=2 and the existing heavy guard; do not install competing native builds into a shared environment.

| Feature | Branch | Status |
|---|---|---|
| Visualization | `rust-core-phase-6-visualization` | Reviewed and merged; 13 Python / 3 Java tests passed; integrated wheel asset check pending |
| Sanitization | `rust-core-phase-6-sanitization` | Reviewed and merged; 321 Rust / 74 Python tests and Java relational tests passed |
| RelBench | `rust-core-phase-6-relbench` | Reviewed and merged; 58 Python / 4 Java tests passed; real dataset matches across hosts |
| Reports/explanations | `rust-core-phase-6-reports` | Contract approved; host APIs and numerical helper in progress |

Three workers can run alongside the coordinator; the fourth starts when a slot frees.
No feature is complete on implementation alone: contracts, affected tests and independent review must pass.
Remaining audit capabilities stay in the Phase 6 scope and are not implicitly completed by these four merges.

Sanitization introduced contract 58.
Reports will build on it with contract 59 for lazy attribution aggregation; that worker owns the warm native target during its builds.
Integrated Python/Java checks are running for the first three merges; final wheel resource verification follows the reports merge.
