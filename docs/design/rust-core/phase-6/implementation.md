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
| Sanitization | `rust-core-phase-6-sanitization` | Contract approved; tests and implementation in progress |
| RelBench | `rust-core-phase-6-relbench` | Contract approved; tests and implementation in progress |
| Reports/explanations | `rust-core-phase-6-reports` | Queued for a worker slot; contract review first |

Three workers can run alongside the coordinator; the fourth starts when a slot frees.
No feature is complete on implementation alone: contracts, affected tests and independent review must pass.
Remaining audit capabilities stay in the Phase 6 scope and are not implicitly completed by these four merges.

Sanitization reserves contract 58 and the warm native target during its builds.
Reports must reconcile with that version before adding any result-semantic operation.
