<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# What remains before merging into main

**Phase 6 is merged and pushed to `rust-core` at `db8f88528`; readiness work below is proposed, not implemented.**
The merge preserves the tested Phase 6 tree unchanged.
We ported the audited main features; we did **not** rebase onto latest main.
Audit baseline: `origin/main` at `847a1e9e3`, checked November 4, 2026.

| Proposal | Finding | Recommended action |
|---|---|---|
| [Repository size](repository-size/design.md) | 22.56M added lines; 20.50M are warehouse captures; further bulk remains in recorded vectors/schemas | Extract small independent regressions, delete redundant archives, enforce size/consumer budgets |
| [CI coverage](ci-coverage/design.md) | Required credentials/config absent; live environment excludes rust-core; 16 live-test declarations are not executed | Separate PR/local, protected live, and stress lanes; configure and verify their resources |
| [Developer contracts](developer-contracts/design.md) | Useful ledger already exists, but guidance is stale and behavior/type coverage is only partly linked | Extend the ledger, generate readable matrices, add one contributor/agent change guide |
| [Validation and performance](validation-performance/design.md) | Focused Phase 6 proof exists; no final whole-matrix or stage-level performance campaign | Reuse valid evidence, run changed-path comparisons and bounded latency/memory benchmarks, then final-candidate acceptance |

```text
Shrink fixtures ───────┐
Improve guide/ledger ──┼→ final candidate → matrix + benchmarks → main review
Configure CI lanes ───┘
```

No fixture deletion, history rewrite, CI reconfiguration or new benchmark campaign was performed in this audit.
Review the proposals first; published-history handling and eventual integration into `main` remain separate decisions.
PR #63's cohort/container proposal is still open at `f240fe448`, outside completed Phase 6 scope.
