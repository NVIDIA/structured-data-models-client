<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Branch/worktree cleanup proposal

**Historical pre-cleanup audit.**
The owner subsequently authorized cleanup; see [completed batches and recovery records](cleanup-result.md).
Snapshot: 2026-11-01, before pushing the new Phase 5 planning branch.
Remote listings were checked directly, not inferred from stale remote-tracking refs.

| Inventory | Result |
|---|---|
| Remote branches | 22; `rust-core-phase-4` and `rust-core-phase-4-tabular` still exist |
| Local branches | 82: 15 ancestry-merged into `rust-core`, 67 not |
| Worktrees | 236: 229 clean, six dirty, one missing registration target |
| Archived Phase 4 commits | 234 archive tags locally and verified on origin |
| Disk | Worktrees sum to 386.59 GiB; shared allocations may be counted repeatedly |

## Proposed order

| Batch | Action after approval | Preservation check |
|---|---|---|
| 1. Merged branches | Delete remote `rust-core-phase-3`, `rust-core-phase-4`, `rust-core-phase-4-tabular`; later remove the 12 enumerated merged local branches | Recheck exact tips are reachable from `rust-core`; remove associated worktrees before deleting local branches |
| 2. Retired worktrees | Remove 161 clean remotely archived worktrees and eight clean merged development trees | Recheck current status, ignored files, active processes and references; exact HEAD must remain recoverable |
| 3. Remaining clutter | Review dirty/unmerged/untagged work separately; prune the missing registration only after confirming its path is still absent | Preserve commits, untracked files and binary-capable diffs before deletion; do not use branch age as proof |

Batch 2 accounts for **267.73 + 15.97 GiB** of per-tree allocations; actual recovered space may be lower because of shared files.
Deleting branch names alone saves little space.
Measure actual filesystem free space after each approved batch.

## Keep

Protect `main`, `rust-core`, `rust-core-reference`, active Phase 5, authoritative test evidence and final artifact/provenance files.
Keep the **87 GiB shared warm Rust cache** for Phase 5; it is not part of the worktree saving estimate.
A separate 33.41 GiB Phase 3 evidence checkout includes ~32 GiB of Rust build files, but requires consumer/evidence review before reclaiming those files.

Five historical worktrees contain changes: two Phase 3 checkpoints, `sdmc-p4-t44-host-context-range-r41`, `sdmc-p4-t46-task-r74`, and `/mnt/localssd/tmp/t310-schema-fresh-G`.
The sixth dirty tree was the active Phase 5 planning checkout.
The `t46-task-r74` tree alone has 19 modifications and four new files.

Remote `main` is `12955d397`, ahead of local `main`; release preparation must reconcile it.
Other remote feature/dependency branches are outside the migration cleanup proposal; four tips were not available locally for ancestry checks.

## Exact inventory

The per-branch/per-worktree manifest, including SHAs, dirty paths, remote archive tags, sizes and proposed batch membership, is saved at:

`~/.local/share/sdm-orchestration/phase-5/branch-worktree-audit-20261101.json`

A short audit report is beside it as `branch-worktree-audit-20261101.md`.
This snapshot is a proposal, not a deletion script; recheck it immediately before cleanup.
