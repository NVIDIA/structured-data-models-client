<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rust migration cleanup: completed batches

Completed 2026-11-01 under the owner's migration-only cleanup authorization.

| Result | Count |
|---|---|
| Removed worktrees | 211 (167 archived/merged, then 44 independently backed up) |
| Removed branches | 10 local and three remote; all deleted branch tips were merged into `rust-core` |
| Remaining worktrees | 25, including protected, dirty and unrelated work |
| Actual recovered space | 324.75 GiB; filesystem usage fell from ~906 GiB to ~581.35 GiB |
| Available space | ~409.52 GiB |

Remote branches removed: `rust-core-phase-3`, `rust-core-phase-4`, `rust-core-phase-4-tabular`.
The local Phase 4 branch/worktree remains; its 317 commits are included in merge `2b9e6ea88`.
The four Java-parity commits are patch-equivalent to merged commits; its local branch remains too.

## Preserved

- Active Phase 5, `main`, `rust-core`, the frozen reference, dirty changes and still-unmerged branch names.
- Original 234 Phase 4 archive tags, plus two new archive tags protecting previously detached commits.
- Final validation evidence/artifacts and copied Java test reports.
- Two shared warm Rust targets: `t44-r7-parallel` and `t44-r8-parallel`; one obsolete third target was removed.
- Existing environments in `sdmc-p3-t39` and `sdmc-p4-core`; evidence build checkouts were not removed.

## Recovery and audit

Records are in `~/.local/share/sdm-orchestration/phase-5/`.
The per-worktree action logs are `cleanup-worktree-actions.jsonl` and `cleanup-second-actions.jsonl`.
`cleanup-final-result-20261101.json` records space measurements and backup SHA256 hashes.

`pre-cleanup-remaining-20261101.bundle` preserves Git refs/history before the second batch.
`cleanup-detached-heads-20261101.bundle` preserves the two additional detached heads; their archive tags were also pushed.
An isolated bare restore confirmed all 44 second-batch commits existed before their worktrees were removed.
The temporary restore checkout was then removed; both bundles remain.

No unrelated project branches/worktrees, dirty trees or source evidence were deleted.
