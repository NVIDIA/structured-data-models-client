<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# What remains before merging into main

**Phase 7 is complete and pushed to `rust-core` at `4e3f4a86f`; Phase 8 implements the four remaining readiness efforts.**
Main `847a1e9e3` was merged with Rust ownership preserved in `711968587`.
The [main merge plan](main-merge/design.md) is complete; [verification](main-merge/verification.md) records the evidence and limits.
Audit baseline: `origin/main` at `847a1e9e3`, checked November 4, 2026.

| Proposal | Finding | Recommended action |
|---|---|---|
| [Main merge](main-merge/design.md) | 20 conflicts, a silent import collision and four reproduced behavior gaps | Resolve by contract, port missing behavior and prove Rust routing before integration |
| [Repository size](repository-size/design.md) | 22.56M added lines; 20.50M are warehouse captures; further bulk remains in recorded vectors/schemas | Extract small independent regressions, delete redundant archives, enforce size/consumer budgets |
| [CI coverage](ci-coverage/design.md) | Required credentials/config absent; live environment excludes rust-core; 16 live-test declarations are not executed | Separate PR/local, protected live, and stress lanes; configure and verify their resources |
| [Developer contracts](developer-contracts/design.md) | Useful ledger already exists, but guidance is stale and behavior/type coverage is only partly linked | Extend the ledger, generate readable matrices, add one contributor/agent change guide |
| [Validation and performance](validation-performance/design.md) | Focused Phase 6 proof exists; no final whole-matrix or stage-level performance campaign | Reuse valid evidence, run changed-path comparisons and bounded latency/memory benchmarks, then final-candidate acceptance |

## One coordinated implementation

All work integrates into `rust-core-phase-8`, starting at `4e3f4a86f`.
Three isolated implementation branches run concurrently; final acceptance runs after their integration.
Cleanup, CI wiring, developer enforcement and benchmark tooling are integrated; final candidate validation is in progress.
The comparator now accepts an explicitly pinned clean baseline, with archived-reference defaults preserved.
Full recordings are preserved outside Git with hashes; maintained fixtures keep independently verified expectations.
CI secrets remain unavailable, and the local AWS login needs renewal for S3 acceptance; neither is counted as passing live evidence.

### Validation checkpoint – November 4

| Check | Current evidence |
|---|---|
| Maintained Python suite | 6,881 passed; nine skips accounted for separately |
| Repaired contract files on the integrated branch | 629 passed, no skips |
| Repository tooling | 164 passed, no skips |
| Platform CI | Eight validation-tooling causes repaired; corrected full [run 41796680139](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41796680139) is running at `bf5f7f412` |
| Live matrix | All 740 Python/Java non-S3 cells pass across 13 connectors with observed subprocesses |
| Live API registry | All 24 Python invocations and six available Java methods pass; Java S3 remains unavailable |
| Current-main comparison | All 370 non-S3 cells executed with matched dependencies: 277 strict matches and 93 reviewed differences; see [verification](validation-performance/verification.md) |
| Benchmarks | Two fixed-response workloads measured across four variants and independently reviewed; remaining workloads pending, see [results](validation-performance/verification.md) |

The installed candidate artifacts are from `08d027795`; the platform CI runner is `7f6fbbc8e`.
Later reviewed runners repair test endpoint selection and authentication observation; product source remains unchanged.
The historical live comparison matches pandas 2.3.3, NumPy 2.5.4 and Arrow 24; earlier mismatched-dependency observations are diagnostics only.
These partial results do not establish completion.
The final source audit found outstanding CI resource/preflight/artifact-sharing work and developer-ledger first-adoption/evidence-view gaps.
These are being completed before acceptance; the tested product implementation remains unchanged.

| Owner | Changes | Shared contract |
|---|---|---|
| Repository-size agent | Archive/extract fixtures, remove obsolete captures/schemas, enforce size and consumers | Preserve independent expectations and all declared type fixtures; provide gate command to CI |
| CI agent | Local/live/stress lanes, execute live registry, outcome accounting | Contracts agent owns manifest edits; missing credentials/skips never count as passed |
| Developer-contracts agent | Existing ledger, ownership/assertion links, readable views, developer guide | Own shared manifest; use actual contract operations and test targets |
| Coordinator | Current-main comparison baseline, integration/review, final matrix and benchmarks | Run final checks on the integrated candidate; no performance tuning |

Archive originals outside Git before deleting them.
Compact expectations must expand to independently verified original expectations, not values generated by the engine under test.
Dedicated CI credentials remain deferred by the owner; implement secure lanes and run available acceptance locally, recording any unavailable coverage explicitly.

```text
Merge main + prove Rust routing
    ↓
Shrink fixtures ───────┐
Improve guide/ledger ──┼→ final candidate → matrix + benchmarks → main review
Configure CI lanes ───┘
```

The initial audit made no implementation changes; Phase 8 now implements its authorized recommendations.
The owner authorized the four remaining efforts for `rust-core-phase-8` after the main merge completes.
Reconcile their dependencies first, use at most three concurrent implementation agents, and run final correctness and performance checks last.
Report comparable warm/cold latency and memory measurements without performance tuning; minor latency regressions are acceptable.
Phase 8 changes stay on that branch; published-history rewriting, release publication and eventual integration into `main` are not authorized.
PR #63's cohort/container proposal is still open at `f240fe448`, outside completed Phase 6 scope.
