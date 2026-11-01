<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Raw-input/type parity: in progress

The [plan](type-parity-plan.md) and implementation goal remain active.
The original 582 historical comparisons are preserved separately.
See the [new coverage matrix](type-parity-matrix.md) for individual cells.

| Work | Current evidence |
|---|---|
| New live comparisons | 202 passed against frozen Python: 101 Python and 101 Java journeys |
| What is compared | Unchanged public reader output, actual model request bodies, results and verified refusals; representative sessions included |
| Types covered | Text/Unicode, nulls, list/sequence refusals, numeric boundaries/decimals/float32, dates and timezone/precision cases |
| Product fixes | Java Parquet nanosecond preservation, frame float32 metadata, consistent list refusals, shared native Arrow date conversion |
| Integration | Product fixes in `2519631e6` and `285c5c053`; permanent fixtures, runners, comparator and this matrix accompany them on `rust-core-phase-4` |
| Remaining live coverage | Eight S3 CSV/Parquet cells require refreshed AWS access; these are incomplete, not passes |
| Review and regressions | Independent review approved; 54 Python harness, nine Java harness and 42 Rust checks passed; repository lint/format, hygiene and attribution checks passed |

Passing comparisons include verified unsupported behavior; they do not mean every type is accepted by both models.
Of the 202 comparisons, 126 completed prediction journeys and 76 matched verified refusals.
Live calls used the existing deployed model endpoints through the recorded private transport bridges.
Raw reader representations remain recorded separately from model inputs.
The source-backed decimal comparison records a narrowly proved accuracy improvement over the old cast.

Pending: the eight S3 comparisons after AWS access is refreshed, followed by final completion reconciliation.
Completed live journeys need not be repeated.

Evidence: `phase-4/type-parity/wire-comparison-final.json` in the orchestration directory, using candidate artifact `285c5c053`.
That report selects the 202 new completed cells; historical cells are outside its scope and retain their original evidence.
