<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Raw-input/type parity: complete

The [plan](type-parity-plan.md) is complete.
The original 582 historical comparisons are preserved separately.
See the [new coverage matrix](type-parity-matrix.md) for individual cells.

| Work | Current evidence |
|---|---|
| New live comparisons | 210 passed against frozen Python: 105 Python and 105 Java journeys |
| What is compared | Unchanged public reader output, actual model request bodies, results and verified refusals; representative sessions included |
| Types covered | Text/Unicode, nulls, list/sequence refusals, numeric boundaries/decimals/float32, dates and timezone/precision cases |
| Product fixes | Java Parquet nanosecond preservation, frame float32 metadata, consistent list refusals, shared native Arrow date conversion |
| Integration | Product fixes in `2519631e6` and `285c5c053`; permanent fixtures, runners, comparator and this matrix accompany them on `rust-core-phase-4` |
| S3 coverage | All eight unchanged-reader CSV/Parquet comparisons pass across both models and languages |
| Review and regressions | Independent review approved; 54 Python harness, nine Java harness and 42 Rust checks passed; repository lint/format, hygiene and attribution checks passed |

Passing comparisons include verified unsupported behavior; they do not mean every type is accepted by both models.
Of the 210 comparisons, 134 completed prediction journeys and 76 matched verified refusals.
Live calls used the existing deployed model endpoints through the recorded private transport bridges.
Raw reader representations remain recorded separately from model inputs.
The source-backed decimal comparison records a narrowly proved accuracy improvement over the old cast.

S3 runs used refreshed temporary AWS credentials through standard environment variables because the Python Parquet reader did not resolve the browser-login profile.
No reader output was repaired; no completed non-S3 live journey was repeated.
Dedicated CI credentials remain deferred as agreed.

Evidence: `phase-4/type-parity/wire-comparison-complete.json` in the orchestration directory, using candidate artifact `285c5c053`.
That report selects the 210 new completed cells; historical cells are outside its scope and retain their original evidence.

## Verified limits

| Input | Observed behavior, matching the frozen client |
|---|---|
| Short text inferred from strings | Prediction and representative sessions succeed |
| Explicit text semantic type | Relational endpoint refuses it; Tabular omits that feature |
| Long prose fixture | Endpoint payload-size refusal; not a successful long-text prediction |
| Multicategorical / numeric lists | Explicit schema refusal |
| Dates / timestamps | Relational succeeds; Tabular endpoint refuses these feature types |

Null and numeric fixtures succeed, including the documented decimal accuracy improvement.
These are targeted type checks plus scalar coverage of every supported reader family, not every possible type × connector combination.
