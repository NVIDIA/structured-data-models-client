<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 9 verification

**Complete on `rust-core-phase-9`; not merged into `rust-core` or `main`, and nothing published.**
**Audit qualification:** the subsequent [PR review audit](pr63-review-audit.md) reproduced inherited defects and found specific missing coverage.
The implementation campaign is complete, but its passing suites do not establish that every PR review concern is resolved.

Both implementation branches were reviewed and integrated, including main commits `506e37024` and `4edabbf9c`.

## What changed

| Effort | Result |
|---|---|
| Column explanations | Rust owns statistics, cohort policy, ordering and request/session lifecycle. Python and Java expose containers and dataframe/Arrow views. Existing sampling SQL is unchanged. |
| Databricks quickstart | Integrated main's Tabular notebook and deployment checks against the new client APIs. |
| Documentation | Removed 67 historical paths after retaining current engine boundaries, date rules, all 17 benchmark tables and dated evidence/limits. Corrected and visually inspected the two maintained diagrams. |

Independent reviews covered engine/binding boundaries, persistence, model-payload isolation, forecast/batch identity, holdout and Arrow ownership.
Java's pre-2.0 `explanations()` return type intentionally becomes a container with separate gradients and columns; repository callers and API inventory are updated.

## Tests and results

These counts describe separate suites and overlap; do not add them together.

| Check | Actual result |
|---|---|
| Rust contract, wire, request and core suites | 577 passed, zero ignored. Includes lifecycle, malformed metadata, ordering and independent numerical expectations. |
| Python integrated public API and skills checks | 202 passed; 16 opt-in live cases skipped. |
| Additional public boundary tests | Real SQLite materialization still explains after its connection closes. The final 12-test public suite passes, including category/class caps, missing values and timestamp cohorts relative to anchors. |
| Java focused JNI suites | 29 passed, one real-endpoint case skipped in this suite; endpoint behavior exercised separately below. Informational constants also tested. |
| Actual CPU model endpoints | Four Python cases and three Java methods passed in 43.42 seconds, including column assertions, both model families, regression/classification and sessions. |
| Selected main-versus-new constructors | Exact values/order/columns match for categorical regression, categorical classification, numerical quantiles and multicategorical inputs. Empty classification improves main's exception into a valid empty result. |
| Notebook checks | 21 passed; imports, structure, deployment mocks and preparation. No notebook deployment was performed. |
| Maintained enforcement | Both main/rust-core API ledger baselines pass: 250 Python and 566 Java symbols. Ledger/developer checks: 39 passed. Ruff, mypy, rustfmt, Clippy, dependency boundaries, notices, hygiene and size checks pass. |

## Evidence and limits

The integrated CPU run rebuilt Python/JNI from source revision `65c22d830659a604684f85dfb2d4f469afa73532`.
Subsequent changes only add tests, ledger entries and completion documentation.
Local receipts are under `~/.local/share/sdm-orchestration/phase-9/cpu-smoke-65c22d830/evidence/`; `local-build-identity.json` records source and native hashes.
This was a local source/native run, not release-package or hosted-CI validation, regardless of the reused helper's packaging label.
Selected old/main comparison receipts are under `~/.local/share/sdm-orchestration/phase-9/column-public-comparison/`.

This campaign does not claim a new exhaustive connector matrix, warehouse deployment run, benchmark campaign or release qualification.
Existing [validation evidence](../../../development/validation.md) and [performance results](../../../development/performance.md) retain their original revisions and limits.
No performance tuning, legacy runtime restoration or new warehouse queries were introduced for column analysis.
