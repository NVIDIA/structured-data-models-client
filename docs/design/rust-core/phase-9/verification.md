<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 9 verification

**In progress; not completion or release evidence.**
Integration branch: `rust-core-phase-9`.
Nothing in this campaign authorizes merging into `rust-core`/`main` or publishing.

| Work | Verified scope | Remaining |
|---|---|---|
| Documentation cleanup, `672201ed1` and `3bc0f8dc4` | Merged into Phase 9. Removed 67 historical files after preserving current contracts, all benchmark tables and dated evidence/limits. Hygiene, size, link, formatting and reproducible diagram checks passed; coordinator visually inspected both rendered diagrams. | Recheck any links changed by final main integration. |
| Rust statistics, agent checkpoint `69ed0a3c5` | Four focused tests passed: independent two-cohort score, empty classification, exact mixed class labels, constant target. Warm targeted compile took 28.17 seconds. | Review corrections, broader numerical/type cases and lifecycle integration; not merged. |
| Databricks notebooks, agent working tree based on `69ed0a3c5` | 21 notebook tests passed in 0.17 seconds, covering public imports, source structure, deployment mocks and data preparation. | Candidate prediction integration; these tests did not deploy an endpoint or invoke a model. |

Notebook command used the existing Phase 8 candidate environment with `PYTHONPATH` pointing to the Phase 9 agent checkout's three maintained package source directories:

```text
python -m pytest packages/structured-data-models-client/tests/test_databricks_notebooks.py -q --disable-warnings
```

The notebook receipt applies to those inspected working-tree files, not to a final artifact revision.
The Rust command was `cargo test --locked -p sdm-request columns::tests --lib`, under the existing shared heavy guard with `CARGO_BUILD_JOBS=2`.
No broader pass is inferred from these targeted checks.

## Required before completion

- Integrate both audited main commits without restoring legacy runtime computation.
- Verify Rust request/context statistics, pre-trim row restoration, forecast identity, batching, failures and post-holdout fitted-session behavior.
- Verify saved-request round trips and that internal analysis metadata never enters model payloads or their size accounting.
- Verify Python/Java APIs and exact labels through dataframe/Arrow presentation; update generated snapshots and maintained per-symbol coverage.
- Run affected public journeys and bounded endpoint tests against the final candidate; record actual model/connector scope.
- Independently review final boundaries and implementation, merge reviewed branches into Phase 9, and run required integration gates.

Old-client comparisons support the review; independent expectations decide correctness when historical behavior is demonstrably wrong.
