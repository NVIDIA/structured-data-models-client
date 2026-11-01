<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 audit: reconciled

The [original audit at `cb64d1cca`](coverage-audit-original.md) identified real gaps that the later parity work addressed.
Feature delivery is complete for the agreed scope; full original merge/switch acceptance is **not yet established**.
See the [acceptance map](acceptance-map.md) for that distinction.

| Original finding | Current disposition and evidence |
|---|---|
| Python in-memory PQL still ran the old validator | Fixed: native paths return Rust-resolved `CoreQuery`; whole-journey legacy-policy traps cover the live comparisons (`9aaeb732e`) |
| Frame schema-edit rules remained in Python | Moved into shared Rust; frame origin and fresh observations are retained (`9aaeb732e`, `01f460072`) |
| Java warehouse ranking/forecast and model-default gaps | Implemented (`f04b23144`); prediction, decomposition and verified unsupported session cases appear in the live matrix |
| Smaller Java direct/options/results/discovery/persistence API | Implemented through `7f7c19f95`; permanent API mappings, local API checks and live restore/result journeys cover the agreed equivalents |
| Missing Java readers, Spark and Snowflake | Readers/Spark landed in `b6196d1fd`; Snowflake followed in `3ec35527d`; both models have applicable live comparisons |
| Remote source/model coverage missing | 582 historical comparisons pass, including real source reads and model calls; builds remain identified per cell |
| Client text-time rules and fallback accounting/collation | Implemented in `2d4021249`; shared provider tests are registered in the permanent manifest |
| Values larger than 2 GiB | Real-model omitted-binary journeys passed for frozen Python, new Python and Java; this does not promise arbitrary huge text features |
| Candidate and input type coverage | Exact candidate/result checks plus 210 additional raw-reader/type comparisons; targeted coverage, not every possible type combination |
| Original slice/switch acceptance | Still requires the explicit evidence reconciliation in the acceptance map; feature comparisons do not replace every original gate |

## What remains intentional

- Tabular reads one table into memory; source-backed `Schema.from_source`/`fromSource` stays Relational-only.
- Public Databricks Model Serving transport still lacks fitted sessions in both languages; session tests use identified session-capable bridges.
- Python/Java retain native object conversion and I/O; tested computation uses Rust.
  Legacy code and routing controls still exist pending Phase 5 removal.
- Java uses owned Arrow tables and data-only snapshots, not pandas objects or Python pickle bytes.
- Dedicated CI credentials/runner deployment remain owner-deferred; local live results are complete.

## Evidence to read

[Original parity results](parity-matrix.md) · [Raw-reader/type results](type-parity-matrix.md) · [Type limits](type-parity-status.md) · [Implementation history](parity-completion-plan.md).
The original matrix has 582 comparisons and the type follow-up adds 210.
These 792 comparisons are accumulated evidence from identified builds, **not one full sweep on the final commit**.
