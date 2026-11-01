<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 completion summary

**The agreed feature work and parity-validation initiatives are delivered on `rust-core-phase-4`.**
This is not yet a claim that every original merge/switch gate passed.
The [acceptance map](acceptance-map.md) separates existing proof from outstanding acceptance evidence.

| Delivered | Scope |
|---|---|
| Python and Java Tabular | Single-table direct/PQL requests, prediction and supported fitted sessions |
| Python and Java Relational | In-memory sampling, warehouse prediction, temporal links, ranking/forecasting and supported sessions |
| Input/API parity | Readers, Spark, Snowflake, schema edits, request options, result access/merge, discovery and persistence |
| Permanent validation | Shared fixtures, API mappings, legacy-policy traps, exact model-input/result comparisons and generated matrices |
| Type fixes | Java Parquet nanoseconds, float32 metadata, shared date conversion and consistent list refusals |

```text
Python / Java: read data and convert native objects
                         ↓
Rust: schema rules → select/sample → build model request
                         ↓
Python / Java: send request to model endpoint
                         ↓
Rust: validate/attribute answer → host presents result
```

## What was verified

- **582 historical comparisons**, including the Java Snowflake follow-up.
- **210 additional raw-reader/type comparisons**: 134 completed prediction journeys and 76 matched, verified refusals.
- Real source/model calls, representative fit/predict/close and persistence journeys, local API/regression checks and independent reviews.
- The latest product fixes were tested on artifact `285c5c053`; final evidence documentation landed at `7449c19c0`.

The [matrices](parity-matrix.md) and [type results](type-parity-status.md) retain exact build identities and limitations.
They do not claim all 792 comparisons ran on one final build, every dtype on every connector, or every deployment/runtime.

## Before closing the original Phase 4 gates

Reconcile the full checkpoint-based contract/differential checks, Databricks runtime matrix and performance requirements listed in the acceptance map.
Existing focused results are useful evidence but do not establish those complete gates.
Dedicated live CI access is explicitly deferred; old-engine removal and release remain Phase 5 work.

No merge into `rust-core` or `main` was performed by this documentation reconciliation.
