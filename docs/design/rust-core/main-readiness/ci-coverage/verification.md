<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hosted CI acceptance

**Passed on 2026-11-05**, testing commit `3cb611a051dabff024f7847c5e00473e18b313bf`.
This report is documentation of that tested commit, not a claim that subsequent commits automatically inherit its release approval.

| Evidence | Result |
|---|---|
| [Full CI](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41929869913) | 81 jobs passed; 32.9 minutes including scheduling |
| [Hosted acceptance](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41934656118) | All 17 jobs passed; 38.0 minutes including scheduling |
| Required matrix | 742/742 materialization cells across 14 connectors, Python/Java and both models |
| Real warehouses | Databricks SQL, Spark and metric views; Snowflake tables and semantic views |
| Local services | Disposable PostgreSQL; generated files and in-process S3 emulation |
| Pretrained CPU smoke | Four Python cases and three Java methods; classification/regression gradients for both models and Java sessions; 55.9 seconds |
| Cleanup and identity | All 14 connector receipts completed; every shard and CPU receipt identifies the same production CI artifacts |

The downloaded aggregate was independently replayed with the maintained verifier and passed.
The `hosted-parity-3cb611a051dabff024f7847c5e00473e18b313bf` artifact contains the report and observations; individual shard artifacts contain execution and build receipts.
Route checks cover the complete observed journeys and reject legacy computation or unobserved subprocesses.
Java connector journeys load the packaged JAR; Java CPU smoke compiles the same checked-out source and uses the verified packaged JNI library, as explicitly recorded in its receipt.

## Scope and controls

- Ordinary PR checks need no warehouse credentials.
- Warehouse acceptance runs after successful main CI or approved manual branch dispatch, without a weekly schedule.
- The protected environment allows only `main` and `rust-core-phase-8`; it contains dedicated Databricks and Snowflake credentials, not user/session credentials.
- The matrix performs materialization without model calls; actual inference is confined to the separate CPU smoke.
- Lakebase's 50 cells remain explicitly excluded, not passed; real AWS S3 is optional and is not certified by emulation.
- Release gates still require successful acceptance for the exact release commit on main.
- The main trigger becomes operational once the workflow is merged there; this acceptance used the approved manual implementation branch.

No release was published and nothing was merged into rust-core or main.
Earlier [Phase 8 validation](../validation-performance/verification.md) and [performance evidence](../validation-performance/performance.md) remain separate and unchanged.

## Runtime and follow-up

The 10-minute aspiration is **not achieved**.
Confirmed Rust cache reuse reduced the Rust job from 29.5 to 15.4 minutes in the [warm verification run](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41925746800); its main test step still took 11.7 minutes.
Linux uses standard GitHub-hosted runners, not a larger runner.
In final hosted acceptance, Spark materialization took 17.0 minutes, Databricks SQL 7.1, metric views 6.1, and Snowflake journeys 2.6–3.4, excluding installation and queueing.
Connector shards remain capped at two; the proposed parallelization change was reverted at the owner's request.

After merging into main, consider larger runners, safe test sharding, increased connector concurrency, and artifact reuse keyed to unchanged build inputs rather than rebuilding for fixture-only commits.
These are follow-ups, not additional acceptance claims or changes made in this pass.
