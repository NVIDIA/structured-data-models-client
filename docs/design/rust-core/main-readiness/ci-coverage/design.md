<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI coverage without a warehouse on every PR

**Status: implemented; hosted acceptance is pending successful execution.**
Earlier Phase 8 full-matrix and benchmark evidence remains in the [verification report](../validation-performance/verification.md) and [performance report](../validation-performance/performance.md).
This smaller recurring CI profile does not replace or relabel those historical results.

```text
PR -> credential-free contracts, local databases and package checks
main / approved manual branch -> reuse full-CI artifacts
                                 + 742 materialization cells / 14 connectors
                                 + separate small pretrained CPU smoke
                                 -> verify results, cleanup and artifact identity
```


This is the current CI plan; it replaces the earlier scheduled full remote-model matrix.
The supplied Snowflake and Databricks API credentials are validated and stored as encrypted `parity-live` secrets.

| Lane | Required work | External dependency |
|---|---|---|
| Every PR | Existing engine/host contracts, local PostgreSQL and generated CSV/Parquet checks | None |
| Required hosted local shards | Full declared materialization matrix for local connectors and S3 emulation | No external service credentials |
| Push to main or approved manual branch run | Real Snowflake and Databricks reader/SQL journeys in both languages, using disposable generated fixtures | Snowflake and Databricks API credentials |
| Small model smoke | Pretrained CPU classification/regression and gradients for both models; Python and Java public paths, plus necessary session checks | Model-weight download access, to verify on a clean runner |
| Optional only | Real S3, Lakebase and giant-value stress | Explicitly configured resources |

Do not multiply pretrained inference across every connector/type combination.
Use deterministic responses for broad client behavior and small real-model checks for model integration; receipts must distinguish those execution modes.
Local S3 emulation and PostgreSQL are not evidence of real AWS or Lakebase service behavior.

Execution steps:
1. Verify clean model downloads and Java CPU timings; select bounded model smoke coverage.
2. Make fixture provisioning portable, add local PostgreSQL/S3 services, and consume dedicated warehouse secrets without exposing values.
3. Update triggers, protected manual execution, coverage accounting and release gates together; remove frozen-client transport dependencies.
4. Run the revised hosted workflow on the implementation branch, review failures and publish concise timing/coverage evidence.

There is no weekly trigger.
The protected environment allows `main` and the explicitly approved `rust-core-phase-8` branch.
Adding another manual branch requires explicit environment-policy approval; ordinary fork PR checks receive no secrets.
Keep release publication and merges into `rust-core` or `main` outside this goal.
The automatic `main` trigger becomes operational after the workflow reaches the default branch; this goal verifies manual execution on the approved implementation branch.

CPU feasibility is verified locally with two CPU threads and GPU visibility disabled.
Four Python pretrained gradient checks passed with a fresh model cache and implicit download authentication disabled: 32.2 seconds process wall time and about 2.6 GiB peak RSS.
Java's four model/task combinations passed through the maintained CPU backend over loopback HTTP: 10.5 seconds test time, 19.9 seconds including Maven startup.
These are local measurements, not hosted-runner timing claims.
The test server uses no archived reference client or Databricks transport bridge.

Implementation now uses a connector execution profile plus separate CPU smoke evidence.
Both must identify the same production CI artifacts; cleanup failures and skipped smoke tests fail acceptance.
Disposable fixture setup/read/cleanup passed locally for PostgreSQL, emulated S3, Databricks SQL/Spark/metric views and Snowflake tables/semantic views.
The combined CPU smoke passed locally: four Python cases and three Java methods, including session coverage, in 27.8 seconds.
The dedicated-credential Java Spark probe also passed all 25 expected outcomes, with whole-journey route checks and disposable-schema cleanup.
Hosted execution is still pending; these local checks do not claim hosted acceptance.

Full CI exposed a pandas 3.1.1 compatibility omission for mixed Decimal/float row labels.
The relevant upstream `pandas/_libs/algos.pyx` is byte-identical to 3.1.0; extending the existing version entry fixed eight reproduced failures, with all 26 ordinary-index checks passing locally.
The Rust sorting algorithm and live acceptance requirements remain unchanged.

Code: [workflow](../../../../../.github/workflows/parity-live.yml), [release gate](../../../../../.github/workflows/publish.yml), [runner guide](../../../../../tests/parity/README.md).

## CI turnaround

Target roughly 10 minutes for a warm run; measure execution separately from hosted-runner queue delays.
Run `41902328500` spent 29.6 minutes in Rust validation and 12–13 minutes in Intel macOS builds.
Manual branch runs were forbidden from saving Rust caches, and their sibling `rust-core` caches were outside GitHub's normal [cache lookup scope](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching).

| Change | Purpose |
|---|---|
| Let trusted manual runs save existing Rust/pip caches | Repeated branch runs can reuse downloaded and compiled dependencies |
| Restore pip downloads before installation in Python jobs | Avoid downloading identical wheels across runs; still resolve dependencies and install the candidate |
| Enable Maven caching in the remaining Java test jobs | Reuse dependencies before compiling tests |
| Compare a cache-populating run with a warm rerun | Verify cache hits and actual time savings without dropping checks |

Cache state is not test evidence: every required assertion still runs, and published artifact receipts remain tied to the candidate commit.
If test execution itself exceeds the target after caching, report that separately; do not claim the target from cache configuration alone.

## Hosted execution findings

The hosted CPU smoke passed four Python cases and three Java methods in 60 seconds on run `41915433899`.
Connector acceptance is still pending: that run exposed a non-Spark Java launcher argument error and incomplete generated semantic-view definitions.
Fix the test setup while retaining every matrix assertion; fixture reads alone do not prove schema construction or materialization.

For readiness dispatch, use `gh workflow run ci.yml --ref rust-core-phase-8 -f fresh=false`.
The optional `fresh=true` mode upgrades dependencies and is a separate compatibility experiment.
Run `41916333063` accidentally used that mode and failed the exact Arrow fixture-version check after upgrading 58.4.0 to 58.4.1; it is not passing readiness evidence.
