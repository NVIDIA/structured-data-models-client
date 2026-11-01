<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Running parity continuously

Local PR checks catch manifest, comparison and routing regressions.
The **Live parity** workflow builds installed Python/Java clients and runs the complete manifest against real sources and models, using isolated frozen Python as the baseline.
It runs weekly or on demand on `main` and the approved Phase 4 branch.
The release workflow requires a successful full live run for the exact release commit on `main`.

| Required setup | Purpose |
|---|---|
| Protected GitHub environment `parity-live` | Restrict deployment branches to `main` and `rust-core-phase-4`; use dedicated CI credentials, never personal login copies |
| Environment secret `PARITY_CONFIG_JSON` | The [runner configuration](../../../../tests/parity/README.md), pointing to verified shared fixtures and immutable model versions |
| Repository variable `PARITY_RUNNER_LABELS` | JSON labels for an approved Linux x86_64 runner with at least 32 GiB RAM; preflight requires 24 GiB available, and all giant tests run sequentially |
| Environment variables `PARITY_GIANT_ENDPOINT`, `PARITY_GIANT_MODEL_VERSION` | Real Databricks Relational serving endpoint name and its immutable model version; use the same dedicated workspace credentials |
| Provider environment secrets, where needed | `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and AWS credentials; `AWS_REGION` is an environment variable |
| Reachable sources and model endpoints | GitHub-hosted workers must reach every required source, including Spark; otherwise an approved runner/network route is needed |

Configuration is written privately in the runner's temporary directory and deleted after execution.
Only sanitized observations, the generated matrix and giant-test outcome summaries are uploaded.
Raw giant-test diagnostics remain private on the runner.
After the ordinary matrix, frozen Python, candidate Python and Java each describe, omit and predict with the existing greater-than-2-GiB binary fixture against the real model.
All three must pass on the same workflow commit; the existing release gate therefore covers this resource-intensive acceptance too.
Missing access, skipped required cells, failed runners or incomplete comparisons fail the workflow.
Private bridges are identified separately and cannot establish public Databricks session support.

The `parity-live` environment is created and restricted to the `main` and `rust-core-phase-4` branches.
GitHub accepted [the corrected workflow](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41264242411) and reached its required-configuration check; that run stopped because dedicated credentials are absent.
**Access setup is still pending:** the environment has no secrets and the repository has no registered runners; dedicated CI credentials and an approved high-memory runner are still needed.
Local endpoint validation can proceed meanwhile; the scheduled workflow is not claimed to be operational.
