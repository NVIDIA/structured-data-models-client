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
| Provider environment secrets, where needed | `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and AWS credentials; `AWS_REGION` is an environment variable |
| Reachable sources and model endpoints | GitHub-hosted workers must reach every required source, including Spark; otherwise an approved runner/network route is needed |

Configuration is written privately in the runner's temporary directory and deleted after execution.
Only sanitized observations and the generated matrix are uploaded.
Missing access, skipped required cells, failed runners or incomplete comparisons fail the workflow.
Private bridges are identified separately and cannot establish public Databricks session support.

**Setup is still pending:** no dedicated CI credentials or approved runner have been provided yet.
Local endpoint validation can proceed meanwhile; the scheduled workflow is not claimed to be operational.
