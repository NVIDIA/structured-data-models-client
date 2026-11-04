<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI coverage without a warehouse on every PR

**Proposal: require deterministic local checks on every PR; run warehouses and pretrained models in a separate, protected acceptance lane.**
This is a plan; the external lane is not operational.

## Verified gaps

Audited November 4, 2026, after the Phase 6 merge.
Successful GitHub metadata queries returned **zero repository secrets, zero `parity-live` secrets, zero organization secrets available to this repository, and zero repository/environment variables**.
No secret values were read.
This establishes missing workflow configuration, not the absence of usable local credentials.

| Dependency | Missing workflow input |
|---|---|
| Snowflake; Databricks SQL/Spark; Lakebase; model endpoints | `PARITY_CONFIG_JSON`, containing connection/endpoint settings |
| Databricks authentication | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` where referenced |
| S3 | AWS access/secret/session credentials and region |
| Giant-value testing | Explicit runner labels, endpoint/version; currently requires 24 GiB available RAM |

The `parity-live` environment allows `main` and obsolete `rust-core-phase-4`, **not `rust-core`**, and has no reviewer rule.
The latest [live attempt](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41264242411) failed at configuration preflight; warehouse/model steps never ran.
The workflow is absent from `origin/main` at `847a1e9e3`; declaring its weekly schedule on `rust-core` does not activate it.
GitHub schedules use the [default branch](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

| Existing machinery | Keep / correct |
|---|---|
| Rust/Python/Java CI, installed wheels/JNI, SQLite/DuckDB/Postgres | Keep real local-database and artifact checks |
| CPU in-process tests | Real tiny random-weight models; not pretrained/GPU evidence |
| API ledger, snapshots, dtype vectors, JUnit checks | Keep declared coverage separate from executed outcomes |
| General live matrix | Unconfigured; separate it from mandatory giant-value testing |
| `manifest.json.live_api_tests` | 16 focused live methods are declared but have no execution consumer |
| Publishing gate | Already requires live success on the exact release commit; incomplete workflow scope can still miss tests |

The last completed pre-merge [Rust CI run](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41565210006) had 81 successful jobs and one intentional skip: **20.6 minutes wall time, 301.5 summed job-minutes**.
These are not billed time or a Phase 6 forecast; Intel macOS wheel building took 20.3 minutes.

## Proposed lanes

```text
PR → local contracts + real local DBs + installed-package checks
       └→ native/package changes: supported-platform smoke checks
trusted commit → protected live shards → evidence by matrix cell
release candidate → exact-commit local + live + all-platform acceptance
```

| When | Coverage | Resources |
|---|---|---|
| Every PR | Both languages/models' contracts; supported tasks; dtype/stype boundaries; local databases; HTTP/session/error fixtures; small CPU models | No private services |
| Native/package/dependency changes | Supported OS/architecture install/load/smoke checks | Public runners/registries |
| Scheduled or maintainer dispatch | Every supported warehouse/reader route, both languages/models where applicable, focused Phase 6 live tests | Dedicated identities, pinned model versions |
| Pre-merge/release candidate | Full applicable matrix and live API cases on the exact candidate; all distribution/runtime targets | Explicitly provisioned capacity |
| Manual stress/benchmarks | Giant values, high memory, GPU stress, performance | Separate dedicated lane |

Keep every required supported cell; reduce fixture size and shard work instead of silently omitting coverage.
Exercise detailed types locally, then actual reader representations and known cross-axis risks live.
Use identical fixtures/model versions across languages, without repairing reader output before prediction.
Execute the live registry through existing pytest/JUnit accounting; missing tests, skipped methods and unavailable endpoints remain incomplete.
Gradients require a supporting endpoint or actual pretrained runtime, not merely a prediction endpoint.

## Resource and security controls

Reuse Actions matrices and maintained runners; build each wheel/JNI once per target and share artifacts.
Shard by connector, cap model/GPU concurrency separately, and choose budgets/timeouts after measuring one complete baseline.
Preflight authentication/capabilities before expensive work; retry bounded infrastructure failures, not assertions.
Do not deploy GPUs per PR.
Use dedicated least-privilege identities, read-only fixtures and isolated scratch namespaces; retain cleanup and sanitized evidence on failure.
Prefer short-lived cloud credentials where supported ([OIDC guidance](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments)).
Never run unreviewed fork code with live credentials or privileged `pull_request_target` checkout.
Correct environment branch rules; approve manual candidates, and use a narrowly scoped trusted environment for unattended schedules ([environment controls](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)).

## Next changes

1. Configure dedicated identities, fixture access and runner/model settings; verify access without logging secrets.
2. Split ordinary live coverage from stress tests, and execute every declared live API requirement.
3. Measure one trusted run, set budgets/cadence, and require complete exact-candidate acceptance before `main` and publication.

Code: [CI](../../../../../.github/workflows/ci.yml), [live workflow](../../../../../.github/workflows/parity-live.yml), [release gate](../../../../../.github/workflows/publish.yml), [runner guide](../../../../../tests/parity/README.md), [ledger](../../../../../tests/parity/manifest.json).
