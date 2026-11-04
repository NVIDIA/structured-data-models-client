<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI coverage without a warehouse on every PR

**Implementation: require deterministic local checks on every PR; separate ordinary live acceptance from explicit stress acceptance.**
Dedicated CI credentials remain owner-deferred; the protected external lane is not operational.
This work changes repository configuration and runners only, without uploading personal credentials or changing remote environments.
Local candidate acceptance can use existing authorized resources; final correctness and performance checks run after the repository-size and developer-contract changes are integrated.

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
| General live matrix | Dedicated CI remains unconfigured; ordinary acceptance is separate from manual giant-value testing |
| `manifest.json.live_api_tests` | Executed through maintained pytest/JUnit accounting; missing prerequisites stay incomplete |
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

1. Repository implementation separates `parity-live.yml` and manual `parity-stress.yml`, and publication requires both on the exact candidate.
Both lanes share a repository-wide concurrency group with capacity one.
The live workflow derives serial connector shards from the maintained manifest, requires every shard artifact, and runs the unfiltered comparator before the focused registry.
Both lanes reuse successful full-CI production artifacts for the exact commit, verifying wheel/native and JAR/POM identities.
The matrix loads the verified JAR; focused Java registry and giant tests use the exact clean Java source commit with the verified packaged JNI, explicitly labeled source-equivalent execution.
Pin the same optional `ci_run_id` for both lanes to retain identical artifact bytes even if CI is rerun on that commit.
Preflight checks authentication, bounded fixture reads and endpoint readiness before acceptance work; optional remote model listing is recorded separately and does not certify task, session or gradient support.
2. `tests/parity/live_api.py` executes the registry through the existing Python/JUnit runners, preserving private diagnostics and reporting missing resources without connection values.
3. Dedicated identities and environment policy remain owner-deferred; no personal credentials are uploaded by this work.
4. Final local acceptance and bounded benchmarks run after integration; measure a configured trusted CI run before selecting operational budgets or claiming scheduled coverage.

Code: [CI](../../../../../.github/workflows/ci.yml), [live workflow](../../../../../.github/workflows/parity-live.yml), [release gate](../../../../../.github/workflows/publish.yml), [runner guide](../../../../../tests/parity/README.md), [ledger](../../../../../tests/parity/manifest.json).

## Final hosted-CI revision

This revision supersedes the required full remote matrix described above; previous Phase 8 evidence remains unchanged.
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

Implementation order:
1. Verify clean model downloads and Java CPU timings; select bounded model smoke coverage.
2. Make fixture provisioning portable, add local PostgreSQL/S3 services, and consume dedicated warehouse secrets without exposing values.
3. Update triggers, protected manual execution, coverage accounting and release gates together; remove frozen-client transport dependencies.
4. Run the revised hosted workflow on the implementation branch, review failures and publish concise timing/coverage evidence.

Remove the weekly trigger.
Only trusted branch code approved for secret access may run the warehouse lane; ordinary fork PR checks receive no secrets.
Keep release publication and merges into `rust-core` or `main` outside this goal.

CPU feasibility is verified locally with two CPU threads and GPU visibility disabled.
Four Python pretrained gradient checks passed with a fresh model cache and implicit download authentication disabled: 32.2 seconds process wall time and about 2.6 GiB peak RSS.
Java's four model/task combinations passed through the maintained CPU backend over loopback HTTP: 10.5 seconds test time, 19.9 seconds including Maven startup.
These are local measurements, not hosted-runner timing claims.
The test server uses no archived reference client or Databricks transport bridge.

Implementation now uses a connector execution profile plus separate CPU smoke evidence.
Both must identify the same production CI artifacts; cleanup failures and skipped smoke tests fail acceptance.
Disposable fixture setup/read/cleanup passed locally for PostgreSQL, emulated S3, Databricks SQL/Spark/metric views and Snowflake tables/semantic views.
The combined CPU smoke passed locally: four Python cases and three Java methods, including session coverage, in 27.8 seconds.
Hosted execution is still pending; these local checks do not claim hosted acceptance.
