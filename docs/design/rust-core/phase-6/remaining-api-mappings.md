<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Remaining API differences, explained

**Approved follow-up: add connection rebinding, improve batch/session convenience, and restore useful typed settings and optional prose summaries.**
This is a code audit of old commit `12955d397` versus Phase 6 `6a411130b`, not a new runtime test run.
The owner approved implementation; the following sections explain the audited starting point.

## 1. Connection rebinding

**Your understanding is correct.**
Push-down sampling needs a live database connection to run SQL.
An already materialized request normally needs only the model endpoint; gathering new examples needs the database again.

```text
Database connection → SQL sampling → prepared Request → model endpoint
```

The old `update_connection()` replaced connections inside an existing graph/model, preserving its table declarations.
The new API constructs a source and schema from the new connection.

Use one application function to rebuild consistently:

```python
schema = make_schema(new_connection)  # Your function: source + tables + overrides
```

**Approved direction:** add rebinding that preserves keys, relationships, time roles and calculated columns, preferably returning a new schema with a new source.
The trade-off is cost: schema construction repeats catalog/preview work and can scan tables to validate unenforced keys.
If that becomes expensive, consider returning a new schema that reuses and revalidates existing declarations.
Connection ownership belongs in Python/Java; Rust continues requesting SQL through the provider.

## 2. Scoped retries and batching

Imagine scoring many customers: batching splits the job into smaller model calls; retries repeat a failed call.
“Scoped” meant settings applied only inside a `with` block, then reverted.

| Behavior | Old client | Current client |
|---|---|---|
| Batch size | `with model.batch_mode(...)` | Automatic splitting; PQL callers can set `rows_per_call` / Java `rowsPerCall` |
| Retries | Transport retries plus an optional per-batch retry loop | Client/transport retry policy with backoff |
| Avoid uploading context repeatedly | Eligible batched jobs automatically opened and closed a temporary model session | Ordinary batched prediction is stateless; callers explicitly use `fit → predict → close` for pinned context |
| Later batch fails | Raised without returning earlier results from that call | Error carries completed results |

Old block settings were isolated per execution context, but stacked retries could multiply attempts.
Current retries cover selected transient failures and avoid blindly repeating session creation; the old outer loop additionally retried HTTP408/500.

**Recommendation:** keep explicit batch options and one centralized retry policy.
Add the missing batch-size option for supplied examples if users need that control.
Automatic context reuse is a separate performance gap: the old client was more convenient here.
Explicit `fit()` does not fully replace it, because fitting a request with multiple contexts is refused.
If restored, reuse must require identical context/task/schema and define session-expiration recovery and stateless fallback.
Rust decides whether requests can share context; hosts own HTTP retries and connections.

## 3. Typed configuration and explanations

| Item | Plain meaning | Status / recommendation |
|---|---|---|
| Typed configuration | Named settings objects/enums with autocomplete | Current keywords/mappings and Java builders cover configuration. Add typed conveniences where useful; defaults and validation stay in Rust. |
| Explanation view | Read/print the model's summary, cohorts, subgraphs and feature importance | Restored as `ExplanationView` in both languages. This is no longer a gap. |
| Generated prose summary | Send structured explanation details to another LLM to write a summary | Old client could do this when prose was absent. Current views only display returned prose; generation is still missing. |

Old `ExplainConfig` only controlled `skip_summary`; it was not the configuration for all explanation computation.
**Recommendation:** if prose generation is wanted, expose an explicit optional summarizer with endpoint configuration in the host languages.
It is a second model call, not sampling-engine work.

## Code anchors

| Topic | Current code | Old engine code at the pinned commit |
|---|---|---|
| Connections | [Schema.from_source](../../../../packages/structured-data-models-client/src/sdm_client/schema.py), [Source.connection](../../../../packages/structured-data-models-connectors/src/sdm_connectors/source.py) | `rfm/graph.py:3567`, `rfm/rfm.py:1959` |
| Batching/retries/sessions | [Rust prediction](../../../../rust/crates/sdm-core/src/predict.rs), [transport](../../../../packages/structured-data-models-client/src/sdm_client/core/transport.py), [frame sessions](../../../../rust/crates/sdm-core/src/frame_session.rs) | `rfm/rfm.py:736,764,1233,1247,1319` |
| Settings/explanations | [Rust options](../../../../rust/crates/sdm-request/src/options.rs), [Python views](../../../../packages/structured-data-models-client/src/sdm_client/reports.py), [Java views](../../../../java/src/main/java/com/nvidia/sdm/ExplanationView.java) | `rfm/inference/config.py:20`, `rfm/explain/result.py:19`, `rfm/explain/summary.py:312` |

## Implementation and tests

Three agents own implementation and tests on separate branches from `rust-core-phase-6`.
The coordinator reviews API contracts before implementation, assigns native contract/build ownership, and merges only reviewed work.

| Effort | Boundary | Required public-workflow checks |
|---|---|---|
| Connection rebinding | Hosts own connections; retain schema declarations and use Rust admission/provider validation | Python/Java memory refusal, SQLite/DuckDB replacement, preserved keys/links/types/time/end/SQL columns, incompatible sources, original-object and connection ownership; read-only warehouse reconnect → materialize → live predict |
| Batching and retries | Rust decides batch/context compatibility and session transitions; hosts perform transport/backoff | Explicit batch size including supplied examples, identical vs different contexts, one context upload when eligible, expiration/recreation, unsupported-session fallback, no duplicate session creation, cleanup after failure, successful partial results; Python/Java and real session-capable endpoint |
| Typed settings and summaries | Thin Python/Java option wrappers; Rust retains defaults/validation; optional summarizer transport belongs to hosts | Typed/mapping equivalence, invalid settings, immutable inputs, matching host APIs; summary generation from structured explanations, no implicit calls, supplied prose preservation, malformed/failing responses and deterministic HTTP contract tests |

Use small shared expected fixtures and focused old-client comparisons, not a new differential-testing framework.
Document and test improvements over old behavior, including bounded retries and connection ownership.
For summary integration use synthetic explanation data; live provider verification requires an existing authorized summarizer endpoint, otherwise state the external validation limit explicitly.
No endpoint deployment or publication is required by this plan.
Each effort receives independent review; run affected checks and final combined production-artifact tests.
Maintain the API ledger and generated snapshots using existing tools.
Use the existing heavy guard, at most two heavy jobs, Cargo jobs=2, and one owner of the shared native target.
Push reviewed work to Phase 6 only; preserve unrelated branches and leave `rust-core`/`main` untouched.
