<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Connection and execution conveniences

The three approved follow-ups are implemented in Python and Java.
[Completion evidence](followup-completion.md) records the passing combined validation and its limits.
The original comparison used old client commit `12955d397` and Phase 6 `6a411130b`.

## 1. Replace a database connection

Push-down sampling needs a database connection; an already prepared request normally needs only the model endpoint.

```text
Replacement connection + existing schema declarations
                         ↓
Rust checks tables/types and preserves declarations
                         ↓
New schema → SQL sampling → prepared request → model
```

Use Python `schema.with_source(source)` or Java `schema.withSource(source)`.
This keeps keys, relationships, time roles and calculated columns without rebuilding the schema by hand.
The backend, namespace and semantic-model settings must match.
Neither connection is closed or modified; the original schema and prepared requests stay unchanged.
It also works when the old connection has already closed.
Rust revalidates source metadata without fresh inference or a full key-uniqueness scan.
The old client mutated the existing object and could close its old connection.

See [rebinding design and tests](rebinding/design.md).

## 2. Batch predictions and retry failures

| Behavior | Old client | New client |
|---|---|---|
| Batch size | Settings inside `batch_mode(...)` | Explicit `rows_per_call` / `rowsPerCall`, including supplied examples |
| Retries | Transport retries plus an optional batch retry loop | One bounded transport policy; HTTP 408 included, HTTP 500 still excluded |
| Repeated context | Automatic temporary sessions for eligible jobs | Rust reuses a session only when complete context, task, schema and inference/output semantics match |
| Expired/unsupported session | Recovery/fallback | One recreation after expiry; unsupported creation falls back to stateless prediction |
| Later batch fails | Earlier answers were lost from the call | Error retains completed answers; cleanup cannot hide the primary failure |

Explicit `fit → predict → close` remains available for caller-managed sessions.
Automatic reuse is request-local, not a cross-request cache.
Uncertain session creation is not blindly retried.
Different contexts and unsupported transports retain stateless prediction.
Rust owns eligibility and lifecycle decisions; Python/Java own HTTP and backoff.
No warehouse sampling algorithm changes.

See [batching design and tests](batching/design.md).

## 3. Typed settings and explanation summaries

| API | What it does | Boundary |
|---|---|---|
| `InferenceConfig` | Named, immutable inference options with autocomplete; plain mappings remain supported | Hosts convert options; Rust supplies defaults and validates them |
| `ExplanationView` | Reads existing summary, cohorts, subgraphs and feature importance | Already restored in both languages |
| `ExplanationSummarizer` | Explicitly sends explanation data to the caller's configured text endpoint when prose is absent | Host-only HTTP; no Rust sampling or automatic extra call |

The summarizer preserves existing prose and never mutates the explanation.
It transmits raw explanation values only when explicitly called.
Old engine configuration classes/imports are not reintroduced.
See [settings and summaries design](settings-summaries/design.md).

## Verification and ownership

Three agents implemented these independently; the coordinator reviewed contracts and arranged reciprocal code review before integration into Phase 6.
Tests exercise public APIs, independently expected values, ownership, malformed inputs and lifecycle failures.
Rebinding uses real SQLite/DuckDB databases and existing warehouse/model endpoints.
Batching compares automatic-session and stateless predictions from the same prepared request.
Controlled HTTP servers cover failures that a live endpoint cannot reliably reproduce.
Summary HTTP behavior is tested locally; no authorized external text-model endpoint was available, so real LLM validation is not claimed.
The maintained API ledger and generated snapshots include the new APIs.
No new endpoint deployment, publication or merge into `rust-core`/`main` is part of this work.
