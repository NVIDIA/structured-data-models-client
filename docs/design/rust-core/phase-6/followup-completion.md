<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Connection and execution follow-up: complete

All three approved efforts are independently reviewed, tested and integrated into `rust-core-phase-6`.
Nothing was published or merged into `rust-core` or `main`.

| Feature | Delivered in Python and Java | Improvement / boundary |
|---|---|---|
| Database rebinding | `Schema.with_source` / `withSource` | Independent schema; no connection closure or mutation. Rust revalidates existing declarations. |
| Batching and retries | Supplied-example batch control and automatic temporary sessions for identical contexts | One retry policy, bounded expiry recovery, safe fallback, preserved partial answers and primary errors. Rust owns lifecycle decisions; hosts own HTTP. |
| Settings and explanations | Immutable `InferenceConfig` and explicit `ExplanationSummarizer` | Rust retains option validation. Summary generation is opt-in host I/O and preserves existing prose. |

## Verification

The final combined production wheel passed **63 Python tests**; the combined Java classes passed **20 tests**.
Ten Python and two Java credentialed cases were excluded from these local runs; live verification was performed separately below.
The wheel has test-only operations disabled, and its installed API matches the generated snapshot.
Artifact source: `b6e238bb4`; later changes only record evidence.

Feature-specific runs overlap the combined checks; do not add these counts together.

| Area | Evidence |
|---|---|
| Rebinding | 23 local Python and three local Java checks; five live Python warehouse/model journeys and one live Java PostgreSQL/model journey |
| Automatic sessions | 92 Rust, 129 local Python and 44 local Java checks; live Python/Java endpoint workflows |
| Session equivalence | Same prepared request through session and stateless paths; row IDs/alignment checked independently; predictions matched at `rtol=1e-5`, `atol=1e-6`; observed one create, three scores and one close |
| Typed settings / summaries | 22 Python and six Java checks; real local HTTP exchanges for success, malformed data, timeout/errors, credentials and no-call cases |
| Repository contracts | API ledger: 238 Python / 516 Java symbols; native schema/attachment and nine wire checks; lint, targeted typing and repository hygiene passed |

The live rebinding sources were PostgreSQL, Databricks SQL, Snowflake, Databricks metric views and Snowflake semantic views.
Tests also cover closed old connections, incompatible replacement types, unchanged prepared requests, distinct-context fallback, retry limits and cleanup failures.
A reproduced temporal-link batching bug was fixed: supplied prediction batches are capped at 200 rows.
No warehouse sampling algorithm changed.

No authorized external text-model endpoint was available.
The summarizer's HTTP contract is tested, but real LLM integration and generated prose quality are not claimed.
Old engine imports, mutable block-scoped settings and standalone Python parser internals remain retired.
See [API explanations](remaining-api-mappings.md) and the three linked designs for exact behavior.
