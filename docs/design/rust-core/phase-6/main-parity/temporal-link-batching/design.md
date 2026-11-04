<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Keep temporal-link requests within the model's batch limit

**Gap:** predicting future links for 201 entities can produce one oversized PQL request instead of two valid requests.

## Applicability (source-audited)

| Dimension | Scope |
|---|---|
| Model/task | **Kumo Relational temporal-link prediction only.** Not Tabular, ordinary node classification/regression, or candidate-by-candidate ranking. Forecast decomposition keeps its separate policy. |
| Connectors | In-memory; SQLite, DuckDB, PostgreSQL/Lakebase, Databricks SQL/Spark and Snowflake whenever the existing route admits temporal-link tasks. Files/S3 apply after loading into in-memory tables. This does not add temporal-link support to a route that currently refuses it. |
| Sampling | Shared batch policy before both in-memory and push-down gathering. A smaller batch can change per-call sampling context, so compare correct entities/limits rather than promise identical numerical predictions. No SQL sampling algorithm rewrite. |
| Languages/input | Python and Java PQL materialization; supplied examples are the already-correct regression control. Arbitrary hand-authored wire payloads are not automatically rematerialized by this fix. |

The audit executed memory/DuckDB/SQLite cases; the other connector scope follows the common Rust materializer and still requires coverage, not a claim of completed E2E tests.
Local Python model readers do not currently list native temporal-link prediction; this fix does not add it.

## Why users need it

For example: “Which products will each user buy in the next 30 days?”
The caller asks once; materialization divides the population into model-sized calls and prediction returns combined answers.

| Question | Finding |
|---|---|
| Did main support it? | Yes. `1fc5a6863` preserves a 200-row temporal-link limit while allowing 10,000 ordinary prediction rows per call. |
| Was it documented? | Yes: `docs/reference/limits.md` describes the 200-row limit and `rows_per_call`; the relational skill explains batching. A dedicated notebook is not required for this automatic behavior. |
| What is missing here? | PQL uses the generic limit. Supplied examples already enforce 200, so two public input routes disagree. |

The audit reproduced this through in-memory, DuckDB and SQLite PQL materialization: 201 rows stay in one payload by default or with `rows_per_call=1000`.
With `rows_per_call=100`, they correctly split into 100/100/1.
No endpoint rejection was claimed; the oversized wire request itself was observed.

## Small design

```text
Python / Java: query + optional rows_per_call
    → Rust: determine task limit and clamp requested batch size
    → existing Rust materialization/sampling for each batch
    → host sends calls → Rust combines answers
```

Use one task-aware policy in `sdm-request` from both PQL and supplied-example planning in `sdm-core`.
For temporal links, effective size is `min(requested size or default, 200)`.
Preserve existing positive-value validation, total-call/byte limits, row order and partial-result handling.
Keep forecasting/ranking decomposition rules separate from ordinary entity batching.
No Python/Java API addition, SQL rewrite or new retry/session behavior is needed.
A future configurable endpoint limit is a separate design, not a reason to remove the established safe cap now.

## Tests and acceptance

- Shared policy and public Python/Java tests at 199/200/201, explicit 100 and explicit 1000; 201 must become 200/1 by default.
- Exercise both PQL and supplied examples; ordinary node prediction limits remain unchanged.
- Verify every entity appears exactly once and a second-call failure retains completed results.
- Run a small 201-entity temporal-link prediction against a supporting endpoint; compare IDs/order and semantic results, allowing documented model numerical variation across changed batching.

Evidence: `sdm-core/src/materialize.rs:3816`, `sdm-request/src/options.rs:801`, and existing [supplied-example batching design](../../batching/design.md).

## Implemented and verified

The shared Rust `BatchTask` policy now caps temporal-link PQL and supplied-example batches at 200.
Forecast/ranking preflight, host APIs, wire contract 67 and sampling SQL are unchanged.
Independent review found no ownership or correctness issues.

| Evidence | Result |
|---|---|
| Before-fix public reproduction | Memory, DuckDB and SQLite emitted 201 rows instead of 200/1; the regression test failed as expected. |
| Rust | Two limit tests passed; Clippy for `sdm-request` and `sdm-core`, including tests, passed with warnings denied. |
| Installed Python | 212 focused tests passed; 11 live cases deselected. Includes new boundary/identity/call-limit/partial-result checks, supplied examples and existing forecast/ranking/session controls. |
| Java | 30 focused tests passed; optional live tests run separately. Covers memory/DuckDB/SQLite PQL, supplied examples, forecast/ranking and partial failures. |
| Real model endpoint | Python and Java each passed the maintained shop fixture expanded to 201 users before ingestion: payload sizes 200/1, exactly 201 ordered entity IDs, rankings present. No wire-payload cleanup. |

The production wheel and JNI library were built from `e74b32178699dfd3e4580a18e95e35ac1bf90031`; later changes are tests, ledger and evidence only.
The wheel's manifest records a clean tree, production features and SHA256 `9f35eee4b2e414ae7144b3a22497f64d1ada8694c642ebc48b47d321fae9dfda`.
Artifacts: `/tmp/sdm-p6-temporal-e74b32178/`; test/build logs: `/tmp/p6-temporal-*.log`.
The maintained parity ledger names both language regression suites and both live tests.

**Separate limitation:** the initial minimal synthetic dataset returned endpoint HTTP500 even for two rows.
Its two-row and 200-row payloads are byte-for-byte equal between the before-fix Phase 6 artifact and this artifact, so the batching change did not introduce that behavior.
The cause is not established; feature/history experiments did not establish an explanation and were not retained.
Local boundary tests keep that original dataset; live validation uses the existing shop dataset, identically expanded in Python and Java.
This is not a claim that the endpoint accepts every otherwise valid synthetic dataset.
