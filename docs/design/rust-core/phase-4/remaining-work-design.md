<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Remaining Phase 4 work: proposed design

**Java in-memory Relational is delegated for implementation; the warehouse sections below are investigation and proposed follow-ups.**
Warehouse changes are separate work; the restart explicitly preserved those paths.

## 1. Java in-memory Relational

Today Java accepts one table for Tabular; Rust already has the multi-table sampler used by Python.

```text
Java: Arrow tables + relationships + query
                   |
                   v
Rust materialize(): build/reuse graph, choose examples and related rows
                   |
             Prepared request
              /           \
       predict()          fit(): upload examples once
           |                    |
           |              fitted.predict(): send new rows
           |                    |
           +------> Server <----+       close(): release server session
                       |
           Rust checks and matches answers
                       |
              Java returns results
```

- Extend `Schema.fromData()` to multiple Arrow Tables using existing key/time/relationship APIs.
- Select the model explicitly at materialization; preserve existing Tabular calls.
- Java/JNI owns conversion, HTTP and native lifetimes; reuse Rust schema, graph, sampling and session logic.
- A retained materializer owns its graph/random state; snapshot caller data, serialize use and release resources explicitly.
- Verify Python/Java agreement, temporal/link/ranking/forecast cases, sessions, mutation, concurrency and cleanup.

SQL providers and push-down sampling remain unchanged.

## 2. Warehouse sessions: precisely what moves

**The public Python API stays unchanged.**
I verified a DuckDB `materialize → fit(holdout=[3]) → predict → close` journey using a simulated endpoint: it succeeds, but calls Python's `session_plan`, `session_prediction_plan` and `normalize`, without creating Rust session state.

| Public API stays in Python | Rules to execute in Rust |
|---|---|
| `client.fit(request, ..., holdout=...)` | Validate one reusable context, exclude held-out entities and their related rows, select upload fields, validate the returned session ID |
| `fitted.predict(request)` | Check the pinned question/schema, detect examples also being scored, send only scoring data, match answers to rows, combine batches and preserve partial answers on failure |
| `fitted.close()` | Decide whether deletion is needed; mark closed only after success so failed deletion remains retryable |

**Also stays in Python:** DataFrame conversion, public `Request`/`FittedModel`/`PredictionResult` objects, credentials, HTTP and displaying warnings/errors.
The fitted object keeps the session ID and Rust-defined metadata; the server keeps uploaded examples.
Python does not decide sampling or session compatibility.

```text
Warehouse → existing materialize() → prepared request
                                      |
                          Rust: fit / score / close rules
                                      |
                          Python HTTP → model server
                                      |
                          Rust checks reply → Python result
```

Reuse the existing Rust Relational session operations, currently named `frame_session`; generalize only frame-specific assumptions after checking warehouse payloads.
Do not introduce a second session registry, change sampling SQL, or claim the existing API was missing.
Verify held-out rows, temporal keys, empty scoring, incompatible questions, batching and failed-close retries against the old client.

**Java fast follow:** allow warehouse-produced `Request` objects in `SdmClient.fit()` and use the same Rust session contract through Java HTTP/results.
Today Java explicitly refuses those requests; no separate Java session-policy implementation is proposed.
Endpoint session support remains necessary in both languages.

## 3. Warehouse temporal links: reproduced gap

Example from a real DuckDB reproduction, with users, orders and items:

```sql
PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days)
FOR EACH users.user_id
```

Meaning: **“Which items will each user order over the next 30 days?”**

| Same fixture and query | Observed materialization result |
|---|---|
| Frozen Python client | Produces a temporal-link request |
| Current Rust client | Refuses with `rule="aggregation"`, before any model call |
| Add `RANK TOP 2` and explicit candidates | Both succeed through candidate-by-candidate classification; this is a different path |

Connect and verify the existing list-label, item-identity and user/item-context pipeline for warehouse reads; do not treat removal of the refusal as proof of support.
The code already reads these rows through bounded `Neighbors` requests, so no new SQL aggregate or provider operation appears necessary; connected-path tests must confirm this.
Compare labels, time windows, sampled related rows and final responses with the old client.
Other warehouse sampling paths must retain their behavior.

**Text timestamps are not this gap:** built-in warehouses already select the supported string-date parser.
A temporal COUNT reproduction using string timestamps succeeded in both clients.
The surviving `ClientRules` refusal concerns an unused/custom-provider mode.

## Audit sources

Original scope: [migration plan](../migration-plan.md).
Delivery constraints: [restart](restart.md), [boundaries](boundaries.md), [Java Tabular](java-tabular.md) and [Relational](relational-in-memory.md).
The restart preserves warehouse boundaries but does not explicitly waive the full phase exit.
The conclusions above come from code traces and public reproductions, not past memory; no current `phase-4/design.md` exists here.
Old-engine removal and release packaging remain Phase 5.
