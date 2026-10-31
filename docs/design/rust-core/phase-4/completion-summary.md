<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 completion summary

The remaining implementation is complete and verified on `rust-core-phase-4`.
This does not claim a merge into `rust-core`, a release, or completion of Phase 5.

## Delivered

| Area | What changed |
|---|---|
| Python and Java Tabular | Single-table input, direct/PQL prediction and fitted sessions |
| Python and Java in-memory Relational | Shared Rust schema preparation, graph sampling, labels, prediction and sessions |
| Python warehouse sessions | Existing public `fit → predict → close` API now uses shared Rust session rules |
| Java warehouse sessions | Warehouse-produced requests use the same fitted-session API and Rust rules |
| Warehouse temporal links | Direct `LIST_DISTINCT` targets now produce link predictions, including fitted sessions |

Java's latest delivery landed at `265495550`.
The earlier Tabular/Python Relational delivery landed at `8eafd69e5`.

```text
Python / Java: input objects, database connections, HTTP, returned objects
                              |
Rust: prepare examples → sample related rows → construct model request
                              |
                     predict or fit
                              |
Rust: check session compatibility, match answers, manage close decisions
                              |
Python / Java: send HTTP requests and present results
```

The host keeps Rust-defined session metadata; the server keeps uploaded examples.
Warehouse SQL, provider operations and sampling time windows are unchanged.
The link fix retains the item table's existing primary key after sampling so ranked items and relationships retain their identity.

## Validation of the remaining work

| Check | Result |
|---|---|
| Warehouse tests | 31 passed on the corrected installed native build, covering DuckDB/SQLite, temporal labels, leakage, large/text IDs, relationships, prediction and sessions |
| Affected Python regression | 226 passed before the item-key correction; the eight failing new link cases were repaired and included in the final warehouse run |
| Rust payload tests | 11 passed |
| Java delivery | Full JDK21 suite: 274 tests, no failures, 13 gated skips; final Java regression: 39 tests, no failures, one skip |
| Java warehouse lifecycle | All 10 DuckDB/SQLite cases passed, including closed JDBC connections, holdout, partial answers and retryable cleanup |
| Shared Rust Java integration | 508 tests passed; contract lock and generated-schema checks passed |
| Live warehouse comparison | Frozen Python and installed Rust client returned exactly equal scalar and temporal-link answers for direct prediction and two fitted predictions; sessions were deleted |
| Reviews | Two independent architecture/code reviews per delivery, plus two reviews of the item-key correction |
| Rust quality | Final core/request Clippy passed with warnings denied; formatting passed |
| Python quality | Changed-file lint/format, client type checks and repository hygiene passed |

Live warehouse checks used local DuckDB data and existing model-serving endpoints through the reviewed validation bridge.
They do not establish every remote warehouse deployment or a new live Java endpoint comparison; Java HTTP lifecycle checks used a local server stub.
The full Python suite was not repeated for this follow-up.
Earlier delivery validation included 7,970 Python passes (13 failures subsequently repaired), 2,573 legacy-engine passes, installed schema/release checks and live Python/Java Tabular comparisons.

## Decisions and remaining scope

- Rust owns sampling and session rules; Python/Java own I/O and native object conversion.
- Preserve meaningful behavior and exact identities, without copying incidental legacy bugs or warning wording.
- Java uses owned Arrow Tables; it is not a zero-copy API.
- No new owner decision is needed for this delivery.
- Merge into `rust-core`, broader platform/performance acceptance, direct Python Arrow input, and Phase 5 release/old-engine removal remain separate.

References: [remaining-work design](remaining-work-design.md), [Java Relational](java-relational.md), [Java Tabular](java-tabular.md), [in-memory Relational](relational-in-memory.md).
