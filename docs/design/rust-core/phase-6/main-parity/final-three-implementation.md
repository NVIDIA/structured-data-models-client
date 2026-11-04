<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Final three main-parity implementations

Active goal: implement, independently review, test and merge the approved DuckDB2, explanation and PostgreSQL fixes into Phase 6.
Follow the [design index](design.md) and the three linked task designs.
Implementation baseline: `21c3a3f7b`; main reference: `847a1e9e3`, including gradients and `EXPLAIN PREDICT`.
PR #63 remains open at `f240fe448`; its unfinished cohort/container API is not silently included as approved scope.

| Effort | Branch | Status | Required evidence |
|---|---|---|---|
| DuckDB2 | `rust-core-phase-6-duckdb2` | Merged `c8b94150d`; 252 tests per driver, 4 live model journeys, Java/3.10 checks passed | Python1.5/2 reads, SQL/catalog/types/ownership/error behavior; Java2 shared SQL; live read→predict |
| PostgreSQL ordering | `rust-core-phase-6-postgres-order` | Merged `d07bf1369`; Python 19 and Java 9 checks passed, including real database-to-model predictions | Public failure reproduction; JSON/XML/domain keyless ordering; Python/Java; other dialects unchanged; model smoke |
| Explanations | `rust-core-phase-6-gradients` | Contract accepted; Rust decoder/host views in progress | Rust validated decoding/identity/session/decomposition; Python/Java APIs and presentation; optional Python runtime; supported live model checks |

Rust owns engine semantics, including SQL dialects, EXPLAIN interpretation, gradient decoding and result identity.
Hosts own connections, transport, presentation and optional Python model execution.
Tests assert correct outcomes rather than preserve old defects; reuse maintained fixtures and targeted comparisons without a new migration harness.

DuckDB and PostgreSQL agents review each other; PostgreSQL agent and coordinator review explanation boundaries and implementation.
Native contract versions are allocated serially by the coordinator.
PostgreSQL gets the first warm native build; gradients use the target after it is released and changes integrated.
Maximum two heavy jobs through the existing guard, Cargo jobs=2.
No publication or merge into rust-core/main.
