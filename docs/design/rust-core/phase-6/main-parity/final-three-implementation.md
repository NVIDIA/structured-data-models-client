<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Final three main-parity implementations

Completed: all three approved implementations are independently reviewed, tested and merged into `rust-core-phase-6`.
**Follow-up audit:** the [main merge plan](../../main-readiness/main-merge/design.md) records four additional reproduced contract gaps and a generated-key concern requiring verification.
The evidence below remains valid for the tested scope; it does not establish complete main parity.
Follow the [design index](design.md) and the three linked task designs.
Implementation baseline: `21c3a3f7b`; main reference: `847a1e9e3`, including gradients and `EXPLAIN PREDICT`.
PR #63 remains open at `f240fe448`; its unfinished cohort/container API is not silently included as approved scope.

| Effort | Branch | Status | Verified scope |
|---|---|---|---|
| DuckDB2 | `rust-core-phase-6-duckdb2` | Merged `c8b94150d`; 252 tests per driver, 4 live model journeys, Java/3.10 checks passed | Python1.5/2 reads, SQL/catalog/types/ownership/error behavior; Java2 shared SQL; live read→predict |
| PostgreSQL ordering | `rust-core-phase-6-postgres-order` | Merged `d07bf1369`; Python 19 and Java 9 checks passed, including real database-to-model predictions | Public failure reproduction; JSON/XML/domain keyless ordering; Python/Java; other dialects unchanged; model smoke |
| Explanations | `rust-core-phase-6-gradients` | Merged `f07dfd1b4`; final Python/Java artifacts passed, including four real-model journeys per language | Rust validated decoding/identity/session/decomposition; Python/Java APIs and presentation; optional Python runtime; supported live model checks |

Rust owns engine semantics, including SQL dialects, EXPLAIN interpretation, gradient decoding and result identity.
Hosts own connections, transport, presentation and optional Python model execution.
Tests assert correct outcomes rather than preserve old defects; reuse maintained fixtures and targeted comparisons without a new migration harness.

DuckDB and PostgreSQL received reciprocal reviews; explanations received independent engine and host reviews.
Native contracts 68 and 69 were allocated serially, using the shared warm target and existing heavy-job guard.

Final gradient checks: 146 Rust unit tests, 11 schema/integration checks, 85 Python regressions, 8 targeted public Python tests and 193 skills checks passed.
Four actual pretrained model/task combinations passed in each language on clean production artifacts from `62df972a8`; the final follow-up commit only records evidence.
The remote endpoint does not serve gradients, so real-model validation used the local GPU runtime and a temporary HTTP bridge for Java.
The bridge is stopped and the GPU released.
Post-merge verification: 101 workflow/ledger tests passed; two optional Snowflake-driver probe checks skipped because that driver is absent in this verification environment.
The API ledger passed with 240 Python and 534 Java symbols; merged product/test files match the tested artifact source.
See the linked task designs for detailed evidence and retained tests.
No publication or merge into `rust-core`/`main`.
