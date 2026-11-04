<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Three approved main-parity fixes

Active goal: implement, independently review, test and merge these three efforts into `rust-core-phase-6`.
Follow the [design index](design.md); other main-parity items remain unimplemented.
No merge to `rust-core`/`main` or package publication is authorized by this effort.

| Effort | Owner branch | Status | Required evidence |
|---|---|---|---|
| Wheel provenance | `rust-core-phase-6-wheel-provenance` | Merged: `91758886c`, independently reviewed; 13 tests + real install/negative passed | Wrong-directory/duplicate/missing/source-import negatives; actual candidate artifacts after dependency resolution |
| Java analysis | `rust-core-phase-6-java-analysis` | Merged: `569709e30`; 6 tests + live CodeQL extraction/analysis passed (101 Java sources) | Workflow regression, actual extraction/source coverage and branch analysis; explicit external limitations |
| Temporal-link batching | `rust-core-phase-6-temporal-batching` | Rust fix committed `e74b32178`; native build and public validation running | Public reproduction; shared Rust policy; Python/Java boundaries and partial results; supported live endpoint |

Wheel and Java agents review each other's changes.
Java agent also independently reviews temporal batching; coordinator checks ownership boundaries and final integration.
Tests assert correct contracts rather than blindly copying old behavior.
Only the temporal agent owns native build output; existing heavy guard and Cargo jobs=2 remain in force.
