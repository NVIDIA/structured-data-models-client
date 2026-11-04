<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Three approved main-parity fixes

Completed: all three efforts are independently reviewed, tested and merged into `rust-core-phase-6`.
Follow the [design index](design.md); other main-parity items remain unimplemented.
No merge to `rust-core`/`main` or package publication is authorized by this effort.

| Effort | Owner branch | Status | Required evidence |
|---|---|---|---|
| Wheel provenance | `rust-core-phase-6-wheel-provenance` | Merged: `e56545a06`; 67 CI/provenance tests + 24 related checks; real installs/negative passed | Wrong-directory/duplicate/missing/source-import negatives; actual candidate artifacts after dependency resolution |
| Java analysis | `rust-core-phase-6-java-analysis` | Merged: `569709e30`; 6 tests + live CodeQL extraction/analysis passed (101 Java sources) | Workflow regression, actual extraction/source coverage and branch analysis; explicit external limitations |
| Temporal-link batching | `rust-core-phase-6-temporal-batching` | Merged: `e669bc9ea`; Python/Java/Rust checks and both live 201-row predictions passed | Public reproduction; shared Rust policy; Python/Java boundaries and partial results; supported live endpoint |

Wheel and Java agents review each other's changes.
Java agent also independently reviews temporal batching; coordinator checks ownership boundaries and final integration.
Tests assert correct contracts rather than blindly copying old behavior.
Only the temporal agent owns native build output; existing heavy guard and Cargo jobs=2 remain in force.

## Final evidence

- Integrated Phase 6 tooling and parity-ledger checks: **79 passed**.
- Temporal-link suites: **212 Python**, **30 Java** and **2 Rust** tests passed; Clippy/Ruff passed.
  Optional live tests were run separately: Python and Java each returned all **201 ordered IDs**, with rankings, from **200/1** payloads against the existing model endpoint.
- Java CodeQL: [branch run41751430654](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41751430654) extracted all **101 shipped Java sources**, analyzed and uploaded successfully.
  The existing Python lane also passed; the unchanged Rust lane was still running at this completion snapshot, so this is not a claim that the entire workflow finished.
- Wheel installation provenance was rechecked with the clean temporal-fix artifact from `e74b32178`.
  Final merged production sources match that tested artifact's source exactly; subsequent differences are CI, tests, ledger and documentation.
- Reciprocal independent reviews approved the final task commits; no public API or wire-contract change was needed.

The initial minimal temporal fixture still produces a server500 even at two rows, with byte-identical pre/post-fix payloads.
Its cause is unproven and recorded in the [temporal design](temporal-link-batching/design.md); no test-specific payload cleanup was used for the successful shop journey.
The reviewed Java alert concerns a long-returning API whose pinned Arrow implementation stores an int row count; no ordinary supported-input defect was established or alert dismissed.
Full release/cross-platform CI was not rerun and nothing was published.

Still outside this completed goal: gradient explanations (including pending PR63 reconciliation), PostgreSQL ordering, DuckDB2 compatibility and the evidence follow-up.
