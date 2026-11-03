<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 3. Structured sanitization reports

**Status: assigned for contract-first implementation. Support both Python and Java through the same Rust report.**
Expose counts Rust already computes when preparing an in-memory relational graph.
This adds observability, not a new data-cleaning policy.
For example: 100 input rows, three excluded for missing keys and two for duplicate keys, leaving 95 usable rows.
Today Rust produces a warning but the client cannot retrieve these numbers as a report; “sanitization” here is not anonymization.
Independent boundary review is incorporated below; this document is the implementation contract.

```text
Input Arrow tables → existing Rust graph preparation
                           ↓
                  graph handle + typed drop counts
                           ↓
           Python / Java prepared request or session report
```

| Layer | Responsibility |
|---|---|
| `sdm-engine` | Existing `Snapshot.dropped` / `DroppedRows` counts and filtering rules |
| `sdm-contract` / `sdm-core` | Include the report in graph-preparation results and propagate it through materialization/session preparation |
| Python / Java | Read-only report accessors; optional text/pandas presentation |

## Proposed contract

Expose `sanitization_report` / `sanitizationReport()` on prepared relational requests and fitted handles, tied to the preparation that produced them.
Use a report status plus per-table input rows, output rows, null-primary-key drops, duplicate-key drops and null-time drops.
Retain zero-drop tables and existing exclusive reason precedence: a row must not be counted twice.
Current `Snapshot.dropped` stores only tables with nonzero drops.
Extend accounting to include clean-table input/output counts where filtering already occurs; do not scan the data again.

Warehouse preparations return an explicit unavailable status when no counts were collected; never imply that unavailable means zero drops.
Define the non-relational case explicitly rather than manufacturing relational statistics for Tabular requests.
Reports describe prepared snapshots, not subsequent edits to the original input.
Copy the small report into the host result so reading it does not depend on a live native handle.
Keep existing warnings and filtering behavior unchanged.

## Acceptance

Use one fixture containing overlapping null-key, duplicate-key and null-time cases plus a clean table.
Assert counts, precedence and input/output totals through both public APIs.
Check report stability across predict, close and source mutation; unavailable warehouse reports must add no queries.
No new model inference or sampling algorithm is required.

## Implementation and review gate

First finalize the public signatures, result shapes, ownership and error behavior here; obtain coordinator contract review before implementation.
Add focused public-API and cross-language tests first where practical, then implement the reviewed behavior.
Use existing parity/snapshot tooling; do not hand-edit generated snapshots or introduce migration-only machinery.
Finish with independent code review, address findings, and run affected tests before merging into `rust-core-phase-6`.

Implementation anchors: `sdm-engine/src/frames.rs` owns `Snapshot.dropped`; `sdm-core/src/materialize/frames.rs` currently emits a warning; `sdm-contract/src/frames.rs` returns the graph handle.
Extend that preparation result, not a separate reporting service or warning parser.
