<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 3. Structured sanitization reports

**Status: design only. Support both Python and Java through the same Rust report.**
Expose counts Rust already computes when preparing an in-memory relational graph.
This adds observability, not a new data-cleaning policy.

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
If current counters do not retain every required field, extend accounting where filtering already occurs; do not scan the data again.

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
