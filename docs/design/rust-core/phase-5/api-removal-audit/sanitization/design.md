<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 3. Structured sanitization reports

**Status: implemented, independently reviewed and merged into `rust-core-phase-6`.**
Expose counts Rust already computes when preparing an in-memory relational graph.
This adds observability, not a new data-cleaning policy.
For example: 100 input rows, three excluded for missing keys and two for duplicate keys, leaving 95 usable rows.
Previously Rust produced only a warning; the new client also exposes these numbers as a report; “sanitization” here is not anonymization.
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

## Exact contract for review

`Request.sanitization_report` and `FittedModel.sanitization_report` return an immutable `SanitizationReport` in Python.
Java exposes equivalent `sanitizationReport()` accessors and immutable `SanitizationReport` / `TableSanitizationReport` records.
Python exports frozen dataclasses with the same names; `tables` is a tuple in Python and a defensively copied list in Java.

| Report field | Meaning |
|---|---|
| `status` | `available` for prepared in-memory relational graphs; `unavailable` for warehouse or older saved requests; `not_applicable` for Tabular |
| `tables` | Ordered table records; empty unless status is `available` |
| Per table | `table`, `input_rows`, `output_rows`, `null_primary_key_rows`, `duplicate_primary_key_rows`, `null_time_rows` |

Counts describe discarded rows, not missing cells or endpoint holdouts.
Every prepared table appears, including clean tables.
Each discarded row has exactly one reason: missing key first, duplicate key second, missing time third.
A first occurrence with missing time still reserves its key, matching the current engine behavior.
`output_rows = input_rows - null_primary_key_rows - duplicate_primary_key_rows - null_time_rows`.
Tables without a primary key have zero key-drop counts.
Preparation failures continue to raise the existing error and produce no successful request/report.

Contract version **58** adds typed available table records to `PreparedFrameGraph` alongside its graph attachment.
Rust collects counts in the existing filtering pass, including clean tables, and derives the existing warning from nonzero records only.
There is no report-fetch operation, no extra table read, and no new sampling or cleaning policy.
Host adapters only decode immutable values and classify paths where graph preparation does not run.

The materializer copies the preparation report onto each request made from its cached graph.
A fitted handle retains its fit request's report independently of native graph/session lifetime; prediction does not replace it.
Closing a handle or mutating the original source does not change the report.
Existing request persistence retains reports; older relational snapshots yield `unavailable`, never invented zero counts.
Older Tabular snapshots remain `not_applicable`.
Tabular stays `not_applicable`, including its fitted handles.

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

## Usage and verification

Read `request.sanitization_report.tables` in Python or `request.sanitizationReport().tables()` in Java.
Fitted handles expose the same report, including after close.
These counts describe graph preparation, not the rows selected into an individual prediction context.

Focused checks cover Rust filtering accounting, Python/Java materialization and sessions, request persistence, immutable reports, warehouse unavailability and Tabular non-applicability.
The maintained API ledger maps the new surface to these tests under `sanitization-reports`.
Java admission rejects missing primary keys before graph preparation; Java public tests exercise duplicate/time overlap, while shared Rust and Python tests cover missing-key precedence.
No model inference algorithm or warehouse query was changed.

Validation: 321 Rust tests, 73 Python lifecycle/materialization tests plus the legacy-snapshot check, Java relational lifecycle tests and malformed-snapshot check, API-ledger checks, Ruff and mypy passed.

Ranking/forecast decomposition retains the preparation report on the carrier and every child request.
Public regression tests reproduced and verified the Python and Java propagation fix.
