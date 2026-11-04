<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First four API restorations: complete

All four are implemented in Python and Java and merged into `rust-core-phase-6`.
Nothing was published or merged into `rust-core` or `main`.

| Feature | Delivered | Boundary |
|---|---|---|
| Visualization | Mermaid, standalone HTML, file export; Python notebook display | Host formatting with one bundled renderer |
| Reports | Metadata/links, immutable explanation views, feature importance | Host presentation; lazy numeric aggregation in Rust |
| Sanitization | Per-table row exclusions on requests and fitted handles, including split requests and persistence | Rust preparation counts; immutable host records |
| RelBench | Shared pinned datasets, verified cache and offline loading | Host I/O through existing schema admission |

## Evidence

- Final source integration: `bd91bb722`; feature contracts and progress are linked from the [tracker](implementation.md).
- Installed Linux x86-64 Python wheel: **157 passed**, four optional skips (two browser checks and two Snowflake-dependent checks).
- Final Java integration: **33 passed**, including report access, split-request propagation, dataset loading and rendering resources.
- Affected Rust suites: **321** engine/contract tests and **256** request/contract tests passed; these overlap and are not a unique total.
- Real browser checks rendered both languages' hostile-name diagrams without script execution or network requests.
- Real pinned RelBench F1 loading matched across languages: nine tables, 97,606 rows, 13 relationships.
  The endpoint rejected raw text columns; explicit public categorical declarations then produced a successful two-row regression prediction.
  Raw full-dataset prediction is not claimed as passed.
- Wheel/JAR resource bytes match the shared source assets and catalogs.
  The installed wheel loads production contract 59; a standalone Java process loads the native library from the JAR and computes attribution.
- API snapshots/ledger, lint, formatting, type checks, repository hygiene and affected Rust clippy checks passed.

The local wheel/JAR use development native builds for integration verification, not publication artifacts or a new multi-platform release qualification.
The wheel was installed separately while reusing the existing dependency environment.

## Review decisions

Independent contract and implementation reviews are complete.
Fixes included Java Arrow dictionary ownership, redacted image-export errors, reports surviving ranking/forecast decomposition, and structured oversized-number errors.
No review findings remain open.
Warehouse sampling and ordinary prediction execution are unchanged.
Feature-importance normalization uses signed totals divided by their absolute mass; unsupported explanation formats return no interpretation.

This completes the four-feature goal, not every remaining Phase 6 restoration.
Inference/edits, the time-series builder, labels-only operations, end-time eligibility and scoped expression/task editing remain in the [broader audit](../phase-5/api-removal-audit/design.md).

## What the first four tests prove

The checks above test restored workflows and explicit invariants, not a complete side-by-side run of every old API.
Old source/tests informed the contracts; identical old/new behavior across every input is not established.

| Feature | Verified benefit / retained behavior | Limit of evidence |
|---|---|---|
| Visualization | Both languages; hostile labels stay inert, offline output and export-error redaction | Real browser checked; notebook/image HTTP behavior largely mocked, not every deployed notebook/service |
| Reports | Detached immutable views; signed absolute-mass normalization, zero/nonfinite/overflow handling; no eager prediction work | Numeric policy intentionally differs from old signed-total normalization; no performance benchmark claim |
| Sanitization | Structured counts for clean and dropped tables, preserved across sessions, persistence and split requests | Extends observability without changing filtering; warehouse reports explicitly unavailable |
| RelBench | Python/Java matching rows and metadata, pinned catalog retained, cache integrity/recovery and ownership tested | One real dataset, not the whole catalog; raw endpoint text limitation remains |

Java support is added capability relative to the old Python-only engine.
Security/validation improvements are backed by focused tests; they are not a claim that every old release was vulnerable or that the new client is faster.
