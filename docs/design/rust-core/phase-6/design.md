<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 6: restore missing user capabilities

**Integration branch: `rust-core-phase-6`, based on merged `rust-core` at `4b5df2e2e`.**
All restoration work merges into this branch before another merge into `rust-core`.
Scope is every missing or partial capability in the [API removal audit](../phase-5/api-removal-audit/audit.md), not just the first four features.
No release or merge into `main` is authorized by this plan.

## What Rust core is for

Rust is the shared data/prediction engine: schema interpretation, validation, PQL, sampling, temporal eligibility, request preparation and result semantics.
Its purpose is that the same supported data and request mean the same thing in every language, with reusable efficient execution.
It is not a repository for every utility that two languages happen to need.

| Put in Rust | Put in Python/Java adapters |
|---|---|
| Which rows are valid, sampled or labeled | Notebook/browser display and file output |
| Schema rules, query interpretation and typed transformations | Natural language-specific API objects and pandas/Arrow views |
| Counts produced during engine execution | Printing or displaying those counts |
| Defined numerical interpretation of model outputs | HTTP downloads, caching and dataset-file reading |

Small presentation helpers can exist in both languages with common fixtures and shared static assets.
A feature should not require a new Rust operation merely to avoid duplicating formatting code.
Downloads/readers feed existing core admission; they do not recreate sampling policy.

## First four designs

The existing documents remain at their stable paths and now track Phase 6 work.
Independent boundary review is complete and incorporated directly into each feature design; those documents are authoritative.
All four should provide useful APIs in both languages; Python notebook display/pandas views are intentional language-specific conveniences.

| Feature | Design | Reviewed boundary |
|---|---|---|
| Visualization | [Design](../phase-5/api-removal-audit/visualization/design.md) | Host formatting/rendering over current schema metadata; share renderer assets and fixtures |
| Metadata/explanation reports | [Design](../phase-5/api-removal-audit/reports/design.md) | Host accessors/presentation; distinguish raw-field access from numerical attribution semantics |
| Rows discarded during preparation | [Design](../phase-5/api-removal-audit/sanitization/design.md) | Engine produces factual counts, hosts expose/read/print them |
| RelBench loading | [Design](../phase-5/api-removal-audit/relbench/design.md) | Host I/O, one dataset manifest, existing schema admission |

“Sanitization” here means excluding unusable graph rows, not anonymizing data.
For example, preparing 100 input rows might keep 95 after excluding three missing keys and two duplicate keys.
The engine already computes drop counts for affected tables and emits a warning, but callers cannot retrieve a structured per-table report.
Expose those counts without changing which rows survive; capture clean-table counts in the same preparation pass and report unavailable when no measurement was made.

## Remaining scope and acceptance

The nine scoped restorations, including end-time, SQL columns and supplied relational examples, are complete.
See [final restoration evidence](final-restorations.md) and the earlier linked completion records.
The owner has approved the [connection and execution follow-up](remaining-api-mappings.md): rebinding, batch/session convenience, typed settings and optional explanation summaries.
These follow-ups are now reviewed, implemented and verified in Python/Java; see [completion evidence](followup-completion.md).
Do not silently expand restoration into unsupported old behavior: the old in-memory table API rejected generic expressions.

Use the existing compatibility manifest for Python/Java mappings and targeted tests, including capability-level coverage outside the old facade snapshot.
Review shared semantic/contract changes before implementation; review completed changes independently.
Prediction/warehouse tests are needed when inputs or sampling semantics change, not for every printing helper.
The first four must not alter warehouse sampling algorithms.

## Execution

See [implementation tracker](implementation.md) for isolated feature branches, contract review and merge status.
