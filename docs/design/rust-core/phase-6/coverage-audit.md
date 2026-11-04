<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Follow-up coverage audit of the first four restorations

This audit examines the maintained tests for visualization, reports, sanitization and RelBench after their initial integration.
It adds focused composition and failure checks without changing product code or public contracts.

| Area | Existing evidence inspected | Added coverage |
|---|---|---|
| Visualization | Shared hostile-name fixture, optional real browser, bundled resource identity, sandboxed notebook, explicit image HTTP format/timeout and redacted failures | Export an admitted RelBench schema to real HTML and Mermaid files in both hosts, preserving its declared relationship |
| Reports | Shared attribution values/errors, detached metadata, explanation ownership, lazy validation and local rendering | Shared accessor precedence/null/unknown-format/malformed-field cases; depth boundary and repeated-container ownership in both hosts |
| Sanitization | Exclusive drop reasons, clean tables, split requests, sessions, older snapshots and warehouse/Tabular status | Persist nonzero duplicate/time/null-key reasons; load a dataset and verify its exact table counts through materialization and request persistence |
| RelBench | Offline cache, declarations, pinned catalog, interrupted HTTP, ownership and real F1 smoke | Complete HTTP response with incorrect digest, unchanged invalid cache and temporary-file cleanup; independently loaded owners and downstream reporting/materialization |

The composed fixture has two customers and two orders, amounts 2.5 and 3.5, one declared customer link, no order primary key and one order time column.
Both hosts assert the same independently specified metadata and sanitization counts.
Python uses the public loader with the existing catalog fixture seam; Java uses the existing package-local catalog/base-URL seam before exercising public schema, report, export, query and persistence APIs.
No endpoint is required because this workflow prepares a request without predicting.
The tiny temporal fixture emits the expected context-before-data warning; it is not representative of model-quality validation.

The shared explanation fixture checks that explicit top-level nulls suppress nested prose, a top-level singular warning precedes nested plural warnings, and supported formats select nested cohorts/subgraphs while unknown formats retain top-level sequences.
Malformed requested fields raise the agreed structured error without hiding unrelated valid prose.
The depth check accepts 128 nested mappings and rejects 129; reused containers are copied without being mistaken for cycles.

## Scope and limits

The old Python RelBench loader at `12955d397` was inspected for declarations, exclusions and type overrides.
A focused run on the same local Parquet fixture checks alias resolution, amounts, exclusions, null primary-key/time declarations and the declared link against the retained behavior.
The existing local fetch/catalog seams avoid downloading a second dataset or introducing an old-client dependency into maintained tests.
This is a focused loader comparison, not complete old/new behavioral equivalence.
The old loader's ambiguity warning and normalization policy are not inferred to be part of the restored contracts.

These checks supplement the initial [completion evidence](completion.md).
They do not repeat its real browser, live F1 prediction, installed artifact or cross-platform qualification, and they do not establish catalog-wide dataset correctness or model accuracy.
Raw F1 text prediction remains outside the successful endpoint evidence; the earlier successful prediction used explicit categorical declarations.

## Validation

The affected Python report/loader suites and the extended dirty-sanitization persistence case passed: 22 tests, including four new test cases and one extended persistence case.
The matching Java report/loader suites and dirty-sanitization persistence case passed: 15 tests, including four new test cases and one extended persistence case.
Both hosts used isolated existing contract-59 native artifacts; no native rebuild or shared environment installation was needed.
Ruff lint/format checks and the whitespace check passed.
No product defect was found by these additions, so no product code, API snapshot or native contract changed.

The composed workflow is selected once under the RelBench API contract group.
The existing whole-file report selections include the new accessor and nesting tests, while the existing sanitization selections retain the extended persistence cases.
