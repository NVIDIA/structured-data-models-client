<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Restore missing user capabilities

**Status: proposed; implementation has not started.**
Restore the useful capabilities identified in the [deleted API audit](audit.md), through the maintained Python and Java clients.
Keep the shared Rust architecture; recovering a feature does not require recovering its old import path or Python engine.
The earlier green CI proves its tested scope, not completion of this work.

## Where the work belongs

```text
Python / Java public API
  ├─ Display, files, downloads, cache → language adapter
  └─ Schema edits, data transformations, labels → shared Rust
                                                   │
                                      existing Arrow/provider interfaces
                                                   │
                                         in-memory data or warehouse
```

Rust owns rules that must agree across languages: inference, validation, transformations, eligibility and result aggregation.
Python/Java own presentation and I/O, with thin wrappers around shared operations.
Use existing crates and message contracts; no new callback cycle or parallel Python sampling implementation.
Python returns pandas-friendly views; Java uses the existing Arrow conventions.

## First four boundary designs

All four provide useful functionality in **both Python and Java**; notebook display and pandas views are Python-specific adapters.

1. [Schema visualization](visualization/design.md)
2. [Metadata and explanation reports](reports/design.md)
3. [Sanitization reports](sanitization/design.md)
4. [RelBench loading](relbench/design.md)

These are proposals for review, not implemented APIs.

## Implementation order and tracker

All rows are **not started**; effort is relative, not an ETA.
Names below describe proposed API placement, not frozen signatures.

| Order | Capability and proposed API | Ownership / existing implementation | Size |
|---|---|---|---|
| 1 | `Schema.to_mermaid()` / Java equivalent; `visualize()` and HTML/file output | Small shared schema formatter in `sdm-schema`; hosts handle rendering, notebooks and files | Small |
| 1 | Schema/table metadata and link reports; explanation accessors/printing on `PredictionResult` | Hosts format existing descriptors/results; shared numeric feature-importance aggregation belongs in Rust | Small–moderate |
| 1 | Structured sanitization report on prepared requests/sessions | Expose existing `sdm-engine` snapshot drop counts through preparation results; hosts provide read-only views | Small |
| 1 | RelBench loading into `Schema` | Host download/cache/parquet I/O; common pinned dataset metadata; existing schema admission | Small–moderate |
| 2 | Explicit metadata/link inference and safe table/link edit helpers | Reuse `sdm-schema` inference and `SchemaEdit`; add provenance for inferred links and atomic cleanup when removing tables | Moderate |
| 2 | Time-series builder returning schema plus predictive query | Rust expands Arrow observation arrays and constructs tables/query; hosts adapt inputs/results | Moderate |
| 2 | Labels-only operation on `PredictiveQuery`, replacing `get_train_table()` | Production Rust labeling operation using existing eligibility/aggregation/provider machinery; return entity, anchor and target columns | Moderate |
| 3 | End-time-column configuration | Carry through both facades, shared schema, engine graph, provider projections and eligibility | Substantive |
| 3 | Generic expression-column and task editing | Define useful edits first; Rust validates expressions, dependencies, entity references and row alignment; providers execute source expressions | Substantive; scope review required |

**Sampling impact:** presentation, reports and loaders need no sampler changes.
Inference/edit conveniences reuse existing algorithms; warehouse metadata previews may be reread.
Labels-only requires a distinct labeling mode, without neighbor gathering or model inference.
End-time support changes entity eligibility, including warehouse selection.
Generic expression editing changes source projection/planning; these two items need focused push-down regression coverage.

## Contracts to settle before coding each affected row

- **Presentation:** preserve offline HTML and Mermaid source; explicitly distinguish network-backed image export, keep rendering safe, and restore bundled-asset notices/security checks.
  Notebook display is language-specific; shared diagram content and file export should be available in both languages.
- **Reports:** retain zero-drop tables and exclusive drop-reason counts; report “not available” for warehouses where counts were not collected instead of adding scans.
- **Inference/edits:** preserve user declarations; old `infer_metadata()` filled missing key/time metadata rather than recomputing every type.
  Removing a table must also clean up its links; inferred-link provenance cannot be guessed by a host.
- **Labels:** zero training rows, at most `size` output labels, bounded trials, correct default anchor/forecast horizon and explicit empty-result behavior.
  The existing test-only labels operation is not equivalent and must not simply be exposed.
- **End time:** eligibility includes `end > anchor OR end IS NULL`; the existing optional contract field alone does not implement this behavior.
- **Expressions/tasks:** choose explicit operations such as adding a validated derived column or selecting aligned rows; do not resurrect arbitrary mutable engine task objects.
- **Loaders/transforms:** preserve RelBench key/link/time metadata and dataset overrides; define empty/mismatched histories, timestamps, timezones and dtype handling for the time-series builder.

## Completion checks

Add each capability and its Python/Java mapping to the existing `tests/parity/manifest.json` before marking it complete.
Reproduce the old useful workflow, then test the replacement through public APIs; retain improvements instead of copying old bugs.
Use shared fixtures for cross-language schema, labels, reports and transformations, plus host-specific rendering/I/O tests.
Run endpoint tests only where a restored feature changes prediction inputs or behavior; presentation and labels-only checks do not require a GPU.
Check end-time equality/null boundaries and expression projections in both in-memory and warehouse paths.
Obtain independent contract/design review before substantive changes, then implementation review and affected tests.

Document replacement usage and explicitly resolve the audit's partial mappings: task editing, connection rebinding, retry/batch scoping and old typed configuration/explanation imports.
Lagged prediction features already exist through `lag_timesteps`; test/document that route rather than creating another computation path.
Standalone Python PQL/engine internals remain retired; package-retirement approval must not be reused as blanket approval to omit user capabilities.
Merge readiness remains open until every missing/partial capability has implementation evidence or an explicit accepted retirement decision.

## Clarification from the old implementation

Generic expression columns were a SQL/push-down feature: `SQLSampler` projected the expression as a named column in its queries, without creating a stored warehouse column.
The old `LocalTable` explicitly rejected expression evaluation and told users to calculate columns on their pandas DataFrame first.
Adding an in-memory expression language would therefore be new functionality, not a parity restoration requirement.

Task narrowing was used internally to cap context rows and split prediction batches (`rfm.py`), with direct tests in `test_task_table.py`.
Tests also constructed `TaskTable` objects with explicit labels/entity IDs/timestamps and passed them to `predict_task()`.
Source docstrings document these APIs; the search of tracked Markdown/notebooks found no user tutorial for general task editing.
Restore demonstrated workflows before committing to a general task editor; existing automatic batching does not itself require exposing task mutation.
