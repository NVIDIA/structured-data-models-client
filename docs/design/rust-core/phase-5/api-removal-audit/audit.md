<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Deleted API audit

**The migration preserved the maintained facade, but did not preserve every useful feature of the archived engine.**
The earlier “no remaining acceptance gap” conclusion was too broad.
Restoration is tracked in [design.md](design.md).
Do not use the green prediction matrix as evidence that these missing utilities are covered.

## Phase 6 resolution

The tables below preserve the original audit findings, not the current implementation status.
The nine scoped restorations are complete; see [restoration evidence](../../phase-6/final-restorations.md).
The additional mappings now have these replacements in Python and Java; passing final combined validation is recorded in [completion evidence](../../phase-6/followup-completion.md).

| Former gap | Maintained replacement and deliberate difference |
|---|---|
| Connection rebinding | `Schema.with_source` / `withSource` returns an independent schema; Rust revalidates declarations and hosts retain connection ownership. |
| Retry/batch scoping | Explicit batch options, including supplied examples; one transport retry policy and Rust-managed temporary sessions for identical contexts. No mutable block-scoped settings or stacked retries. |
| Typed inference settings | Immutable `InferenceConfig` in both languages; Rust still supplies defaults and validates options. Old `RunMode` and engine imports stay retired. |
| Explanation helpers and generated prose | `ExplanationView` provides structured access; explicit `ExplanationSummarizer` optionally calls a configured text endpoint. No implicit second model call. |
| Task editing | `RelationalTask` plus supplied context/prediction tables; slice inputs before materialization instead of mutating a prepared task. |

Standalone Python parser/AST APIs and internal engine samplers remain retired; they are not restored by these user-facing replacements.

## What was compared

Candidate: `304da1e94`.
Baselines: `origin/main` at `12955d397`, plus the parent of retirement commit `af4c801a9` to catch interfaces removed during final cleanup.
All five Python package source trees were scanned for removed classes, functions, methods, constructors, container protocols, annotated and directly assigned instance fields, assignments, package initializer import aliases and literal `__all__` exports.
The maintained public snapshot was compared recursively, including signatures, properties and attributes, not just top-level names.
Java production sources were compared with the merged Phase 4 branch (`2b9e6ea88`).

| Surface | Finding |
|---|---|
| Supported Python facade: `sdm_client`, `sdm_api`, connectors | No removed snapshot entries, methods, attributes or changed existing signatures; version updates and additive connector APIs |
| Java since Phase 4 | No removed public methods; only the internal native loader changed, adding Windows support |
| Archived engine | Its imports and useful convenience APIs disappeared; capability gaps below |
| Archived Python PQL | Its parser/AST/validator/rewriter imports disappeared; `PredictiveQuery` remains, but is not a replacement for programmatic AST manipulation |
| Internal client interfaces | Backends, adapters, materializers and helper functions were removed or changed as policy moved into Rust; these were explicitly internal in the baseline stability policy |

The combined inventory has **2,051 removed declaration/field/value records**, not 2,051 user-facing APIs: 1,053 PQL, 868 engine and 130 client implementation records.
Of these, 290 are generated parser implementation records.
The main-only comparison has 1,985 records; the extra baseline catches cleanup-only interfaces too.
Exact names and source locations are in [removed-symbols.csv](removed-symbols.csv), [removed-exports.csv](removed-exports.csv) and [changed-symbols.csv](changed-symbols.csv).
These files are audit evidence, not new build tooling.
Five apparent `Request` field removals were excluded after tracing their delegated assignments in `_rust/direct.py`; those fields still exist.
The connector `read`/`read_table` imports also remain available through lazy exports and were excluded.

## Missing or reduced capabilities

“Missing” means no equivalent exposed operation was found in the maintained client; it does not mean the model itself lacks the ability.
Package-retirement approval is not approval to drop each capability.

| Old API | Current situation | Disposition |
|---|---|---|
| `Graph.visualize()`, `Graph.to_mermaid()` | No `Schema` replacement; bundled renderer and notebook/HTML/image export removed | Confirmed gap; restore visualization without restoring the engine |
| `Graph.metadata`, `print_metadata()`, `print_links()`; `Table.metadata`, `print_metadata()` | Tables, columns and relationships remain inspectable, but formatted reports are gone | Lost convenience; restore useful inspection/reporting |
| `Graph.from_relbench()` | No maintained loader; manually constructing frames is not equivalent to fetching and adapting a dataset | Confirmed missing loader |
| `Graph.graph_and_pquery_from_timeseries()` | No helper that expands array-valued history into entity/observation tables and creates the forecasting PQL | Confirmed missing convenience; forecasting itself still exists |
| `KumoRelational.get_train_table()` | No standalone labels-only API with the old size/anchor/iteration controls | Confirmed missing interface; sampled materialized context is not an equivalent replacement |
| `KumoRelational.sanitization_report`; `GraphSanitizationReport`, `TableSanitizationReport`, `SanitizationStatus` | Warnings/validation remain; no equivalent structured input/output/drop-count report | Confirmed diagnostics gap |
| `Table.end_time_column`, `has_end_time_column()` and constructor argument | Python/Java public schema builders write a null end-time column, despite an optional field in the Rust contract | Confirmed admission gap; effects on interval-aware queries need dedicated tests |
| `Graph.infer_metadata()`, `infer_links()`; `Table.infer_metadata()`, `infer_primary_key()`, `infer_time_column()`, `time_column_choice` | Initial inference exists in `Schema.from_data/from_source`; no equivalent explicit re-inference/reporting operation after edits | Partial replacement, not full parity |
| `Table.add_column(s)` with `ColumnSpec`/expression objects | No equivalent public ColumnSpec/expression editing API; semantic-source expressions remain supported internally | Missing SQL expression editing interface; old LocalTable rejected expressions and required DataFrame preprocessing |
| `Explanation.print()`, typed `summary/details/cohorts/subgraphs/feature_importance` accessors, `ExplainConfig` | `PredictionResult.explanation` retains explanation data as a dictionary; old typed/printing helpers are gone | Partial replacement; do not describe explanations as entirely removed |
| `TaskTable` editing/narrowing and `predict_task()` | Materialize/predict survives through `PredictiveQuery`, `Request`, `Split`, `SDMClient`; arbitrary old task-object editing has no exact facade equivalent | Partial replacement; decide which editing use cases to retain |

## Removed names whose main workflow survives

These mappings are confirmed in code; they are not claims of exact signature, error or mutation equivalence.

| Removed family | Current path / remaining distinction |
|---|---|
| `Graph.from_data()` | `Schema.from_data()` |
| `Graph.from_sqlite/duckdb/postgres/databricks/snowflake()` | Connector source plus `Schema.from_source()` |
| `Graph.from_databricks_metric_view/from_snowflake_semantic_view()` | `source.semantic_model(...)` plus `Schema.from_source()` |
| `Graph.table/has_table/tables`, table column lookup and key/time access | `schema[name]`, membership, `.tables`, `table[name]`, `.columns`, `.primary_key`, `.time_column` |
| `Graph.add_table/remove_table/link/unlink/edges` | Schema table mapping and `Relationship` list; direct mutation is not automatically equivalent to all old validation/cleanup behavior |
| `Graph.validate()` | `Schema.check_consistent()` and admission validation; old backend-specific checks must not be presumed identical |
| `Graph.conversion_messages/conversion_notes` | `Schema.conversion_notes`; no old exception/report class compatibility implied |
| `Graph.inferred_edges` | Relationships survive, but no equivalent distinct inference-provenance view was found |
| `Graph/Table.backend`, source metadata access | Source/table descriptors remain; old backend objects and raw metadata introspection are not preserved as APIs |
| `Graph.update_connection()`, model counterpart | Use connector/source construction and a new schema; no dedicated live-handle rebinding API |
| Engine `KumoRelational.predict/materialize_task/predict_task` | `PredictiveQuery.materialize()` and `SDMClient.predict()`; old import/class signatures intentionally not retained |
| `KumoRelational.add_lagged_target()` | `lag_timesteps` materialization option exists and has retained payload tests; mutating an existing `TaskTable` is not supported |
| `KumoRelational.retry/batch_mode` | `SDMClient(max_retries=...)` and materialization batching options; context-manager scoping and retry semantics are not exact replacements |
| Engine initialization/global state, `NimClient`, transport and exception types | Explicit `SDMClient` lifecycle, transports and structured client errors; old global state and exception identities are retired |
| `MaterializedPredictionRequest` payload/batch fields | Materialized `Request` and `Split` expose the new workflow; old per-batch object access is not a retained contract |
| Engine `RunMode`, inference configuration classes and dtype/stype enums | Materialization options and maintained wire vocabulary preserve the prediction configuration path; old typed imports are retired |
| Python PQL `validate`, parsers, profiles, AST and rewrite utilities | Shared Rust PQL implementation behind `PredictiveQuery`; standalone Python parser/AST/rewriter API is retired |

All removed engine/PQL names, including backend-specific table/sampler/expression helpers, metadata inference functions, enums, aliases and constants found by the scan, are listed in the inventory.
Do not restore internal samplers or Python query planners merely to recover convenient user features.

## Why the checks missed it

The baseline `docs/reference/stability.md` explicitly limited supported APIs to the facade exports and described direct engine imports as “at your own risk”.
The old README nevertheless advertised `Graph.visualize()`, and the engine contained real documented utilities.
Our maintained snapshot and matrix followed the narrow stability boundary; the archive review checked runtime independence, not a complete user-capability migration map.
Approving archive removal was incorrectly treated as sufficient evidence that no useful functionality remained to port.
That was my audit/process mistake.

## What must change before merge

1. Restore visualization and agree an explicit retain/replace/retire decision for the missing and partial capabilities above.
2. Add those capability decisions to the existing parity manifest, with concrete Python/Java expectations; a renamed API needs a replacement test, not just a new snapshot.
3. Test the restored public workflows and add an upgrade guide for intentionally removed import paths and conveniences.

A local import probe against this checkout confirmed that the eight listed `Schema` visualization/loading/inference/printing helpers are absent, `Table` has no end-time constructor argument, and both lazy connector readers remain callable.
Accessing `Schema.visualize` raises `AttributeError`.
No product code was changed by this audit and no new live-model tests were run.
This is a source/API deletion audit, not a proof of all behavioral compatibility: dynamic attributes, downstream callers and arbitrary runtime semantics cannot be certified by declaration comparison.
The incomplete equivalences above are explicitly open, rather than marked parity.
