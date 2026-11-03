<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 2. Schema reports and explanation conveniences

**Status: implemented in Python and Java, independently reviewed, with shared metadata and attribution fixtures.**
Expose readable information already held by schemas/results, without another warehouse or model call.
Independent boundary review is incorporated below; this document is the implementation contract.

```text
Schema fields / prediction explanation
        ├─ records and text → Python / Java reporting helpers
        └─ numeric attribution aggregation → Rust → report records
```

| Layer | Responsibility |
|---|---|
| Python / Java | Metadata/link records, explanation accessors, text printing and local display |
| Rust result-processing code | Shared feature-importance aggregation and numeric/error rules; reuse the existing result path |
| Existing schema/result descriptors | Source of truth; no inference or schema mutation during reporting |

## Acceptance

Compare metadata records and attribution values across languages using the same fixtures.
Cover absent explanations, nested cohorts/subgraphs, empty attribution, repeated columns and malformed numeric values.
Test host text/pandas views separately; verify reporting does not mutate inputs or perform I/O.
No GPU tests are needed for wrappers over captured result payloads.

## Implementation and review gate

First finalize the public signatures, result shapes, ownership and error behavior here; obtain coordinator contract review before implementation.
Add focused public-API and cross-language tests first where practical, then implement the reviewed behavior.
Use existing parity/snapshot tooling; do not hand-edit generated snapshots or introduce migration-only machinery.
Finish with independent code review, address findings, and run affected tests before merging into `rust-core-phase-6`.

If shared attribution computation is needed, place its pure numerical policy alongside `sdm-request` result policies and expose it through the existing boundary; leave raw explanation data unchanged.

## Approved concrete API

`Schema.metadata_report()` / `metadataReport()` returns `MetadataReport(tables)`.
`Table` and Python `SourceTable` provide the same method returning one `TableMetadata(name, columns, primary_key, time_column)`.
Columns are `ColumnMetadata(name, dtype, stype)` records.
`Schema.link_report()` / `linkReport()` returns `LinkReport(links)` of `LinkMetadata(source_table, source_column, target_table, target_column)`.
All records are detached immutable snapshots, preserving descriptor order, with nullable time columns and immutable key sequences.
Reports provide `to_text()` / `toText()`; Python also provides `to_pandas()`, `print()`, schema/table `print_metadata()` and schema `print_links()`.
Metadata frames have fixed columns `table,column,dtype,stype,primary_key,time_column`, with boolean key/time flags; link frames have the four relationship fields.
Empty frames retain columns, and metadata text names zero-column tables.

`PredictionResult.explanation_view` / `explanationView()` returns nullable `ExplanationView`, leaving the existing raw accessor and result type unchanged.
The view snapshots only the explanation into immutable JSON mappings/sequences, preserving unknown fields and nulls.
`raw` is the full payload; `details` is the nullable nested mapping.
`summary` is nullable text; `warnings` is a nullable string sequence accepting `warnings` or singular `warning`, with top-level fields taking precedence over nested details.
`cohorts` / `subgraphs` are nullable sequences from nested details for `format=kumo_rfm_v2_1`, otherwise from the top level.
Missing/null fields return null; malformed requested fields raise `INVALID_REQUEST` with a field path without preventing access to unrelated valid fields.
Cycles, non-JSON objects, non-string mapping keys and nesting beyond 128 levels are rejected by the new view, without changing the raw accessor.
`to_text()` / `toText()` and Python `print()` render existing summary/warnings without prediction-row reads, network or model calls.

`feature_importance(normalize=False)` / `featureImportance(boolean normalize)` returns nullable `FeatureImportance(rows, normalized)` with immutable `FeatureImportanceRow(table,column,score)` records.
Only `format=kumo_rfm_v2_1` and `details.subgraphs[].tables[table][node].cells[column].score` are interpreted numerically.
Unsupported formats or missing/null subgraphs return null; an explicit empty list returns an empty report.
Hosts validate structural containers and omit missing/null scores; Rust aggregates finite signed numbers by table/column and rejects booleans, strings, nonfinite values and arithmetic overflow with `INVALID_REQUEST` and a field path.
The boundary carries only table, column, path and JSON scalar score text, preserving huge integers and nonfinite spellings for shared numeric validation.
Rows sort lexicographically by table then column.
Normalization divides aggregated scores by the sum of their absolute values: scores `-2, 6` become `-0.25, 0.75`; a zero denominator leaves zeros unchanged.
Only requested feature importance invokes the pure `sdm-request` numerical helper, using the existing boundary with an attribution-only descriptor and no prediction rows or cell values.
Shared fixtures cover values and errors in both hosts; rendering remains host-specific.
