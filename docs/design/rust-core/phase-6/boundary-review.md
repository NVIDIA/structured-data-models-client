<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Independent review: first four Phase 6 boundaries

**Recommendation: treat Rust as the shared data/prediction engine, not a home for every common helper.**
The four capabilities should be available in Python and Java, but their presentation need not be implemented in Rust or be byte-identical.
Reviewed the four restoration designs against current code and the archived implementation at `origin/main`; no product changes or new tests were made.

## Decision rule

Put a rule in Rust when it determines data admitted to the model, sampling, query meaning, shared result interpretation, or engine-owned lifecycle/state.
Keep formatting, notebooks, files, downloads and ergonomic object wrappers in language libraries.
Share declarative assets and conformance fixtures where that is sufficient; duplicated short presentation code is not itself an architectural defect.
No new Python callback or round trip is needed for these features.

| Feature | Recommended boundary | Change to current proposal |
|---|---|---|
| Visualization | Python/Java format their existing schema descriptors and render/export through a common pinned Mermaid asset/template | Remove the new Rust Mermaid operation. Test equivalent nodes, keys and relationships plus escaping, not identical formatting bytes. Notebook display remains Python-specific. |
| Metadata and explanation views | Language wrappers over existing schema/result data | Keep prints, pandas views, Java records, summary/cohort/subgraph accessors outside Rust. No additional engine call merely to inspect a result. |
| Feature-importance aggregation | One pure shared result-semantic helper is reasonable in Rust | Its justification is interpreting model attribution consistently, not simply doing arithmetic. Define format/version, signed-score, missing/nonfinite and normalization rules; compute on demand, without copying all prediction rows or changing ordinary prediction results. |
| Sanitization report | Rust produces typed preparation facts; Python/Java expose read-only objects | Keep proposed ownership. Extend existing preparation output rather than parsing warnings, rescanning data or adding a report service. |
| RelBench loader | Language I/O/readers plus one shared dataset manifest and ordinary Rust schema admission | Keep proposed ownership. Do not add Rust networking/cache code or a dataset-specific engine subsystem. Store exclusions/overrides declaratively rather than copying Python behavior into Java. |

A shared Mermaid formatter would work technically, but adds an engine operation for presentation only and creates unnecessary binding/version coupling.
By contrast, attribution aggregation interprets a model-specific payload into a numerical result; it can reasonably live beside existing shared result policies in `sdm-request`.
Keep raw explanation data available unchanged, and preserve host-only access if no aggregation is requested.

## What sanitization reporting actually fixes

Example: preparation receives 100 customer rows, drops two with missing IDs and three duplicate IDs, then samples from the remaining 95.
The engine already performs and counts these removals.
Today the caller receives warning text and a prepared graph handle, but no durable structured report that can be displayed or checked programmatically.
The new feature exposes those existing facts, for example `input=100, output=95, null_key=2, duplicate_key=3, null_time=0`.
It does not introduce new cleaning or change predictions.

Concrete current code:

- `rust/crates/sdm-engine/src/frames.rs:119`: `Snapshot` owns `dropped: Vec<DroppedRows>`; each record has input rows and three drop counts.
- `frames.rs:427`: records are currently retained only when drops occurred, so clean-table entries need small accounting changes.
- `rust/crates/sdm-core/src/materialize/frames.rs:287`: counts become a `FrameRowsDropped` warning summary.
- `rust/crates/sdm-contract/src/frames.rs:61`: `PreparedFrameGraph` returns only `graph`, not the counts.

Return the report with graph preparation and carry it into requests/fitted handles.
Reports describe that prepared snapshot and stay readable after close; unavailable warehouse statistics must not be represented as zero or trigger extra warehouse scans.

## Other evidence and acceptance

Python `PredictionResult.explanation` already returns explanation data; Java `PredictionResult.java:230` reads the corresponding Arrow metadata.
The old `Explanation.feature_importance` sums cell scores by table/column, while its other conveniences largely select or print existing fields.
`sdm-request/src/result.rs` and `sdm-core/src/result.rs` already own shared result policies, but do not yet implement that attribution aggregation.
Java `Readers.java:147` already reads Parquet; the RelBench loader can use existing readers in both languages.

Use shared semantic fixtures for all four features and host-specific rendering/I/O tests.
Parity means the same useful capability and data meaning, not matching notebook APIs, whitespace, exception prose or mutable helper classes.
No sampler or warehouse push-down change is required for these four items.
