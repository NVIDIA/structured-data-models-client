<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Infer missing metadata and safely edit a schema

**Approved for contract-first implementation: restore in Python and Java.**
This helps users repair or extend a schema without rebuilding it and losing their declarations.

## What is missing?

Illustrative workflow: load `customers` and `orders`, declare `customers.customer_id` as the key, then ask the client to discover the matching `orders.customer_id` relationship.
Later, remove `orders` and its relationships together.

| Question | Answer |
|---|---|
| Old support | `Graph.infer_metadata()`, `infer_links()`, table key/time inference, table removal and link helpers |
| Intended use | Fill missing key/time choices, discover links after edits, inspect which links were inferred, safely remove tables |
| Documentation | Source docstrings, including a graph construction/inference example; inference and inferred-edge tests. No exact-API examples found in tracked Markdown/notebooks |
| New client today | Initial inference in `Schema.from_data/from_source`; explicit key/time/type edits and relationship replacement already exist |
| Cost of leaving it out | Users rebuild schemas or manually manage relationship lists; no explicit rerun or reliable inferred-link provenance |

## Proposed API and boundary

Use `schema.infer_metadata()` / `inferMetadata()` and `infer_links()` / `inferLinks()` as explicit operations returning a new schema in both languages.
Provide safe table removal and link add/remove helpers through the same schema-edit path; retain current lower-level setters.
Expose recorded link origin separately from the existing relationship shape, without guessing historical provenance.

```text
Python/Java: requested edit + current schema
                  ↓
Rust sdm-schema: infer missing choices / validate atomic edit
                  ↓
Python/Java: new schema + readable inference information
```

Rust owns inference, declarations, link provenance and cleanup because they determine graph meaning and later sampling.
Hosts supply existing Arrow/provider inputs and format results; they do not independently infer relationships.
Extend `SchemaEdit` and existing inference orchestration, not a second inference engine.

## Contract and checks

- Fill missing key/time metadata; do not silently re-infer every semantic type or overwrite declarations.
- Preserve explicitly disabled key/time inference; distinguish “not chosen yet” from “user chose none”.
- Add only unambiguous missing links; repeated inference is idempotent, and existing user links win.
- Record inferred versus declared links at the engine decision; old saved schemas have unknown origin.
- Remove a table and incident links atomically; reevaluate inferred time choices whose meaning depends on links.
- Keep prepared requests/sessions unchanged; rebuilding a schema does not mutate an existing native snapshot.

Shared Python/Java fixtures cover declarations, ambiguity, repeated inference, table deletion and provenance persistence.
Warehouse inference may read metadata/previews through existing providers; sampling SQL and algorithms remain unchanged.
Check one materialization after an edit in memory and via a warehouse provider.

## Test standard: parity or an explained improvement

Use existing public-API tests and small shared fixtures, not a new migration comparison framework.
Be thorough about behaviors, edge cases and connector differences; use focused old-client comparisons where they resolve a real semantic question.
Before implementation, settle signatures, ownership and error contracts with coordinator review; finish with independent code review and affected tests.

Run small public workflows through the old Python client, new Python and Java using identical input data.
Compare keys, time choices, relationships and resulting materialized rows; compare semantics rather than printed text or object classes.
Independently assert declared choices survive, removing a table leaves no dangling links, repeated inference is stable and failed edits are atomic.
Cover both in-memory data and provider metadata/previews; compare existing warehouse prediction queries before/after to detect unintended sampling changes.
When old output violates an invariant, record the small reproduction, corrected expected output and intentional difference in the maintained test fixture.
Add one live prediction after inferred/edited schema preparation; it verifies integration, not whether the model predicts the exact same floating-point value.

## Verified code anchors

Old source at `12955d397`: engine `rfm/graph.py:2579,2647,2996,3188` and `rfm/base/table.py:970` under `packages/structured-data-models-relational-engine/src/sdm_relational_engine/`.
Old tests: engine `tests/test_inferred_edges.py` and `tests/metadata_inference/test_time_col.py`.
Current implementation seams: `rust/crates/sdm-schema/src/{infer,assemble,edit}.rs`, Python `schema.py`, Java `Schema.java`.
