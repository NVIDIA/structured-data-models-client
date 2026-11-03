<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tell the engine when an entity stops being active

**Proposal: restore end-time declarations in Python and Java, including expiry handling and protection against future-data leakage.**
This is design only; implementation has not started.

## What users lose today

A closed account should stop being chosen as an active example after its closing date.
When predicting earlier in history, its future closing date must not be shown to the model.
Today the public schema cannot declare that role, and the engine does not implement those two behaviors.
An optional field in the query contract is not working end-time support.

Illustrative example, predicting as of January 4:

| Account closes | Eligible as a sampled example? | Closing date visible to the model? |
|---|---|---|
| January 3 or January 4 | No | If explicitly requested for prediction, its already-known date may remain visible |
| January 5 | Yes, if already created | No: replace the future date with null |
| Unknown/null | Yes, if already created | Null |

The old client supported this through `Table.end_time_column` and constructor arguments across memory and SQL tables.
It documented the role in source docstrings and tested it, but no exact usage example was found in tracked Markdown or notebooks.
Its test explicitly predicts an expired entity while excluding expired entities from sampled context: explicit IDs must not silently disappear.

## Proposed API and ownership

Python follows the existing `time_column` property pattern: `schema['accounts'].end_time_column = 'closed_at'`.
Java adds the corresponding immutable schema edit, `withEndTimeColumn(table, column)`.
Both expose the role in metadata, persist it and allow clearing it.

```text
Python / Java: declare closed_at as the end-time column
                          ↓
Rust schema: validate the role and preserve it through edits/save/load
                          ↓
Rust engine: choose eligible examples at each anchor time
                          ↓
Rust request assembly: hide end dates later than that anchor
                          ↓
Existing model request
```

This belongs in Rust because it changes which data a prediction may use.
Hosts must not duplicate eligibility or masking rules.
Providers project the extra column through existing reads; the old client filtered fetched candidates in its shared sampler, so no warehouse sampling SQL rewrite is inherently required.

## Contract and validation

- Sampled eligibility: `start <= anchor` when a start exists, and `end > anchor OR end IS NULL` when an end exists.
- Preserve explicit prediction-ID alignment; treat it separately from sampled context/label eligibility.
- Hide future end values in every model-input copy, using that copy's anchor, without mutating source data.
- Apply the same meaning to materialization, labels and fitted-session preparation; do not invent interval joins or change target-window aggregation.
- Preserve the old entity-time-anchor special case initially: it does not add an expiry filter; document and test it explicitly.
- The end role must differ from the primary key and start-time column; null end values are valid and must not trigger start-time null-row removal.
- Schemas without an end role retain their current behavior; validate timestamp roles through existing schema rules and make invalid edits atomic.

Use shared Python/Java fixtures for equality boundaries, null/end-only columns, timezones, multiple anchors, explicit expired IDs, masking, persistence and invalid declarations.
Run memory and local SQL paths plus a real warehouse journey, inspecting selected IDs and prepared values.
One endpoint prediction confirms request compatibility; exact fixture assertions establish the semantics.

## Verified evidence

Old source is pinned at `12955d397`; paths below are under the retired engine package.
`rfm/base/table.py:647` defines the role; `rfm/base/sampler.py:322` masks future dates and `:705` handles eligibility.
`tests/test_rfm.py:1026` demonstrates expired explicit IDs and masked context values.
Current seams: `sdm-contract/src/ops.rs` (`QueryTable` versus `DescribedTable`), `sdm-engine/src/graph.rs`, `sample.rs`, and `sdm-core/src/materialize.rs`.
