<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tell the engine when an entity stops being active

**Implemented for review: Python and Java end-time declarations, Rust expiry handling and protection against future-data leakage.**
Local and live validation passed; coordinator review and branch integration remain.

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

## API and ownership

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

## Implementation checkpoint

Contract 63 adds optional `end_time_column` to held/described tables and an end-role schema edit; old records default to no end role.
Python uses the existing table property/constructor pattern; Java uses `withEndTimeColumn` and `withoutEndTimeColumn`.
Rust validates distinct timestamp roles, selects valid examples before random draws and masks copied features after sampling.
Providers use existing projections and reads; warehouse sampling SQL is unchanged.

| Check | Result |
|---|---|
| Public Python: memory, SQLite, DuckDB, metadata, persistence, invalid edits, timezones, fit preparation, forecast | 16 passed |
| Public Java: same shared fixture, memory, SQLite, DuckDB | 3 passed |
| Rust schema and engine suites | Passed |
| No-end exact ordering/random-state and malformed validity lengths | Passed |
| Explicit-ID forecast rendering | Passed |
| Real warehouse and endpoint | Databricks SQL and Snowflake: eligibility, masking and real relational prediction passed; existing fixtures unchanged |
| Adjacent regressions | 60 Python schema/labels tests and 107 forecast/decomposition tests passed |
| Static checks | Affected Rust crates clippy, Python lint/format/types, contract lock and API ledger passed |

Forecast parts change target windows, not the prediction anchor: a future closing date stays hidden in every part.
A separate request at a later anchor may expose that date once it is known.
Explicit forecast IDs bypass initial eligibility, as in the old client; automatic forecast selection checks eligibility at its first anchor.
A reproduced existing Rust bug dropped explicit `FOR` IDs when rendering forecast parts; the fix reuses the existing entity-clause renderer, preserving IDs and filters.
The no-end invariant applies to unaffected sampling, not that demonstrated forecast bug.

The pinned old client independently returned eligible IDs 3, 4 and 5 for the shared account fixture.
Its sampler also confirmed the repeated account's end is hidden on January 4 and visible on January 5, while an explicitly scored expired account retains its known end.
An unrelated clock table supplies the old client's required global time bounds; account rows are unchanged.
Live checks are maintained in `test_end_time_warehouse.py` and opt in through `SDM_END_TIME_CONFIG`; credentials are not stored in the repository.
