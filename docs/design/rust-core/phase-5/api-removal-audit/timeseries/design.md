<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Build a forecasting query from arrays of history

**Proposal only: restore a Python/Java convenience builder.**
Forecasting already works; this missing helper prepares its tables and query from a common input layout.

## What is missing?

Illustrative input: customer `7` has `sales=[10,20]`, daily spacing and cutoff January 3.
The helper creates:

```text
customer row: 7, sales=[10,20]
             ↓ prepare history
entity: customer_id=7
target: (7, Jan 1, 10), (7, Jan 2, 20)
             ↓ ordinary schema + forecasting query
materialize → predict through the existing Relational path
```

The dates above assume a January 3 cutoff: the last synthetic observation is one interval before it.

| Question | Answer |
|---|---|
| Old support | `Graph.graph_and_pquery_from_timeseries()` accepted pandas arrays and returned a graph plus PQL string |
| Intended use | Avoid manually exploding histories, creating keys/relationships and writing the forecast query |
| Documentation | Full runnable-shaped example in its source docstring, plus `test_timeseries_graph.py`; no exact-helper example found in tracked Markdown/notebooks |
| New client today | Users can construct the two tables themselves and use existing `FORECAST` queries |
| Cost of leaving it out | More setup and opportunities for timestamp/key mistakes; forecasting capability itself is retained |

## Proposed API and boundary

Proposed Python `build_timeseries(frame, ...)` and Java `TimeSeries.build(table, options)` return `ForecastInput(schema, query, anchor_time)`.
Use the existing pandas/Arrow input conventions and ordinary query/materialization APIs.
Names are proposals, not an additional client or model type.

```text
Python/Java builder: input adapter + public workflow
                  ↓ Arrow histories + explicit options
Rust data preparation: validate lengths, expand rows, resolve time grid
                  ↓ entity/observation tables + effective interval
Python/Java: assemble schema/query using existing Rust admission/PQL
```

The public builder is a recipe in the language adapters.
A narrow Rust preparation function belongs alongside `sdm-data` Arrow transformations because row alignment and timestamp construction define engine inputs; sharing code alone is not the justification.
Reuse `sdm-schema` validation and `sdm-pql` parsing; add no sampler, query parser, serving operation or warehouse transformation.
Return Arrow buffers through existing ownership rules, without per-row callbacks.

## Contract and checks

- Retain static entity columns; generate IDs only when none are supplied, refusing name collisions and invalid supplied keys.
- Accept explicit timestamp arrays or positive fixed interval plus cutoff; when explicit timestamps omit it, infer the median adjacent interval from sorted timestamp copies, then require a positive supported interval.
- Validate lengths **per entity**, timestamps, timezone consistency, numeric observations, interval overflow and positive forecast count before returning anything.
- Do not silently remove duplicate timestamps or drop zero intervals from interval inference; repeated observations retain the existing target aggregation meaning.
- Preserve the old numerical target and `MAX`-over-window forecast recipe; do not silently replace it with sum, mean or interpolation.
- Define empty histories explicitly: retain their entities if other observations establish a valid schema; reject an entirely empty history set with a clear error.

The old loop concatenated values/timestamps without checking each pair's length; equal total lengths could hide incorrect alignment.
Fix that rather than preserving it.
Shared fixtures compare generated rows, timestamps, schema and query in both languages, including mismatches and irregular histories.
Run one resulting forecasting request against the model endpoint; push-down sampling is untouched.

## Verified code anchors

Old source at `12955d397`: engine `rfm/graph.py:2306–2510`, under `packages/structured-data-models-relational-engine/src/sdm_relational_engine/`.
Old tests: `packages/structured-data-models-relational-engine/tests/test_timeseries_graph.py`.
Current reuse points: `sdm-data`, `sdm-schema`, `sdm-pql`, and the existing Python/Java `Schema` and `PredictiveQuery` APIs.
