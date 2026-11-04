<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Build a forecasting query from arrays of history

**Approved for contract-first implementation: restore a Python/Java convenience builder.**
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

Python `build_timeseries(frame, timeseries_col, *, timestamps_col=None, time_delta=None, anchor_time=None, entity_col=None, num_timeframes=1)` accepts pandas or Arrow and returns frozen `ForecastInput(schema, query, anchor_time)`.
Java `TimeSeries.build(Arrow Table, TimeSeriesOptions)` returns `ForecastInput(schema, query, anchorTime)`; options use `Duration`, `Instant` and equivalent column names.
Returned schema data is independently owned; inputs are never consumed, and temporary Java input allocators may close after building.
Naive timestamps mean UTC, aware timestamps preserve instants, and output timestamps/cutoff are UTC nanoseconds; mixed naive/aware pandas histories are refused.
Observations are Float64 model values; large integers and decimals may round, while entity keys retain their original types.
Null observation values are retained, NaN becomes null, infinity and nonnumeric/bool observations are refused; null history lists and null timestamps are refused.
Individual empty histories retain entities; no observations across all entities is an error.
Intervals must be exact positive whole minutes, forecast count a positive integer, and timestamp/horizon arithmetic must fit signed nanoseconds.
Entity IDs must be nonnull and unique; generated `entity_id` must not collide, and the entity key cannot be `value`, `timestamp` or either history column.
Invalid options/data return `INVALID_REQUEST`; existing schema-admission errors remain unchanged.
Rust emits the existing warning channel when inferred intervals are irregular.

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

## Test standard: parity or an explained improvement

Use existing public-API tests and small shared fixtures, not a new migration comparison framework.
Be thorough about behaviors, edge cases and connector differences; use focused old-client comparisons where they resolve a real semantic question.
Before implementation, settle signatures, ownership and error contracts with coordinator review; finish with independent code review and affected tests.

Feed the old builder, new Python and Java the same histories; compare generated rows, keys, times, effective interval and parsed query meaning.
Use hand-calculated tiny examples as an independent oracle, so agreement between implementations is not the only check.
Include per-entity length mismatches whose total lengths happen to match: old concatenation can misalign them; new clients must reject them before admission.
Cover explicit/synthetic times, zero/duplicate intervals, irregular histories, nulls, empty histories, ID collisions, timezone differences and overflow.
Assert input buffers are unchanged and returned Java data survives temporary input closure.
Run one resulting forecast against the endpoint; ordinary transformation/error fixtures stay fast and local in CI.
Record each deliberate old/new difference with its reason and regression test; do not inherit old accidental behavior just to make comparison green.

## Verified code anchors

Old source at `12955d397`: engine `rfm/graph.py:2306–2510`, under `packages/structured-data-models-relational-engine/src/sdm_relational_engine/`.
Old tests: `packages/structured-data-models-relational-engine/tests/test_timeseries_graph.py`.
Current reuse points: `sdm-data`, `sdm-schema`, `sdm-pql`, and the existing Python/Java `Schema` and `PredictiveQuery` APIs.

## Implemented example and verification

```python
import pandas as pd
from sdm_client import build_timeseries, KumoRelational

forecast = build_timeseries(
    frame, "sales", entity_col="id",
    time_delta=pd.Timedelta("1D"), anchor_time=pd.Timestamp("2024-02-01"),
    num_timeframes=2,
)
request = forecast.query.materialize(
    model=KumoRelational(), indices=[1, 2], anchor_time=forecast.anchor_time,
)
result = client.predict(request, model=KumoRelational())
```

Java uses `TimeSeries.build(table, TimeSeriesOptions.builder("sales").entityColumn("id").timeDelta(Duration.ofDays(1)).anchorTime(cutoff).numTimeframes(2).build())`, then the existing schema/query materialization API.
Shared Python/Java fixtures check exact rows, times, nulls and query meaning, plus independent output ownership and invalid-input cleanup.
The old public helper was reproduced accepting unequal per-entity lengths when their totals match and silently assigning the second entity's timestamp to the first entity's value; the new clients reject that input.
Whole-minute precision, UTC-naive interpretation and strict key/null validation are deliberate improvements, not compatibility accidents.
A live public builder → materialize → prediction journey returned four forecast rows for two entities and two windows using the existing Relational endpoint.
The Java adapter works around Arrow's unsupported timezone-list copier using public readers, preserving original timestamp units and instants before Rust normalization.
