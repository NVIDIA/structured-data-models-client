<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Supplied relational examples

`examples.json` is shared by Python and Java tests.
Labels are deliberately supplied by the caller, not calculated from graph features.
Expected IDs, labels, order and end-time visibility are independently specified.

| Case | What it must prove |
|---|---|
| Regression, binary, multiclass | Preserve labels, order, repeated entities and context/prediction overlaps |
| Temporal links | Preserve ordered lists of destination IDs, including empty lists |
| Forecast | Scalar numeric context labels; nanosecond step; original prediction anchor plus steps 1, 2, 3 |
| Static overlap | Missing anchors use the graph's latest start timestamp |
| Timeless static overlap | Missing anchors use the existing timeless sentinel |

For `schema_variant: timeless`, use `timeless_tables` without relationships or time/end roles.
For the ordinary graph, account 3 closes on January 5: its copied end is hidden on January 4 and visible on January 5.
Supplied examples remain present even when an entity is expired or overlaps a prediction row.

## Verified old and live behavior

Executed the retired client pinned at `12955d397` using `TaskTable`, `_get_context` and `predict_request_to_json` for regression, binary, multiclass, temporal-link, forecast and static-overlap cases.
It preserves every supplied context row, including identical entity/anchor overlaps with prediction rows.
Multiclass classes retain first-appearance order; link labels travel as string lists while preserving their order.
Forecast context labels are scalars, not vectors; the wire carries `step_size` in nanoseconds and `num_forecasts`.
The old response parser retains the original anchor and adds `FORECAST_STEP`.

A real prediction through the existing forecast-capable model route returned three steps for one supplied prediction row, with matching instance ID and row index and a driver-backed response.
The default scalar model route refuses native forecasting; use the existing forecast route for the live test.
The registered forecast driver's source requires one prediction row per native call and a horizon no greater than the context count.
Two prediction rows in one native call reproduced a server assertion failure; the new client must batch them separately and preserve both anchors, rather than silently dropping a row.
`additional_prediction_examples` supplies that alignment regression.
The client must reject an oversized horizon before network access.

This fixture stores semantic expectations, not model prediction numbers or credentials.
No warehouse, model artifact or deployment was changed for this investigation.

The old timeless default is `Timestamp.min`, but its local sampler rounds through whole seconds and overflows when converting back to nanoseconds.
Execution produced a positive near-maximum timestamp instead of the negative sentinel.
The timeless fixture intentionally requires the correct sentinel, improving this old bug rather than copying it.

The old `predict_task()` and `RFMPredictRequest` default `use_prediction_time` to false (verified in the emitted payloads).
Anchors still govern sampling and row identity; including them as a model feature is a separate option and must not be silently enabled.
