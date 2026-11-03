// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.time.Instant;

/** Independent schema and query for the ordinary relational forecasting API. */
public record ForecastInput(Schema schema, PredictiveQuery query, Instant anchorTime) {}
