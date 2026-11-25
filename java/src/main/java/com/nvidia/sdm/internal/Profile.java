// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

/** The profiles a query is validated for, as the contract names them. */
public enum Profile {
  FINE_TUNING,
  RELATIONAL,
  TABULAR,
  TIMESERIES,
  ANOMALY,
}
