// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

/**
 * Where a schema's data is read: a caller-owned database connection or Spark session, or the
 * caller's own {@link com.nvidia.sdm.provider.DataProvider} ({@link ProviderSource}). The library
 * never closes it.
 */
public sealed interface Source permits JdbcSource, DatabricksSparkSource, ProviderSource {}
