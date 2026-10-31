// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

/** A caller-owned database connection or Spark session; the library never closes it. */
public sealed interface Source permits JdbcSource, DatabricksSparkSource {}
