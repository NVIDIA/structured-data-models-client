// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.sql.Connection;

/** A source accessed through the caller's JDBC connection. */
public sealed interface JdbcSource extends Source
    permits DuckDbSource, SqliteSource, PostgresSource, DatabricksSource {
  /** The caller-owned connection, kept open while reading source data. */
  Connection connection();
}
