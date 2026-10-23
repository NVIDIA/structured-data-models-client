// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.sql.Connection;

/**
 * Where a schema's tables are: a database reached through the caller's own {@link Connection}, as
 * Python's {@code Source} classes wrap a driver connection (design choice 10). The library runs
 * the core's statements on it and never closes it; the caller keeps it open while the schema, a
 * request gathered from it, or a prediction is in use.
 *
 * <p>A source is one of {@link DuckDbSource}, {@link SqliteSource} and {@link PostgresSource}.
 */
public sealed interface Source permits DuckDbSource, SqliteSource, PostgresSource {
  /** The connection the tables are read through. */
  Connection connection();
}
