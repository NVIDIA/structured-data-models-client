// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Providers;
import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import java.sql.Connection;
import java.util.Map;
import java.util.Objects;

/**
 * A DuckDB database, through DuckDB's JDBC driver ({@code org.duckdb:duckdb_jdbc}), as Python's
 * {@code DuckDBSource}: each read runs on a cursor of its own ({@code duplicate()}), and the core's
 * statements read the tables of the connection's current schema.
 */
public final class DuckDbSource extends BuiltInSource {
  private final Connection connection;

  private DuckDbSource(Connection connection) {
    this.connection = Objects.requireNonNull(connection, "connection");
  }

  /** The DuckDB database {@code connection} is open on; a connection of DuckDB's driver. */
  public static DuckDbSource of(Connection connection) {
    return new DuckDbSource(connection);
  }

  /** The caller-owned connection, kept open while reading source data. */
  public Connection connection() {
    return connection;
  }

  @Override
  Map<String, Object> sqlChoice() {
    return Providers.sqlChoice("duckdb", null, false);
  }

  @Override
  Host.Provider driver() {
    return new JdbcProvider(connection, JdbcProvider.Driver.DUCKDB);
  }

  @Override
  public String toString() {
    return "DuckDbSource(" + connection + ")";
  }
}
