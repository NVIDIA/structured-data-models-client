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
 * A SQLite database, through a SQLite JDBC driver (such as {@code org.xerial:sqlite-jdbc}), as
 * Python's {@code SQLiteSource}: a key set a read joins to is kept in a temporary table of the
 * connection, and the database itself is left as it was.
 */
public final class SqliteSource extends BuiltInSource {
  private final Connection connection;

  private SqliteSource(Connection connection) {
    this.connection = Objects.requireNonNull(connection, "connection");
  }

  /** The SQLite database {@code connection} is open on. */
  public static SqliteSource of(Connection connection) {
    return new SqliteSource(connection);
  }

  /** The caller-owned connection, kept open while reading source data. */
  public Connection connection() {
    return connection;
  }

  @Override
  Map<String, Object> sqlChoice() {
    return Providers.sqlChoice("sqlite", null, false);
  }

  @Override
  Host.Provider driver() {
    return new JdbcProvider(connection, JdbcProvider.Driver.SQLITE);
  }

  @Override
  public String toString() {
    return "SqliteSource(" + connection + ")";
  }
}
