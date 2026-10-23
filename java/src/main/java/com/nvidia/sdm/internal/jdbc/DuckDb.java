// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import com.nvidia.sdm.internal.ArrowData;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.duckdb.DuckDBConnection;
import org.duckdb.DuckDBResultSet;

/**
 * DuckDB's driver's own Arrow interface, which the provider reaches only for a DuckDB connection,
 * so the driver need be on the class path only where one is used: a cursor of the connection, a
 * stream registered as a relation, a result exported as Arrow by the driver (the same conversion
 * as Python's DuckDB driver's). Not part of the API.
 */
final class DuckDb {
  private DuckDb() {}

  private static DuckDBConnection duckdb(Connection connection) throws SQLException {
    if (connection.isWrapperFor(DuckDBConnection.class)) {
      return connection.unwrap(DuckDBConnection.class);
    }
    throw new SQLException("a DuckDB source's connection is DuckDB's driver's");
  }

  /** A connection of its own to the same database: the cursor a fetch runs on. */
  static Connection duplicate(Connection connection) throws SQLException {
    return duckdb(connection).duplicate();
  }

  /** {@code stream} registered on {@code connection} as the relation {@code name}. */
  static void register(Connection connection, String name, ArrowArrayStream stream)
      throws SQLException {
    duckdb(connection).registerArrowStream(name, stream);
  }

  /**
   * {@code rows}, every batch the driver exports of at most {@code batch} rows, read whole into
   * the host's memory, so the result outlives the cursor it was read on.
   */
  static ArrowReader arrow(ResultSet rows, BufferAllocator allocator, long batch)
      throws SQLException {
    DuckDBResultSet result = rows.unwrap(DuckDBResultSet.class);
    ArrowReader exported = (ArrowReader) result.arrowExportStream(allocator, batch);
    return ArrowData.reader(ArrowData.ipc(exported), allocator);
  }
}
