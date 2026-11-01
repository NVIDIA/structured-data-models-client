// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.types.pojo.Schema;
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

  /** Export the driver's stream without serializing its buffers into a Java byte array. */
  static ArrowReader arrow(ResultSet rows, BufferAllocator allocator, long batch,
      AutoCloseable... owners) throws SQLException {
    DuckDBResultSet result = rows.unwrap(DuckDBResultSet.class);
    ArrowReader exported = (ArrowReader) result.arrowExportStream(allocator, batch);
    return new Owned(exported, allocator, owners);
  }

  /** The driver stream retains its result, statement, and (for fetches) duplicate cursor. */
  private static final class Owned extends ArrowReader {
    private final ArrowReader source;
    private final AutoCloseable[] owners;
    private boolean closed;

    Owned(ArrowReader source, BufferAllocator allocator, AutoCloseable[] owners) {
      super(allocator);
      this.source = source;
      this.owners = owners;
    }

    @Override public VectorSchemaRoot getVectorSchemaRoot() throws IOException {
      return source.getVectorSchemaRoot();
    }
    @Override public Map<Long, Dictionary> getDictionaryVectors() throws IOException {
      return source.getDictionaryVectors();
    }
    @Override public Dictionary lookup(long id) { return source.lookup(id); }
    @Override public Set<Long> getDictionaryIds() { return source.getDictionaryIds(); }
    @Override public boolean loadNextBatch() throws IOException { return source.loadNextBatch(); }
    @Override public long bytesRead() { return source.bytesRead(); }
    @Override protected Schema readSchema() throws IOException {
      return source.getVectorSchemaRoot().getSchema();
    }
    @Override protected void closeReadSource() throws IOException { close(); }
    @Override public void close() throws IOException { close(true); }
    @Override public void close(boolean closeSource) throws IOException {
      if (closed) return;
      if (!closeSource) { source.close(false); return; }
      closed = true;
      Throwable failure = null;
      try { source.close(); } catch (Throwable error) { failure = error; }
      for (AutoCloseable owner : owners) {
        try { owner.close(); } catch (Throwable error) {
          if (failure == null) failure = error; else failure.addSuppressed(error);
        }
      }
      if (failure instanceof Error error) throw error;
      if (failure instanceof RuntimeException error) throw error;
      if (failure != null) throw new IOException("DuckDB Arrow stream could not be closed", failure);
    }
  }
}
