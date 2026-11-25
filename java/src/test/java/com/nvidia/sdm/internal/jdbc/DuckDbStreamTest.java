// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.*;
import java.sql.DriverManager;
import com.nvidia.sdm.internal.ArrowExchange;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;

class DuckDbStreamTest {
  @Test void resultRemainsOpenAcrossBatchesAndClosesWithReader() throws Exception {
    try (var connection = DriverManager.getConnection("jdbc:duckdb:");
        var allocator = new RootAllocator()) {
      var statement = connection.createStatement();
      var rows = statement.executeQuery("SELECT i::BIGINT id, i::DECIMAL(18, 3) amount FROM range(7) t(i)");
      try (ArrowReader reader = DuckDb.arrow(rows, allocator, 2, rows, statement)) {
        assertFalse(rows.isClosed());
        assertFalse(statement.isClosed());
        int count = 0;
        while (reader.loadNextBatch()) {
          var root = reader.getVectorSchemaRoot();
          for (int i = 0; i < root.getRowCount(); i++) {
            assertEquals((long) count, root.getVector("id").getObject(i));
            assertEquals(new java.math.BigDecimal(count + ".000"), root.getVector("amount").getObject(i));
            count++;
          }
        }
        assertEquals(7, count);
      }
      assertTrue(rows.isClosed());
      assertTrue(statement.isClosed());
      assertFalse(connection.isClosed());
      try (var probe = connection.createStatement(); var result = probe.executeQuery("SELECT 42")) {
        assertTrue(result.next()); assertEquals(42, result.getInt(1));
      }
    }
  }

  @Test void closeBeforeReadReleasesOwnedCursorAndCanBeRepeated() throws Exception {
    try (var connection = DriverManager.getConnection("jdbc:duckdb:");
        var allocator = new RootAllocator()) {
      var cursor = DuckDb.duplicate(connection);
      var statement = cursor.createStatement();
      var rows = statement.executeQuery("SELECT 1");
      ArrowReader reader = DuckDb.arrow(rows, allocator, 2, rows, statement, cursor);
      reader.close(); reader.close();
      assertTrue(rows.isClosed()); assertTrue(statement.isClosed()); assertTrue(cursor.isClosed());
      assertFalse(connection.isClosed());
    }
  }
  @Test void failedExportClosesResultAndStatementButNotCallerConnection() throws Exception {
    try (var connection = DriverManager.getConnection("jdbc:duckdb:");
        var allocator = new RootAllocator();
        var noCapacity = allocator.newChildAllocator("cannot-export", 0, 0)) {
      var statement = connection.createStatement();
      var rows = statement.executeQuery("SELECT 1");
      ArrowReader reader = DuckDb.arrow(rows, allocator, 2, rows, statement);
      assertThrows(org.apache.arrow.memory.OutOfMemoryException.class,
          () -> ArrowExchange.export(noCapacity, reader));
      assertTrue(rows.isClosed()); assertTrue(statement.isClosed());
      assertFalse(connection.isClosed());
    }
  }

}
