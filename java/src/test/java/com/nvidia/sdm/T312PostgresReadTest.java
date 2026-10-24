// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.jdbc.ConversionTable;
import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.postgresql.core.BaseConnection;
import org.postgresql.core.TransactionState;

/**
 * Claude T3.12 r1 P2-3: a Postgres read takes a transaction as Python's provider takes one
 * (psycopg's {@code connection.transaction()}): none under autocommit; outside a transaction, one
 * of its own, committed; inside the caller's, a savepoint, the caller's work left to the caller.
 */
class T312PostgresReadTest {
  /** A fetch step running {@code statements} on Postgres, the last one's rows read. */
  private static Map<String, Object> step(String... statements) {
    List<Object> planned = new ArrayList<>();
    for (String text : statements) {
      planned.add(Map.of("dialect", "postgres", "text", text, "params", List.of()));
    }
    return Map.of("plan", Map.of("plan", "sql", "statements", planned));
  }

  private static int rows(Connection connection, Map<String, Object> step) throws Exception {
    JdbcProvider provider =
        new JdbcProvider(connection, JdbcProvider.Driver.POSTGRES, ConversionTable.REVIEWED);
    try (ArrowReader read = provider.answer(step, Map.of(), ArrowData.allocator())) {
      int count = 0;
      while (read.loadNextBatch()) {
        count += read.getVectorSchemaRoot().getRowCount();
      }
      return count;
    }
  }

  private static boolean exists(Connection connection, String table) throws SQLException {
    try (Statement statement = connection.createStatement();
        ResultSet found = statement.executeQuery("SELECT to_regclass('" + table + "') IS NOT NULL")) {
      found.next();
      return found.getBoolean(1);
    }
  }

  private static TransactionState state(Connection connection) throws SQLException {
    return connection.unwrap(BaseConnection.class).getTransactionState();
  }

  @Test
  void aReadOutsideATransactionCommitsItsOwn() throws Exception {
    try (Connection connection = T312Shop.open("postgres")) {
      connection.setAutoCommit(false);
      connection.commit();
      assertEquals(TransactionState.IDLE, state(connection));
      assertEquals(
          1, rows(connection, step("CREATE TEMPORARY TABLE read_made AS SELECT 1 AS v", "SELECT v FROM read_made")));
      // Its own transaction, committed: nothing left open, and what it did kept.
      assertEquals(TransactionState.IDLE, state(connection));
      connection.rollback();
      assertTrue(exists(connection, "pg_temp.read_made"));
    }
  }

  @Test
  void aReadInsideTheCallersTransactionLeavesItToTheCaller() throws Exception {
    try (Connection connection = T312Shop.open("postgres")) {
      connection.setAutoCommit(false);
      try (Statement statement = connection.createStatement()) {
        statement.execute("CREATE TEMPORARY TABLE callers AS SELECT 1 AS v");
      }
      assertEquals(TransactionState.OPEN, state(connection));
      assertEquals(1, rows(connection, step("SELECT v FROM callers")));
      assertEquals(TransactionState.OPEN, state(connection));
      connection.rollback();
      assertFalse(exists(connection, "pg_temp.callers"));
    }
  }

  @Test
  void aReadUnderAutocommitTakesNoTransaction() throws Exception {
    try (Connection connection = T312Shop.open("postgres")) {
      assertTrue(connection.getAutoCommit());
      assertEquals(1, rows(connection, step("SELECT 1 AS v")));
      assertEquals(TransactionState.IDLE, state(connection));
    }
  }

  /**
   * Codex T3.12 r1 P0-2, Claude P1-3: a value is read by its type's identity (a user's enum
   * array is text, as psycopg reads a type its registry does not name), and one psycopg refuses
   * is refused, never read as another value: an infinite or BC date, a time of 24:00, an
   * interval written in a style other than {@code postgres}. (Value by value against Python's
   * provider: {@code tests_test_ops/test_t312_driver_parity.py}.)
   */
  @Test
  void aValueIsReadByItsTypeOrRefusedAsPsycopgRefusesIt() throws Exception {
    try (Connection connection = T312Shop.open("postgres")) {
      try (Statement statement = connection.createStatement()) {
        statement.execute("CREATE TYPE mood AS ENUM ('sad', 'happy')");
      }
      JdbcProvider provider =
          new JdbcProvider(connection, JdbcProvider.Driver.POSTGRES, ConversionTable.REVIEWED);
      try (ArrowReader read =
          provider.answer(step("SELECT '{happy,sad}'::mood[] AS v"), Map.of(), ArrowData.allocator())) {
        read.loadNextBatch();
        assertEquals("Utf8", read.getVectorSchemaRoot().getSchema().getFields().get(0).getType().toString());
        assertEquals("{happy,sad}", read.getVectorSchemaRoot().getVector(0).getObject(0).toString());
      }
      for (String refused :
          List.of(
              "SELECT 'infinity'::date AS v",
              "SELECT '0044-03-15 BC'::date AS v",
              "SELECT 'infinity'::timestamp AS v",
              "SELECT '24:00:00'::time AS v",
              "SELECT '24:00:00+00'::timetz AS v",
              "SELECT '{infinity}'::date[] AS v")) {
        org.junit.jupiter.api.Assertions.assertThrows(
            Exception.class, () -> rows(connection, step(refused)), refused);
      }
      org.junit.jupiter.api.Assertions.assertThrows(
          Exception.class,
          () -> rows(connection, step("SET IntervalStyle = 'iso_8601'", "SELECT '1 day'::interval AS v")));
      assertEquals(
          1, rows(connection, step("SET IntervalStyle = 'postgres'", "SELECT '1 day'::interval AS v")));
    }
  }
}
