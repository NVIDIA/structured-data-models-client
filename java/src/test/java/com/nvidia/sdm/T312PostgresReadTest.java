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

  /** The first column's values of a read, as the Arrow vector gives them. */
  private static List<Object> values(Connection connection, Map<String, Object> step) throws Exception {
    JdbcProvider provider =
        new JdbcProvider(connection, JdbcProvider.Driver.POSTGRES, ConversionTable.REVIEWED);
    List<Object> out = new ArrayList<>();
    try (ArrowReader read = provider.answer(step, Map.of(), ArrowData.allocator())) {
      while (read.loadNextBatch()) {
        org.apache.arrow.vector.FieldVector column = read.getVectorSchemaRoot().getVector(0);
        for (int row = 0; row < column.getValueCount(); row++) {
          Object value = column.getObject(row);
          out.add(value == null ? null : value.toString());
        }
      }
    }
    return out;
  }

  /**
   * Claude T3.12 r2 P1-1: a float4 is the double of the server's text, as psycopg reads it, an
   * exact tie of the shortest digits included (Postgres writes {@code 1.03528336e+08}).
   */
  @Test
  void aFloat4IsTheDoubleOfTheServersText() throws Exception {
    try (Connection connection = T312Shop.open("postgres")) {
      assertEquals(
          List.of("1.03528336E8"), values(connection, step("SELECT '103528336'::float4 AS v")));
      assertEquals(
          List.of("[1.03528336E8]"),
          values(connection, step("SELECT '{103528336}'::float4[] AS v")));
    }
  }

  /**
   * Claude T3.12 r2 P1-2: on a connection that forces binary transfer ({@code
   * prepareThreshold=-1}) every value is still read in text, and the connection keeps its own
   * setting.
   */
  @Test
  void aConnectionForcingBinaryIsStillReadInText() throws Exception {
    try (Connection shop = T312Shop.open("postgres")) {
      java.net.URI uri = java.net.URI.create(System.getenv("SDM_POSTGRES_TEST_DSN"));
      String[] user = uri.getUserInfo().split(":", 2);
      try (Connection connection =
          java.sql.DriverManager.getConnection(
              shop.getMetaData().getURL() + "?prepareThreshold=-1", user[0], user[1])) {
        org.postgresql.jdbc.PgConnection pg =
            connection.unwrap(org.postgresql.jdbc.PgConnection.class);
        assertTrue(pg.getForceBinary());
        for (int run = 0; run < 7; run++) {
          assertEquals(List.of("00:00"), values(connection, step("SELECT '00:00-08'::timetz AS v")));
          assertEquals(List.of("12300"), values(connection, step("SELECT '12345'::numeric(5,-2) AS v")));
          assertEquals(List.of("(1,2)"), values(connection, step("SELECT '(1,2)'::point AS v")));
          assertEquals(List.of("[1]"), values(connection, step("SELECT '{1}'::oid[] AS v")));
        }
        assertTrue(pg.getForceBinary());
      }
    }
  }

  /**
   * Codex T3.12 r2 P0-1: a zoned range is written in the zone psycopg reads the session's in:
   * UTC for a zone Python does not name ({@code GMT+3}), the named zone otherwise.
   */
  @Test
  void aZonedRangeIsWrittenInTheZonePsycopgReadsIn() throws Exception {
    try (Connection connection = T312Shop.open("postgres")) {
      String range = "SELECT '[2024-01-01 00:00+00,2024-01-02 00:00+00)'::tstzrange AS v";
      assertEquals(
          List.of("[2024-01-01 00:00:00+00:00, 2024-01-02 00:00:00+00:00)"),
          values(connection, step("SET TimeZone = 'GMT+3'", range)));
      assertEquals(
          List.of("[2024-01-01 05:30:00+05:30, 2024-01-02 05:30:00+05:30)"),
          values(connection, step("SET TimeZone = 'Asia/Kolkata'", range)));
    }
  }
}
