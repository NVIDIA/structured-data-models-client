// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.NativeCore;
import com.nvidia.sdm.internal.PythonText;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.arrow.vector.types.pojo.Field;
import org.junit.jupiter.api.Test;

/**
 * The JDBC-to-Arrow conversion table's driver columns (T3.12; plan section 3, "JDBC types: host,
 * as a reviewed table"), each held to what the reference's Python driver gives for the same
 * values, as recorded from it: ADBC SQLite's type inference and its text of a value, psycopg's
 * reading of an interval and of a JSON value, pyarrow's text of a float, and the Postgres key
 * set's parameter.
 */
class T312ConversionTableTest {
  private static ConversionTable.Result sqlite(String setup, String query) throws SQLException {
    try (Connection connection = DriverManager.getConnection("jdbc:sqlite::memory:");
        Statement statement = connection.createStatement()) {
      for (String each : setup.split(";")) {
        if (!each.isBlank()) {
          statement.execute(each);
        }
      }
      try (ResultSet rows = statement.executeQuery(query)) {
        return ConversionTable.sqlite(rows);
      }
    }
  }

  private static List<String> types(ConversionTable.Result result) {
    return result.fields().stream().map(Field::getType).map(Object::toString).toList();
  }

  /** As ADBC SQLite 1.12 reads it (probed from the reference's driver). */
  @Test
  void sqliteColumnsAreTypedByTheirFirstBatchAsAdbcTypesThem() throws Exception {
    ConversionTable.Result result =
        sqlite(
            "CREATE TABLE t(a INTEGER, b REAL, c TEXT, d, e BOOLEAN, g BLOB, h NUMERIC);"
                + "INSERT INTO t VALUES (1, 1.5, 'x', 1, 1, x'6162', '1.5');"
                + "INSERT INTO t VALUES (NULL, NULL, NULL, NULL, NULL, NULL, NULL);"
                + "INSERT INTO t VALUES (2, 2, 'y', 's', 0, x'', 3);"
                + "INSERT INTO t VALUES (3, 't', 5, 2.5, 1, x'63', NULL)",
            "SELECT * FROM t");
    assertEquals(
        List.of("Int(64, true)", "Utf8", "Utf8", "Utf8", "Int(64, true)", "Binary", "FloatingPoint(DOUBLE)"),
        types(result));
    assertEquals(Arrays.asList("1.500000e+00", null, "2.000000e+00", "t"), result.columns().get(1));
    assertEquals(Arrays.asList("1", null, "s", "2.500000e+00"), result.columns().get(3));
    assertEquals(Arrays.asList(1.5, null, 3.0, null), result.columns().get(6));
  }

  @Test
  void anEmptyOrNullOnlySqliteColumnIsInt64() throws Exception {
    assertEquals(
        List.of("Int(64, true)"), types(sqlite("CREATE TABLE t(a TEXT)", "SELECT a FROM t")));
    assertEquals(List.of("Int(64, true)"), types(sqlite("", "SELECT NULL AS x")));
    assertEquals(
        List.of("FloatingPoint(DOUBLE)"), types(sqlite("", "SELECT 1 AS x UNION ALL SELECT 2.5")));
  }

  @Test
  void aSqliteValueAfterTheFirstBatchMustFitItsColumn() throws Exception {
    try (Connection connection = DriverManager.getConnection("jdbc:sqlite::memory:");
        Statement statement = connection.createStatement()) {
      statement.execute("CREATE TABLE t(a, s)");
      try (PreparedStatement insert = connection.prepareStatement("INSERT INTO t VALUES (?, ?)")) {
        for (int at = 0; at < 2000; at++) {
          insert.setLong(1, at);
          insert.setString(2, "a");
          insert.addBatch();
        }
        insert.executeBatch();
      }
      statement.execute("UPDATE t SET s = 8.25 WHERE rowid = 1501");
      try (ResultSet rows = statement.executeQuery("SELECT s FROM t")) {
        ConversionTable.Result result = ConversionTable.sqlite(rows);
        // A real after the first batch, in a text column, as SQLite's own text of it.
        assertEquals("8.25", result.columns().get(0).get(1500));
      }
      statement.execute("UPDATE t SET a = 'x' WHERE rowid = 1501");
      try (ResultSet rows = statement.executeQuery("SELECT a FROM t")) {
        assertThrows(SQLException.class, () -> ConversionTable.sqlite(rows));
      }
    }
  }

  /**
   * Codex T3.12 r1 P0-1: C's {@code %e}, as ADBC SQLite writes a real into a text column, rounds
   * the double's exact value to seven digits, an exact tie to even (glibc's printf), never half up
   * (as Python's {@code '%e' % value} writes each, probed).
   */
  @Test
  void aRealIsWrittenAsCsExponentFormWritesIt() {
    double[] values = {1234562.5, 1234563.5, 0.1, 1e300, 2.5e-7, 1.0000005, 1.0000015, -0.0, 0.0, 9.9999995, 1e-310};
    String[] written = {
      "1.234562e+06", "1.234564e+06", "1.000000e-01", "1.000000e+300", "2.500000e-07",
      "1.000001e+00", "1.000001e+00", "-0.000000e+00", "0.000000e+00", "9.999999e+00", "1.000000e-310"
    };
    for (int at = 0; at < values.length; at++) {
      assertEquals(written[at], ConversionTable.cExponent(values[at]), Double.toString(values[at]));
    }
    assertEquals("inf", ConversionTable.cExponent(Double.POSITIVE_INFINITY));
    assertEquals("-inf", ConversionTable.cExponent(Double.NEGATIVE_INFINITY));
  }

  /** psycopg's timedelta of each, as days, seconds and microseconds (probed). */
  @Test
  void anIntervalIsReadAsPsycopgReadsIt() throws Exception {
    long[][] expected = {{32, 10800, 0}, {-2, 86399, 0}, {365, 0, 0}, {0, 0, 500000}, {422, 14706, 789000}, {-1, 86399, 999999}};
    String[] given = {"1 mon 2 days 03:00:00", "-1 days -00:00:01", "1 year", "00:00:00.5", "1 year 2 mons -3 days +04:05:06.789", "-00:00:00.000001"};
    for (int at = 0; at < given.length; at++) {
      long micros = (expected[at][0] * 86_400L + expected[at][1]) * 1_000_000L + expected[at][2];
      PostgresText.Delta read = PostgresText.interval(given[at], "postgres");
      assertEquals(micros, read.days() * 86_400_000_000L + read.micros(), given[at]);
    }
  }

  /**
   * {@code json.dumps(json.loads(text), separators=(',', ':'))} (probed), as the core writes it
   * (Codex T3.12 r1 P1-4: the rule is the core's alone, at any depth Python's reader reads), a
   * JSON null none.
   */
  @Test
  void aJsonValueIsWrittenBackAsPythonWritesIt() {
    assertEquals(
        "{\"a\":2,\"b\":[1.0,1e+16,-0.0,\"\\u00e9\\ud800\",100.0,0,12345678901234567890]}",
        NativeCore.jsonText(
            "{\"a\": 1, \"b\": [1.0, 1e16, -0.0, \"\\u00e9\\ud800\", 1E2, -0, 12345678901234567890], \"a\": 2}"));
    assertEquals("[true,null,\"x\\n\\u0001\\u007f\"]", NativeCore.jsonText("[true,null,\"x\\n\\u0001\\u007f\"]"));
    assertEquals(
        "{\"k\":0.1,\"z\":Infinity,\"e\":1.5e-05,\"n\":0.0001}",
        NativeCore.jsonText("{\"k\": 0.1, \"z\": 1e400, \"e\": 1.5e-05, \"n\": 0.0001}"));
    assertNull(NativeCore.jsonText("null"));
    String deep = "[".repeat(920) + "]".repeat(920);
    assertEquals(deep, NativeCore.jsonText(deep));
  }

  /** Python's {@code repr} of each (probed). */
  @Test
  void aFloatIsWrittenAsPythonsRepr() {
    double[] values = {1.0, 1e16, 1e-5, 0.0001, 123456789012345678.0, -0.0, 0.1, 1.5, 100.0, 9007199254740993.0, 5e-324, 1.7976931348623157e308};
    String[] written = {"1.0", "1e+16", "1e-05", "0.0001", "1.2345678901234568e+17", "-0.0", "0.1", "1.5", "100.0", "9007199254740992.0", "5e-324", "1.7976931348623157e+308"};
    for (int at = 0; at < values.length; at++) {
      assertEquals(written[at], PythonText.floatRepr(values[at]));
    }
  }

  /** pyarrow's cast of each to text (probed). */
  @Test
  void aFloatIsWrittenAsArrowWritesIt() {
    double[] values = {1.5, 1e16, 1e21, 0.1, 123456789.0, 1e-7, -0.0, Double.POSITIVE_INFINITY, Math.pow(2, 70), 1e15, 12345678901234567.0};
    String[] written = {"1.5", "1e+16", "1e+21", "0.1", "123456789", "1e-7", "-0", "inf", "1.1805916207174113e+21", "1e+15", "1.2345678901234568e+16"};
    for (int at = 0; at < values.length; at++) {
      assertEquals(written[at], ArrowText.floating(values[at]));
    }
    assertEquals("0.1", ArrowText.floating32(0.1f));
    assertEquals("1.5", ArrowText.floating32(1.5f));
  }

  @Test
  void aPostgresKeySetIsItsStatementsOneParameter() throws Exception {
    assertEquals(
        "SELECT ?::jsonb, '$1 ?', \"a$1\", ?? FROM t",
        KeySets.jdbcText("SELECT $1::jsonb, '$1 ?', \"a$1\", ? FROM t"));
    assertThrows(JdbcProvider.PlanNotRun.class, () -> KeySets.jdbcText("SELECT 1"));
    assertThrows(JdbcProvider.PlanNotRun.class, () -> KeySets.jdbcText("SELECT $1, $1"));
    List<List<Object>> columns = new ArrayList<>();
    columns.add(Arrays.asList(1L, null));
    columns.add(Arrays.asList("é\"", 2.5));
    KeySets.Keys keys =
        new KeySets.Keys(
            new byte[0],
            List.of(ArrowColumns.field("a", ArrowColumns.INT64, null), ArrowColumns.field("b", ArrowColumns.UTF8, null)),
            columns);
    assertEquals("[[\"1\",\"é\\\"\"],[null,\"2.5\"]]", KeySets.postgresText(keys));
  }

  @Test
  void theReviewedTableCanBeSeededForL5() {
    ConversionTable seeded =
        ConversionTable.REVIEWED.withPostgres(
            "timestamp", (p, s) -> ConversionTable.REVIEWED.postgres("timestamptz", p, s));
    assertEquals(
        "Timestamp(MICROSECOND, UTC)",
        seeded.postgres("timestamp", 26, 6).field("t").getType().toString());
    assertEquals(
        "Timestamp(MICROSECOND, null)",
        ConversionTable.REVIEWED.postgres("timestamp", 26, 6).field("t").getType().toString());
    assertEquals(ArrowData.allocator(), ArrowData.allocator());
  }
}
