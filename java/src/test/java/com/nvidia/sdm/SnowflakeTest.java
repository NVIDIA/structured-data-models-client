// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.math.BigDecimal;
import java.net.URI;
import java.sql.Connection;
import java.sql.DriverManager;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.UUID;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.TimeNanoVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.table.Table;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.junit.jupiter.api.Test;

class SnowflakeTest {
  @Test
  void sourceUsesExistingCoreDialectNamespaceAndViewWithoutChangingConnection() throws Exception {
    try (Connection connection = DriverManager.getConnection("jdbc:sqlite::memory:")) {
      var source = SnowflakeSource.of(connection, "db", "schema").semanticModel("view", false);
      var choice = SourceAccess.sqlChoice(source);
      assertEquals("snowflake", choice.get("dialect"));
      assertEquals(List.of("db", "schema"), choice.get("namespace"));
      assertEquals(Map.of("name", "view", "trust_expressions", false), choice.get("view"));
      assertSame(connection, source.connection());
      assertTrue(SourceAccess.held(source).containsKey("semantic_model"));
      assertFalse(connection.isClosed());
    }
  }

  /**
   * A connection to the CI Snowflake account, from the same variables live parity uses; skipped
   * without them, and a failure when {@code SDM_REQUIRE_SNOWFLAKE=1} says CI must run it.
   */
  static Connection live() throws Exception {
    String url = System.getenv().getOrDefault("SNOWFLAKE_URL", "");
    boolean required = "1".equals(System.getenv("SDM_REQUIRE_SNOWFLAKE"));
    if (url.isBlank()) {
      if (required) fail("SDM_REQUIRE_SNOWFLAKE=1 but SNOWFLAKE_URL is not set");
      assumeTrue(false, "requires SNOWFLAKE_URL, SNOWFLAKE_USERNAME and SNOWFLAKE_ACCESS_TOKEN");
    }
    String host = URI.create(url.contains("://") ? url : "https://" + url).getHost();
    Properties properties = new Properties();
    properties.put("user", System.getenv("SNOWFLAKE_USERNAME"));
    properties.put("password", System.getenv("SNOWFLAKE_ACCESS_TOKEN"));
    String warehouse = System.getenv("PARITY_SNOWFLAKE_WAREHOUSE");
    if (warehouse != null && !warehouse.isBlank()) properties.put("warehouse", warehouse);
    properties.put("db", System.getenv("PARITY_SNOWFLAKE_DATABASE"));
    properties.put("TIMEZONE", "America/Los_Angeles");
    properties.put("CLIENT_OUT_OF_BAND_TELEMETRY_ENABLED", "false");
    return DriverManager.getConnection("jdbc:snowflake://" + host, properties);
  }

  @Test
  void liveSchemaFromSourceDescribesAndChecksKeysThroughTheSource() throws Exception {
    try (Connection connection = live()) {
      String database = System.getenv("PARITY_SNOWFLAKE_DATABASE");
      String schema = "SDM_CI_JAVA_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12).toUpperCase();
      try (var statement = connection.createStatement()) {
        statement.execute("CREATE SCHEMA " + schema);
        try {
          statement.execute("CREATE TABLE " + schema + ".USERS (USER_ID NUMBER(38,0) PRIMARY KEY, COUNTRY VARCHAR)");
          statement.execute(
              "CREATE TABLE " + schema + ".EVENTS (USER_ID NUMBER(38,0), KIND VARCHAR, TS TIMESTAMP_NTZ)");
          statement.execute("INSERT INTO " + schema + ".USERS VALUES (1, 'DE'), (2, 'US')");
          statement.execute(
              "INSERT INTO " + schema + ".EVENTS VALUES (1, 'a', '2024-01-01'), (1, 'a', '2024-01-01'),"
                  + " (2, 'b', '2024-01-02')");
          Schema described = Schema.fromSource(SnowflakeSource.of(connection, database, schema));
          assertEquals(
              List.of("EVENTS", "USERS"),
              described.tables().stream().map(com.nvidia.sdm.Table::name).sorted().toList());
          assertEquals(List.of("USER_ID"), described.table("USERS").primaryKey());
          assertEquals(List.of(), described.table("EVENTS").primaryKey());
          assertEquals("TS", described.table("EVENTS").timeColumn().orElseThrow());
        } finally {
          statement.execute("DROP SCHEMA IF EXISTS " + schema);
        }
      }
    }
  }

  @Test
  void liveExactNumbersTemporalValuesAndCallerSettings() throws Exception {
    try (Connection connection = live();
        RootAllocator allocator = new RootAllocator()) {
      var input = SnowflakeSource.of(connection);
      String sql =
          "SELECT 9007199254740993::NUMBER(38,0) AS small, "
              + "1208925819614629174706177::NUMBER(38,0) AS wide, "
              + "1.234567890123456789::NUMBER(38,18) AS fraction, NULL::NUMBER(38,0) AS missing, "
              + "'12:34:56.123456789'::TIME(9) AS clock, "
              + "'2024-03-10 01:23:45.123456789'::TIMESTAMP_NTZ(9) AS ntz, "
              + "'2024-03-10 01:23:45.123456789 -08:00'::TIMESTAMP_TZ(9) AS tz, "
              + "'2024-03-10 01:23:45.123456789 -08:00'::TIMESTAMP_LTZ(9) AS ltz";
      try (Table table = Readers.readQuery(input.connection(), sql, allocator)) {
        assertEquals(new BigDecimal("9007199254740993"), value(table, 0));
        assertEquals(new BigDecimal("1208925819614629174706177"), value(table, 1));
        assertEquals(new BigDecimal("1.234567890123456789"), value(table, 2));
        assertNull(value(table, 3));
        try (var time = table.getVectorCopy(4)) {
          assertEquals(45_296_123_456_789L, ((TimeNanoVector) time).get(0));
        }
        for (int i = 5; i < 8; i++) {
          try (var timestamp = table.getVectorCopy(i)) {
            assertEquals(
                i == 5 ? 1710033825123456789L : 1710062625123456789L,
                ((TimeStampVector) timestamp).get(0));
            assertEquals(
                new ArrowType.Timestamp(TimeUnit.NANOSECOND, i == 5 ? null : "America/Los_Angeles"),
                timestamp.getField().getType());
          }
        }
      }
      try (var statement = connection.createStatement();
          var rows = statement.executeQuery("SHOW PARAMETERS LIKE 'TIMEZONE' IN SESSION")) {
        assertTrue(rows.next());
        assertEquals("America/Los_Angeles", rows.getString("value"));
      }
      try (Table times =
          Readers.readQuery(
              input.connection(),
              "SELECT '2024-03-10 01:23:45'::TIMESTAMP_NTZ(0) AS s, "
                  + "'2024-03-10 01:23:45.123'::TIMESTAMP_NTZ(3) AS ms, "
                  + "'2024-03-10 01:23:45.123456'::TIMESTAMP_NTZ(6) AS us",
              allocator)) {
        var units = List.of(TimeUnit.SECOND, TimeUnit.MILLISECOND, TimeUnit.MICROSECOND);
        long[] expected = {1710033825L, 1710033825123L, 1710033825123456L};
        for (int i = 0; i < 3; i++)
          try (var vector = times.getVectorCopy(i)) {
            assertEquals(new ArrowType.Timestamp(units.get(i), null), vector.getField().getType());
            assertEquals(expected[i], ((TimeStampVector) vector).get(0));
          }
      }
      try (Table vectors =
          Readers.readQuery(
              input.connection(),
              "SELECT CAST([1.25,2.5] AS VECTOR(FLOAT,2)) AS v, CAST([1,NULL,3] AS"
                  + " ARRAY(NUMBER(38,0))) AS a",
              allocator)) {
        assertEquals(List.of(1.25f, 2.5f), value(vectors, 0));
        assertEquals("[\n  1,\n  undefined,\n  3\n]", value(vectors, 1).toString());
        try (var vector = vectors.getVectorCopy(0)) {
          assertEquals(new ArrowType.FixedSizeList(2), vector.getField().getType());
        }
      }
      assertFalse(connection.isClosed());
    }
  }

  private static Object value(Table table, int column) {
    try (var vector = table.getVectorCopy(column)) {
      return vector.getObject(0);
    }
  }
}
