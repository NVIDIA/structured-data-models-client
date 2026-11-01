// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.Providers;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.util.List;
import java.util.Map;
import java.util.Properties;
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
      var choice = Providers.of(source).choice();
      assertEquals("snowflake", choice.get("dialect"));
      assertEquals(List.of("db", "schema"), choice.get("namespace"));
      assertEquals(Map.of("name", "view", "trust_expressions", false), choice.get("view"));
      assertSame(connection, source.connection());
      assertTrue(Providers.held(source).containsKey("semantic_model"));
      assertFalse(connection.isClosed());
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  void liveExactNumbersTemporalValuesAndCallerSettings() throws Exception {
    String config = System.getProperty("sdm.snowflake.config", "");
    assumeTrue(!config.isBlank(), "requires private Snowflake configuration");
    var root = (Map<String, Object>) Json.read(Files.readAllBytes(Path.of(config)));
    var source = (Map<String, Object>) ((Map<String, Object>) root.get("sources")).get("snowflake");
    var connect = (Map<String, Object>) source.get("connect");
    Properties properties = new Properties();
    for (String key : List.of("user", "authenticator", "token", "warehouse", "schema")) {
      if (connect.containsKey(key)) properties.put(key, connect.get(key));
    }
    properties.put("db", connect.get("database"));
    properties.put("TIMEZONE", "America/Los_Angeles");
    properties.put("CLIENT_OUT_OF_BAND_TELEMETRY_ENABLED", "false");
    String url = "jdbc:snowflake://" + connect.get("account") + ".snowflakecomputing.com";
    try (Connection connection = DriverManager.getConnection(url, properties);
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
      try (Table table = Readers.readQuery(input, sql, allocator)) {
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
              input,
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
              input,
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
