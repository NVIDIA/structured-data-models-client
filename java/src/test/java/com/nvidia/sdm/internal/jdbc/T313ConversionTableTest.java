// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.internal.ArrowData;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Field;
import org.junit.jupiter.api.Test;

/**
 * The Databricks column of the conversion table (T3.13), per type of {@code
 * java.databricks-sql.types}: each column the Databricks JDBC driver describes reaches the core as
 * the Arrow type the Databricks SQL connector gives Python's provider for it (C-1's recorded
 * tables: an integer at its width, a float at its width, a decimal at its precision and scale, a
 * time stamp at microseconds with no zone), each null a null where the driver's getter reads 0 or
 * false, a NaN kept, an integer past 2**53 and a decimal past a double's precision exact.
 */
class T313ConversionTableTest {
  private record Read(Field field, List<Object> values) {}

  /** The answer of a provider over {@code columns}, by column name. */
  private static Map<String, Read> answered(FakeJdbc.Column... columns) throws Exception {
    FakeJdbc.Database database = new FakeJdbc.Database(text -> List.of(columns));
    JdbcProvider provider =
        new JdbcProvider(database.connection(), JdbcProvider.Driver.DATABRICKS);
    ArrowReader answer =
        provider.answer(
            T313DatabricksProviderTest.plan(T313DatabricksProviderTest.statement("SELECT *")),
            new HashMap<>(),
            ArrowData.allocator());
    Map<String, Read> out = new LinkedHashMap<>();
    try (answer) {
      VectorSchemaRoot root = answer.getVectorSchemaRoot();
      for (Field field : root.getSchema().getFields()) {
        out.put(field.getName(), new Read(field, new ArrayList<>()));
      }
      while (answer.loadNextBatch()) {
        for (FieldVector vector : root.getFieldVectors()) {
          for (int row = 0; row < root.getRowCount(); row++) {
            out.get(vector.getName()).values().add(vector.getObject(row));
          }
        }
      }
    }
    return out;
  }

  private static List<Object> values(Object... values) {
    return Arrays.asList(values);
  }

  @Test
  void eachScalarTypeReachesTheCoreAsTheConnectorsArrow() throws Exception {
    Map<String, Read> got =
        answered(
            FakeJdbc.Column.of("tier", "TINYINT", (byte) 1, null),
            FakeJdbc.Column.of("visits", "SMALLINT", (short) 200, null),
            FakeJdbc.Column.of("age", "INT", 31, null),
            FakeJdbc.Column.of("points", "BIGINT", 9007199254740993L, null),
            FakeJdbc.Column.of("ratio", "FLOAT", 0.25f, null),
            FakeJdbc.Column.of("score", "DOUBLE", Double.NaN, null),
            FakeJdbc.Column.decimal(
                "balance", 38, 6, new BigDecimal("12345678901234567890.123456"), null),
            FakeJdbc.Column.of("country", "STRING", "Zoë", null),
            FakeJdbc.Column.of("active", "BOOLEAN", true, null),
            FakeJdbc.Column.of("placed_on", "DATE", LocalDate.of(2023, 5, 3), null),
            FakeJdbc.Column.of(
                "shipped_at", "TIMESTAMP", Instant.parse("2023-06-01T02:00:00.000123Z"), null),
            FakeJdbc.Column.of(
                "delivered_at", "TIMESTAMP_NTZ", LocalDateTime.of(2023, 6, 2, 0, 0, 1), null));
    Map<String, String> types = new LinkedHashMap<>();
    got.forEach((name, read) -> types.put(name, read.field().getType().toString()));
    assertEquals(
        Map.ofEntries(
            Map.entry("tier", "Int(8, true)"),
            Map.entry("visits", "Int(16, true)"),
            Map.entry("age", "Int(32, true)"),
            Map.entry("points", "Int(64, true)"),
            Map.entry("ratio", "FloatingPoint(SINGLE)"),
            Map.entry("score", "FloatingPoint(DOUBLE)"),
            Map.entry("balance", "Decimal(38, 6, 128)"),
            Map.entry("country", "Utf8"),
            Map.entry("active", "Bool"),
            Map.entry("placed_on", "Date(DAY)"),
            Map.entry("shipped_at", "Timestamp(MICROSECOND, null)"),
            Map.entry("delivered_at", "Timestamp(MICROSECOND, null)")),
        types);
    // Each null a null, never the getter's zero.
    got.forEach((name, read) -> assertNull(read.values().get(1), name));
    assertEquals((byte) 1, got.get("tier").values().get(0));
    assertEquals((short) 200, got.get("visits").values().get(0));
    assertEquals(31, got.get("age").values().get(0));
    assertEquals(9007199254740993L, got.get("points").values().get(0));
    assertEquals(0.25f, got.get("ratio").values().get(0));
    assertTrue(Double.isNaN((Double) got.get("score").values().get(0)));
    assertEquals(
        new BigDecimal("12345678901234567890.123456"), got.get("balance").values().get(0));
    assertEquals("Zoë", got.get("country").values().get(0).toString());
    assertEquals(true, got.get("active").values().get(0));
    assertEquals(
        (int) LocalDate.of(2023, 5, 3).toEpochDay(),
        ((Number) got.get("placed_on").values().get(0)).intValue());
    // A TIMESTAMP is its instant, a TIMESTAMP_NTZ its wall time, each at microseconds.
    assertEquals(
        LocalDateTime.of(2023, 6, 1, 2, 0, 0, 123_000), got.get("shipped_at").values().get(0));
    assertEquals(
        LocalDateTime.of(2023, 6, 2, 0, 0, 1), got.get("delivered_at").values().get(0));
  }

  @Test
  void zonedInstantsAndZoneFreeWallTimesSurviveDstAndANonUtcJvm() throws Exception {
    java.util.TimeZone original = java.util.TimeZone.getDefault();
    try {
      for (String zone : List.of("America/Los_Angeles", "Europe/Berlin", "Asia/Kolkata")) {
        java.util.TimeZone.setDefault(java.util.TimeZone.getTimeZone(zone));
        Instant before = Instant.parse("2024-03-10T09:59:59.123456Z");
        Instant after = Instant.parse("2024-03-10T10:00:00.654321Z");
        LocalDateTime wall = LocalDateTime.parse("2024-03-10T02:30:00.123456");
        Map<String, Read> got = answered(
            FakeJdbc.Column.of("zoned", "TIMESTAMP", before, after),
            FakeJdbc.Column.of("unzoned", "TIMESTAMP_NTZ", wall, wall));
        assertEquals(List.of(LocalDateTime.ofInstant(before, java.time.ZoneOffset.UTC),
            LocalDateTime.ofInstant(after, java.time.ZoneOffset.UTC)), got.get("zoned").values(), zone);
        assertEquals(List.of(wall, wall), got.get("unzoned").values(), zone);
      }
    } finally {
      java.util.TimeZone.setDefault(original);
    }
  }

  @Test
  void aBinaryAnArrayAndAVoidAreTheConnectorsAndAnyOtherTypeIsTheDriversText()
      throws Exception {
    byte[] token = "u1".getBytes(StandardCharsets.UTF_8);
    Map<String, Read> got =
        answered(
            FakeJdbc.Column.of("token", "BINARY", token, null, null),
            FakeJdbc.Column.of("tags", "ARRAY<INT>", List.of(1, 2), List.of(), null),
            FakeJdbc.Column.of("nothing", "VOID", null, null, null),
            FakeJdbc.Column.of("attrs", "STRUCT<a: INT>", "{\"a\":1}", null, null),
            FakeJdbc.Column.of("span", "INTERVAL DAY TO SECOND", "1 02:00:00.000000000", null, null),
            FakeJdbc.Column.of("label", "VARCHAR(10)", "x", null, null));
    assertEquals("Binary", got.get("token").field().getType().toString());
    assertArrayEquals(token, (byte[]) got.get("token").values().get(0));
    assertEquals("List", got.get("tags").field().getType().toString());
    assertEquals(
        "Int(32, true)", got.get("tags").field().getChildren().get(0).getType().toString());
    assertEquals(List.of(1, 2), got.get("tags").values().get(0));
    assertEquals(List.of(), got.get("tags").values().get(1));
    assertNull(got.get("tags").values().get(2));
    assertEquals("Null", got.get("nothing").field().getType().toString());
    assertEquals("Utf8", got.get("attrs").field().getType().toString());
    assertEquals("{\"a\":1}", got.get("attrs").values().get(0).toString());
    assertEquals("Utf8", got.get("span").field().getType().toString());
    assertEquals("Utf8", got.get("label").field().getType().toString());
    assertEquals("x", got.get("label").values().get(0).toString());
  }

  @Test
  void theTableSaysWhatEachDatabricksTypeReachesTheCoreAs() {
    for (String type :
        List.of(
            "TINYINT", "SMALLINT", "INT", "BIGINT", "FLOAT", "DOUBLE", "DECIMAL", "STRING",
            "BOOLEAN", "DATE", "TIMESTAMP", "TIMESTAMP_NTZ", "BINARY", "ARRAY", "VOID")) {
      assertTrue(ConversionTable.DATABRICKS.containsKey(type), type);
    }
  }
}
