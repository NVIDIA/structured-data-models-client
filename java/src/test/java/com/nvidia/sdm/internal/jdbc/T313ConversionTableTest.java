// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
    return answeredWithConversions(List.of(), columns);
  }

  private static Map<String, Read> answeredWithConversions(
      List<Map<String, Object>> conversions, FakeJdbc.Column... columns) throws Exception {
    FakeJdbc.Database database = new FakeJdbc.Database(text -> List.of(columns));
    JdbcProvider provider =
        new JdbcProvider(database.connection(), JdbcProvider.Driver.DATABRICKS);
    ArrowReader answer =
        provider.answer(
            Map.of("plan", Map.of("plan", "sql", "statements",
                List.of(T313DatabricksProviderTest.statement("SELECT *")),
                "result_conversions", conversions)),
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

  private static List<String> utf8Items(Read column, int row) {
    assertEquals("List", column.field().getType().toString());
    assertEquals("Utf8", column.field().getChildren().get(0).getType().toString());
    List<?> items = assertInstanceOf(List.class, column.values().get(row));
    List<String> text = new ArrayList<>(items.size());
    for (Object item : items) {
      text.add(item == null ? null
          : assertInstanceOf(org.apache.arrow.vector.util.Text.class, item).toString());
    }
    return text;
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
        LocalDateTime europeanGap = LocalDateTime.parse("2024-03-31T02:30:00.654321");
        Map<String, Read> got = answeredWithConversions(
            List.of(Map.of("conversion", "timestamp_micros", "column", 1, "name", "unzoned")),
            FakeJdbc.Column.of("zoned", "TIMESTAMP", before, after),
            FakeJdbc.Column.of("unzoned", "BIGINT",
                wall.toEpochSecond(java.time.ZoneOffset.UTC) * 1_000_000L + wall.getNano() / 1000,
                europeanGap.toEpochSecond(java.time.ZoneOffset.UTC) * 1_000_000L + europeanGap.getNano() / 1000));
        assertEquals(List.of(LocalDateTime.ofInstant(before, java.time.ZoneOffset.UTC),
            LocalDateTime.ofInstant(after, java.time.ZoneOffset.UTC)), got.get("zoned").values(), zone);
        assertEquals(List.of(wall, europeanGap), got.get("unzoned").values(), zone);
      }
    } finally {
      java.util.TimeZone.setDefault(original);
    }
  }

  @Test
  void zoneFreeTimestampsRetainNullsMicrosecondsAndPreEpochWallFields() throws Exception {
    List<LocalDateTime> walls = Arrays.asList(
        LocalDateTime.parse("1969-12-31T23:59:59.999999"),
        LocalDateTime.parse("2024-06-02T00:00:01.123456"), null);
    Map<String, Read> got = answered(
        FakeJdbc.Column.of("unzoned", "TIMESTAMP_NTZ", walls.toArray()));
    assertEquals(walls, got.get("unzoned").values());
  }

  @Test
  void explicitTransferPreservesNullZeroAndNegativeMicroseconds() throws Exception {
    Map<String, Read> got = answeredWithConversions(
        List.of(Map.of("conversion", "timestamp_micros", "column", 0, "name", "at")),
        FakeJdbc.Column.of("at", "BIGINT", null, 0L, -1L, 1710037800123456L));
    assertEquals(Arrays.asList(null, LocalDateTime.parse("1970-01-01T00:00:00"),
        LocalDateTime.parse("1969-12-31T23:59:59.999999"),
        LocalDateTime.parse("2024-03-10T02:30:00.123456")), got.get("at").values());
  }

  @Test
  void aConversionMustMatchTheExactOutputPositionNameAndIntegerType() {
    List<Map<String, Object>> bad = List.of(
        Map.of("conversion", "timestamp_micros", "column", 1, "name", "at"),
        Map.of("conversion", "timestamp_micros", "column", 0, "name", "another"),
        Map.of("conversion", "not_supported", "column", 0, "name", "at"));
    for (Map<String, Object> conversion : bad) {
      assertThrows(java.sql.SQLException.class, () -> answeredWithConversions(
          List.of(conversion), FakeJdbc.Column.of("at", "BIGINT", 0L)));
    }
    assertThrows(java.sql.SQLException.class, () -> answeredWithConversions(
        List.of(Map.of("conversion", "timestamp_micros", "column", 0, "name", "at")),
        FakeJdbc.Column.of("at", "DOUBLE", 0.5)));
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
  void defaultComplexSupportPreservesNumericAndStringArraysBesideScalarColumns()
      throws Exception {
    Map<String, Read> got = answered(
        FakeJdbc.Column.of("id", "BIGINT", 1L, 2L, 3L),
        FakeJdbc.Column.of("whole", "ARRAY<BIGINT>", "[9007199254740993,null,-7]", null, "[]"),
        FakeJdbc.Column.of("single", "ARRAY<FLOAT>", "[1.25,null]", null, "[]"),
        FakeJdbc.Column.of("double", "ARRAY<DOUBLE>", "[1.25,2.5]", null, "[]"),
        FakeJdbc.Column.of("text", "ARRAY<STRING>",
            "[\"Zoë\",\"a,b\",\"quote\\\"\",null]", null, "[]"));
    assertEquals(List.of(1L, 2L, 3L), got.get("id").values());
    assertEquals(Arrays.asList(9007199254740993L, null, -7L),
        got.get("whole").values().get(0));
    assertEquals(Arrays.asList(1.25f, null), got.get("single").values().get(0));
    assertEquals(List.of(1.25, 2.5), got.get("double").values().get(0));
    assertEquals(Arrays.asList("Zoë", "a,b", "quote\"", null), utf8Items(got.get("text"), 0));
    for (String name : List.of("whole", "single", "double", "text")) {
      assertEquals("List", got.get(name).field().getType().toString());
      assertNull(got.get(name).values().get(1), name);
      assertEquals(List.of(), got.get(name).values().get(2), name);
    }
    assertEquals("Int(64, true)",
        got.get("whole").field().getChildren().get(0).getType().toString());
    assertEquals("FloatingPoint(SINGLE)",
        got.get("single").field().getChildren().get(0).getType().toString());
    assertEquals("FloatingPoint(DOUBLE)",
        got.get("double").field().getChildren().get(0).getType().toString());
    assertEquals("Utf8", got.get("text").field().getChildren().get(0).getType().toString());
  }

  @Test
  void enabledComplexSupportReadsNativeArraysWithoutSerializingTheirStrings() throws Exception {
    Map<String, Read> got = answered(
        FakeJdbc.Column.of("text", "ARRAY<STRING>",
            Arrays.asList("Zoë", "a,b", "quote\"", null), List.of(), null));
    assertEquals(Arrays.asList("Zoë", "a,b", "quote\"", null), utf8Items(got.get("text"), 0));
    assertEquals(List.of(), got.get("text").values().get(1));
    assertNull(got.get("text").values().get(2));
  }

  @Test
  void tinyAndDateArraysReadNativeDriverClassesUnderBothComplexModesAndJvmZones()
      throws Exception {
    java.util.TimeZone original = java.util.TimeZone.getDefault();
    try {
      for (String zone : List.of("UTC", "America/Los_Angeles", "Europe/Berlin")) {
        java.util.TimeZone.setDefault(java.util.TimeZone.getTimeZone(zone));
        List<Integer> dateDays = Arrays.asList(-1, 19792, 19813, null);
        for (boolean complexEnabled : List.of(false, true)) {
          Object tiny = complexEnabled ? Arrays.asList("-128", "127", null) : "[-128,127,null]";
          Object days = complexEnabled ? Arrays.asList(java.sql.Date.valueOf("1969-12-31"),
              java.sql.Date.valueOf("2024-03-10"), java.sql.Date.valueOf("2024-03-31"), null)
              : "[-1,19792,19813,null]";
          Object empty = complexEnabled ? List.of() : "[]";
          Map<String, Read> got = answered(
              FakeJdbc.Column.of("tiny", "ARRAY<TINYINT>", tiny, empty, null),
              FakeJdbc.Column.of("dates", "ARRAY<DATE>", days, empty, null));
          assertEquals(Arrays.asList((byte) -128, (byte) 127, null),
              got.get("tiny").values().get(0), zone);
          assertEquals(dateDays,
              assertInstanceOf(List.class, got.get("dates").values().get(0)), zone);
          assertEquals("Int(8, true)",
              got.get("tiny").field().getChildren().get(0).getType().toString());
          assertEquals("Date(DAY)",
              got.get("dates").field().getChildren().get(0).getType().toString());
          for (String name : List.of("tiny", "dates")) {
            assertEquals(List.of(), got.get(name).values().get(1), zone);
            assertNull(got.get(name).values().get(2), zone);
          }
        }
      }
    } finally {
      java.util.TimeZone.setDefault(original);
    }
  }

  @Test
  void tinyArrayOutOfRangeAndUnknownDateShapesAreNamedDataFailures() {
    for (String number : List.of("-129", "128")) {
      java.sql.SQLException refused = assertThrows(java.sql.SQLDataException.class,
          () -> answered(FakeJdbc.Column.of("tiny", "ARRAY<TINYINT>", List.of(number))));
      assertEquals("22003", refused.getSQLState());
      assertTrue(refused.getMessage().contains("TINYINT"));
    }
    java.sql.SQLException refused = assertThrows(java.sql.SQLDataException.class,
        () -> answered(FakeJdbc.Column.of("dates", "ARRAY<DATE>", List.of(7L))));
    assertEquals("22000", refused.getSQLState());
    assertTrue(refused.getMessage().contains("java.lang.Long"));
  }

  @Test
  void anUnexpectedArrayValueIsRefusedByItsJdbcType() {
    java.sql.SQLException refused = assertThrows(java.sql.SQLDataException.class,
        () -> answered(FakeJdbc.Column.of("broken", "ARRAY<BIGINT>", 42L)));
    assertEquals("22000", refused.getSQLState());
    assertTrue(refused.getMessage().contains("java.lang.Long"));
  }

  @Test
  void otherJdbcBackendsLoadWithoutTheDatabricksDriver() throws Exception {
    String separator = java.io.File.pathSeparator;
    String testClasspath = System.getProperty(
        "surefire.test.class.path", System.getProperty("java.class.path"));
    String classpath = Arrays.stream(testClasspath.split(java.util.regex.Pattern.quote(separator)))
        .filter(entry -> !java.nio.file.Path.of(entry).getFileName().toString()
            .startsWith("databricks-jdbc-"))
        .collect(java.util.stream.Collectors.joining(separator));
    Process child = new ProcessBuilder(
        java.nio.file.Path.of(System.getProperty("java.home"), "bin", "java").toString(),
        "-cp", classpath, WithoutDatabricksDriver.class.getName())
        .redirectErrorStream(true).start();
    boolean finished = child.waitFor(30, java.util.concurrent.TimeUnit.SECONDS);
    if (!finished) {
      child.destroyForcibly();
    }
    assertTrue(finished, "the JDBC classpath control timed out");
    String output = new String(child.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    assertEquals(0, child.exitValue(), output);
  }

  /** Runs without the optional Databricks driver, using the other providers' shared converter. */
  public static final class WithoutDatabricksDriver {
    private WithoutDatabricksDriver() {}

    public static void main(String[] arguments) throws Exception {
      try {
        Class.forName("com.databricks.jdbc.api.impl.ComplexDataTypeParser");
        throw new AssertionError("the Databricks driver must be absent");
      } catch (ClassNotFoundException expected) {
        // The classpath deliberately excludes the provided driver.
      }
      assertEquals("Int(64, true)", ConversionTable.REVIEWED.postgres("int8", 0, 0)
          .field("id").getType().toString());
      try (java.sql.Connection sqlite = java.sql.DriverManager.getConnection("jdbc:sqlite::memory:");
          java.sql.Statement statement = sqlite.createStatement();
          java.sql.ResultSet rows = statement.executeQuery("SELECT 7 AS id")) {
        new JdbcProvider(sqlite, JdbcProvider.Driver.SQLITE);
        assertEquals(List.of(List.of(7L)), ConversionTable.sqlite(rows).columns());
      }
      try (java.sql.Connection duck = java.sql.DriverManager.getConnection("jdbc:duckdb:")) {
        new JdbcProvider(duck, JdbcProvider.Driver.DUCKDB);
      }
      new JdbcProvider(new FakeJdbc.Database(text -> List.of()).connection(),
          JdbcProvider.Driver.POSTGRES);
    }
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
