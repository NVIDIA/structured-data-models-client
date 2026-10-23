// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import com.nvidia.sdm.internal.PythonText;
import java.math.BigDecimal;
import java.sql.Array;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.OffsetTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;

/**
 * The JDBC-to-Arrow conversion table (plan section 3: "JDBC types: host, as a reviewed table"):
 * for each driver, the Arrow type each of its columns reaches the core as, and how each value is
 * read, so the same table reaches the core as the same Arrow through Java's driver as through
 * Python's (L5 for Java). How the reference reads a value is the core's ({@code sdm-data}'s value
 * rules); this table only moves the driver's values into Arrow, as Python's providers do.
 *
 * <table>
 *   <caption>The table, by driver</caption>
 *   <tr><th>Driver</th><th>Column</th><th>Arrow</th></tr>
 *   <tr><td>DuckDB</td><td>every type</td><td>the driver's own Arrow export ({@code
 *       arrowExportStream}), the same C++ conversion as Python's {@code to_arrow_table}</td></tr>
 *   <tr><td>SQLite</td><td>by the values of the first {@value #SQLITE_BATCH} rows, as the
 *       reference's driver (ADBC SQLite) infers it</td><td>a {@code BLOB} among them: binary;
 *       else {@code TEXT}: utf8; else {@code REAL}: float64; else (integers, nulls, no row):
 *       int64</td></tr>
 *   <tr><td>Postgres</td><td>by the column's type name ({@link #POSTGRES})</td><td>as Python's
 *       Postgres provider types it from psycopg's registry; a name it does not hold, utf8</td></tr>
 * </table>
 *
 * <p>An entry can be replaced ({@link #withPostgres}): L5 for Java seeds a unit or scale change
 * in one driver's path that way, and fails. Not part of the API.
 */
public final class ConversionTable {
  /** The rows ADBC SQLite reads before it settles a column's type. */
  public static final int SQLITE_BATCH = 1024;

  /** The field metadata marking a column of JSON text (Arrow's {@code arrow.json}). */
  public static final Map<String, String> JSON_TAG =
      Map.of("ARROW:extension:name", "arrow.json", "ARROW:extension:metadata", "");

  /** The field metadata marking an interval built as the reference's driver builds one. */
  public static final Map<String, String> INTERVAL_TAG = Map.of("sdm:interval", "timedelta");

  /** The field metadata key of a column of exact numbers as their text, its key unknown. */
  public static final String EXACT_NUMBER_MARKER = "sdm:exact_number";

  /** The field metadata key naming the declared scale of numbers past Arrow's decimals. */
  public static final String DECLARED_SCALE_MARKER = "sdm:declared_scale";

  /** How one Postgres column is read: its field, and each value's reading. */
  public interface Column {
    Field field(String name);

    Object read(ResultSet rows, int at) throws SQLException;
  }

  /** How a Postgres type is read, given the column's declared precision and scale. */
  @FunctionalInterface
  public interface PostgresType {
    Column column(int precision, int scale);
  }

  /** The reviewed table. */
  public static final ConversionTable REVIEWED = new ConversionTable(postgres());

  private final Map<String, PostgresType> postgres;

  private ConversionTable(Map<String, PostgresType> postgres) {
    this.postgres = Map.copyOf(postgres);
  }

  /** This table with the Postgres type {@code name} read by {@code type} instead. */
  public ConversionTable withPostgres(String name, PostgresType type) {
    Map<String, PostgresType> changed = new LinkedHashMap<>(postgres);
    changed.put(name, type);
    return new ConversionTable(changed);
  }

  // --- SQLite --------------------------------------------------------------------------

  /** One SQLite value, as its storage class gave it: SQLite's text of a real kept beside it. */
  private record Real(double value, String text) {}

  /**
   * The SQLite column: a result read as the reference's driver (ADBC SQLite) reads it. Each
   * column's type is settled by its values in the first {@link #SQLITE_BATCH} rows: a {@code
   * BLOB} among them makes it binary, else {@code TEXT} utf8, else {@code REAL} float64, else
   * int64 (integers, nulls, or no row at all). Those rows are written in that type: an integer as
   * its digits, a real in C's {@code %e} form ({@code 1.500000e+00}) in a utf8 column, any value's
   * text as its bytes in a binary one, an integer as a double in a float64 one. A later row must
   * fit: an integer or a real in a float64 column, any value but a {@code BLOB} in a utf8 one (a
   * real as SQLite's own text of it), and in an int64 or binary column only its own kind; any
   * other fails the read, as ADBC's does.
   *
   * @return the columns' fields, then each column's values
   */
  static Result sqlite(ResultSet rows) throws SQLException {
    ResultSetMetaData meta = rows.getMetaData();
    int width = meta.getColumnCount();
    List<List<Object>> read = new ArrayList<>();
    for (int at = 0; at < width; at++) {
      read.add(new ArrayList<>());
    }
    while (rows.next()) {
      for (int at = 0; at < width; at++) {
        Object value = rows.getObject(at + 1);
        if (value instanceof Double real) {
          value = new Real(real, rows.getString(at + 1));
        } else if (value instanceof Integer small) {
          value = small.longValue();
        }
        read.get(at).add(value);
      }
    }
    List<Field> fields = new ArrayList<>();
    List<List<Object>> columns = new ArrayList<>();
    for (int at = 0; at < width; at++) {
      List<Object> values = read.get(at);
      ArrowType type = sqliteType(values.subList(0, Math.min(values.size(), SQLITE_BATCH)));
      fields.add(ArrowColumns.field(meta.getColumnLabel(at + 1), type, null));
      List<Object> written = new ArrayList<>(values.size());
      for (int row = 0; row < values.size(); row++) {
        written.add(sqliteValue(values.get(row), type, row < SQLITE_BATCH));
      }
      columns.add(written);
    }
    return new Result(fields, columns);
  }

  /** A result read from rows: its fields, then each column's values. */
  record Result(List<Field> fields, List<List<Object>> columns) {}

  private static ArrowType sqliteType(List<Object> first) {
    boolean blob = false;
    boolean text = false;
    boolean real = false;
    for (Object value : first) {
      blob |= value instanceof byte[];
      text |= value instanceof String;
      real |= value instanceof Real;
    }
    return blob
        ? ArrowColumns.BINARY
        : text ? ArrowColumns.UTF8 : real ? ArrowColumns.FLOAT64 : ArrowColumns.INT64;
  }

  private static Object sqliteValue(Object value, ArrowType type, boolean first)
      throws SQLException {
    if (value == null) {
      return null;
    }
    if (type == ArrowColumns.INT64) {
      if (value instanceof Long) {
        return value;
      }
    } else if (type == ArrowColumns.FLOAT64) {
      if (value instanceof Real real) {
        return real.value();
      }
      if (value instanceof Long whole) {
        return whole.doubleValue();
      }
    } else if (type == ArrowColumns.UTF8) {
      if (value instanceof String text) {
        return text;
      }
      if (value instanceof Long whole) {
        return whole.toString();
      }
      if (value instanceof Real real) {
        return first ? cExponent(real.value()) : real.text();
      }
    } else if (type == ArrowColumns.BINARY) {
      if (value instanceof byte[] bytes) {
        return bytes;
      }
      if (first) {
        String text =
            value instanceof Real real ? cExponent(real.value()) : String.valueOf(value);
        return text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
      }
    }
    throw new SQLException("[SQLite] Type mismatch: a value of another type than its column's");
  }

  /** A double in C's {@code %e} form, as ADBC SQLite writes a real into a text column. */
  static String cExponent(double value) {
    if (Double.isInfinite(value)) {
      return value > 0 ? "inf" : "-inf";
    }
    return String.format(java.util.Locale.ROOT, "%e", value);
  }

  // --- Postgres ------------------------------------------------------------------------

  /**
   * The Postgres column, by type name, as the Python provider reads each through psycopg
   * ({@code _rust/providers/postgres.py}): its {@code _SCALARS}, {@code numeric}, the times, the
   * interval, JSON, and every type psycopg reads as text; an array ({@code _<element>}) as a list
   * of its element's type ({@link #element}).
   */
  public static final Map<String, String> POSTGRES = Map.ofEntries(
      Map.entry("bool", "bool"),
      Map.entry("int2", "int16"),
      Map.entry("int4", "int32"),
      Map.entry("int8", "int64"),
      Map.entry("oid", "int64"),
      Map.entry("float4", "double (the server's shortest text of the value)"),
      Map.entry("float8", "double"),
      Map.entry("bytea", "binary"),
      Map.entry("date", "date32"),
      Map.entry("numeric", "decimal128(p, s) to 38 digits, decimal256(p, s) to 76, else exact"
          + " text marked sdm:exact_number (no precision, a negative scale) or"
          + " sdm:declared_scale (past 76 digits)"),
      Map.entry("timestamp", "timestamp[us]"),
      Map.entry("timestamptz", "timestamp[us, tz=UTC], the instant"),
      Map.entry("time", "time64[us]"),
      Map.entry("timetz", "time64[us], its offset left out"),
      Map.entry("interval", "month_day_nano_interval, a month as 30 days, tagged sdm:interval"),
      Map.entry("json", "utf8 tagged arrow.json: the JSON text of the value read"),
      Map.entry("jsonb", "utf8 tagged arrow.json: the JSON text of the value read"));

  /** The Postgres types read as text, as psycopg gives them (its {@code _TEXT}). */
  static final Set<String> TEXT = Set.of(
      "text", "varchar", "bpchar", "char", "name", "uuid", "xml", "inet", "cidr", "macaddr",
      "macaddr8", "money", "bit", "varbit");

  private static Map<String, PostgresType> postgres() {
    Map<String, PostgresType> types = new LinkedHashMap<>();
    types.put("bool", (p, s) -> plain(ArrowColumns.BOOL, (r, i) -> nullable(r, r.getBoolean(i))));
    types.put("int2", (p, s) -> plain(ArrowColumns.INT16, (r, i) -> nullable(r, r.getShort(i))));
    types.put("int4", (p, s) -> plain(ArrowColumns.INT32, (r, i) -> nullable(r, r.getInt(i))));
    types.put("int8", (p, s) -> plain(ArrowColumns.INT64, (r, i) -> nullable(r, r.getLong(i))));
    types.put("oid", (p, s) -> plain(ArrowColumns.INT64, (r, i) -> nullable(r, r.getLong(i))));
    // Read as the reference reads it, a double of the value the server writes (its shortest
    // text), never the float32 the server holds.
    types.put("float4", (p, s) -> plain(ArrowColumns.FLOAT64, ConversionTable::doubleOfText));
    types.put("float8", (p, s) -> plain(ArrowColumns.FLOAT64, ConversionTable::doubleOfText));
    types.put("bytea", (p, s) -> plain(ArrowColumns.BINARY, (r, i) -> r.getBytes(i)));
    types.put("date", (p, s) -> plain(ArrowColumns.DATE32, ConversionTable::days));
    types.put("numeric", ConversionTable::numeric);
    types.put("timestamp", (p, s) -> plain(ArrowColumns.TIMESTAMP_US, ConversionTable::wallMicros));
    types.put(
        "timestamptz", (p, s) -> plain(ArrowColumns.TIMESTAMP_US_UTC, ConversionTable::instantMicros));
    types.put("time", (p, s) -> plain(ArrowColumns.TIME64_US, ConversionTable::timeMicros));
    types.put("timetz", (p, s) -> plain(ArrowColumns.TIME64_US, ConversionTable::zonedTimeMicros));
    types.put(
        "interval",
        (p, s) -> tagged(ArrowColumns.MONTH_DAY_NANO, INTERVAL_TAG, ConversionTable::interval));
    types.put("json", (p, s) -> tagged(ArrowColumns.UTF8, JSON_TAG, ConversionTable::jsonText));
    types.put("jsonb", (p, s) -> tagged(ArrowColumns.UTF8, JSON_TAG, ConversionTable::jsonText));
    return types;
  }

  /** The Postgres column of type {@code name} with {@code precision} and {@code scale}. */
  public Column postgres(String name, int precision, int scale) {
    PostgresType type = postgres.get(name);
    if (type != null) {
      return type.column(precision, scale);
    }
    if (name.startsWith("_")) {
      return array(name.substring(1));
    }
    // A type psycopg reads as text (TEXT), or one its registry does not name (a user's enum or
    // composite), which it reads as text too.
    return plain(ArrowColumns.UTF8, (r, i) -> r.getString(i));
  }

  /** A reading of one value. */
  @FunctionalInterface
  interface Reading {
    Object read(ResultSet rows, int at) throws SQLException;
  }

  private static Column plain(ArrowType type, Reading reading) {
    return tagged(type, null, reading);
  }

  private static Column tagged(ArrowType type, Map<String, String> metadata, Reading reading) {
    return new Column() {
      @Override
      public Field field(String name) {
        return ArrowColumns.field(name, type, metadata);
      }

      @Override
      public Object read(ResultSet rows, int at) throws SQLException {
        return reading.read(rows, at);
      }
    };
  }

  private static Object nullable(ResultSet rows, Object value) throws SQLException {
    return rows.wasNull() ? null : value;
  }

  private static Object doubleOfText(ResultSet rows, int at) throws SQLException {
    String text = rows.getString(at);
    return text == null ? null : parseDouble(text);
  }

  private static Double parseDouble(String text) {
    return switch (text) {
      case "NaN" -> Double.NaN;
      case "Infinity" -> Double.POSITIVE_INFINITY;
      case "-Infinity" -> Double.NEGATIVE_INFINITY;
      default -> Double.valueOf(text);
    };
  }

  private static Object days(ResultSet rows, int at) throws SQLException {
    LocalDate date = rows.getObject(at, LocalDate.class);
    return date == null ? null : (int) date.toEpochDay();
  }

  private static Object wallMicros(ResultSet rows, int at) throws SQLException {
    LocalDateTime time = rows.getObject(at, LocalDateTime.class);
    return time == null ? null : micros(time.toInstant(ZoneOffset.UTC));
  }

  private static Object instantMicros(ResultSet rows, int at) throws SQLException {
    OffsetDateTime time = rows.getObject(at, OffsetDateTime.class);
    return time == null ? null : micros(time.toInstant());
  }

  private static long micros(java.time.Instant instant) {
    return Math.addExact(
        Math.multiplyExact(instant.getEpochSecond(), 1_000_000L), instant.getNano() / 1_000L);
  }

  private static Object timeMicros(ResultSet rows, int at) throws SQLException {
    LocalTime time = rows.getObject(at, LocalTime.class);
    return time == null ? null : time.toNanoOfDay() / 1_000L;
  }

  /** A {@code timetz}'s time of day, its offset left out, as the reference reads a time. */
  private static Object zonedTimeMicros(ResultSet rows, int at) throws SQLException {
    OffsetTime time = rows.getObject(at, OffsetTime.class);
    return time == null ? null : time.toLocalTime().toNanoOfDay() / 1_000L;
  }

  /**
   * An interval as the reference's driver reads it, a {@code timedelta} with a month counted as
   * 30 days and a year as 365 (psycopg's), then as Arrow's {@code MonthDayNano} of that
   * timedelta: no months, its days, and its seconds and microseconds as nanoseconds, the days
   * taking the sign as a timedelta normalizes.
   */
  private static Object interval(ResultSet rows, int at) throws SQLException {
    String text = rows.getString(at);
    if (text == null) {
      return null;
    }
    long micros = PostgresText.intervalMicros(text);
    long days = Math.floorDiv(micros, 86_400_000_000L);
    long rest = Math.floorMod(micros, 86_400_000_000L);
    return new ArrowColumns.MonthDayNano(0, Math.toIntExact(days), rest * 1_000L);
  }

  /** A JSON value as the JSON text of the value the reference's driver read. */
  private static Object jsonText(ResultSet rows, int at) throws SQLException {
    String text = rows.getString(at);
    return text == null ? null : PythonText.json(text);
  }

  /**
   * A {@code numeric} column: a decimal of its declaration where Arrow's decimals hold it, a
   * NaN missing; otherwise each value's exact text, its declaration marked: declared without a
   * precision, or with a negative scale, {@link #EXACT_NUMBER_MARKER}; declared past the 76
   * digits Arrow's widest decimal holds, {@link #DECLARED_SCALE_MARKER} with its scale.
   */
  private static Column numeric(int precision, int scale) {
    if (precision <= 0 || scale < 0 || precision > 76) {
      Map<String, String> marked =
          precision <= 0 || scale < 0
              ? Map.of(EXACT_NUMBER_MARKER, "true")
              : Map.of(DECLARED_SCALE_MARKER, Integer.toString(scale));
      return tagged(ArrowColumns.UTF8, marked, ConversionTable::numericText);
    }
    return plain(
        ArrowColumns.decimal(precision, scale),
        (r, i) -> {
          String text = r.getString(i);
          return text == null || "NaN".equals(text) ? null : new BigDecimal(text);
        });
  }

  /** A numeric's exact text, as Python's {@code str} of the driver's {@code Decimal} writes it. */
  private static Object numericText(ResultSet rows, int at) throws SQLException {
    String text = rows.getString(at);
    if (text == null) {
      return null;
    }
    return switch (text) {
      case "NaN", "Infinity", "-Infinity" -> text;
      default -> new BigDecimal(text).toString();
    };
  }

  /**
   * An array column ({@code _<element>}): a list of its element's type, each element read as the
   * Python provider reads one ({@code _element_type}, {@code _cell}), a multi-dimensional array's
   * in row-major order; a JSON array's elements each the JSON text of its value, marked as JSON.
   */
  private static Column array(String element) {
    Field item;
    ElementReading reading;
    switch (element) {
      case "json", "jsonb" -> {
        item = ArrowColumns.field("item", ArrowColumns.UTF8, JSON_TAG);
        reading = (value, text) -> PythonText.json(text);
      }
      case "bool" -> {
        item = ArrowColumns.field("item", ArrowColumns.BOOL, null);
        reading = (value, text) -> value instanceof Boolean flag ? flag : "t".equals(text);
      }
      case "int2" -> {
        item = ArrowColumns.field("item", ArrowColumns.INT16, null);
        reading = (value, text) -> Short.valueOf(text);
      }
      case "int4" -> {
        item = ArrowColumns.field("item", ArrowColumns.INT32, null);
        reading = (value, text) -> Integer.valueOf(text);
      }
      case "int8", "oid" -> {
        item = ArrowColumns.field("item", ArrowColumns.INT64, null);
        reading = (value, text) -> Long.valueOf(text);
      }
      case "float4", "float8", "numeric" -> {
        // A numeric element, whose array carries no scale of its own, as a double, as the
        // reference reads one.
        item = ArrowColumns.field("item", ArrowColumns.FLOAT64, null);
        reading = (value, text) -> parseDouble(text);
      }
      case "bytea" -> {
        item = ArrowColumns.field("item", ArrowColumns.BINARY, null);
        reading = (value, text) -> value instanceof byte[] bytes ? bytes : text.getBytes(
            java.nio.charset.StandardCharsets.UTF_8);
      }
      case "date" -> {
        item = ArrowColumns.field("item", ArrowColumns.DATE32, null);
        reading = (value, text) -> (int) LocalDate.parse(text).toEpochDay();
      }
      case "timestamp" -> {
        item = ArrowColumns.field("item", ArrowColumns.TIMESTAMP_US, null);
        reading = (value, text) -> micros(PostgresText.timestamp(text).toInstant(ZoneOffset.UTC));
      }
      case "time" -> {
        item = ArrowColumns.field("item", ArrowColumns.TIME64_US, null);
        reading = (value, text) -> LocalTime.parse(text).toNanoOfDay() / 1_000L;
      }
      default -> {
        // Any other element as the driver's text of it.
        item = ArrowColumns.field("item", ArrowColumns.UTF8, null);
        reading = (value, text) -> text;
      }
    }
    Field elementField = item;
    ElementReading read = reading;
    return new Column() {
      @Override
      public Field field(String name) {
        return ArrowColumns.list(name, elementField);
      }

      @Override
      public Object read(ResultSet rows, int at) throws SQLException {
        Array array = rows.getArray(at);
        if (array == null) {
          return null;
        }
        List<Object> out = new ArrayList<>();
        collect(array, out, read);
        return out;
      }
    };
  }

  /** One element's reading, from the driver's object and its text of it. */
  @FunctionalInterface
  private interface ElementReading {
    Object read(Object value, String text);
  }

  /** {@code array}'s elements, a nested array's in row-major order, each read. */
  private static void collect(Array array, List<Object> out, ElementReading read)
      throws SQLException {
    try (ResultSet elements = array.getResultSet()) {
      while (elements.next()) {
        Object value = elements.getObject(2);
        if (value instanceof Array nested) {
          collect(nested, out, read);
          continue;
        }
        String text = elements.getString(2);
        out.add(value == null ? null : read.read(value, text));
      }
    } finally {
      array.free();
    }
  }

  /** The Postgres type names of a result's columns, and their precision and scale. */
  public static List<Object[]> described(ResultSetMetaData meta) throws SQLException {
    List<Object[]> columns = new ArrayList<>();
    for (int at = 1; at <= meta.getColumnCount(); at++) {
      int scale = meta.getScale(at);
      // pgjdbc reads a negative scale's 11-bit field as unsigned (numeric(5, -2): 2046).
      if (scale > 1000) {
        scale = ((scale + 1024) & 2047) - 1024;
      }
      columns.add(new Object[] {meta.getColumnTypeName(at), meta.getPrecision(at), scale});
    }
    return columns;
  }

}
