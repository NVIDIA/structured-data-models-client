// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import com.nvidia.sdm.internal.NativeCore;
import java.math.BigDecimal;
import java.sql.Array;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
 *   <tr><td>Databricks</td><td>by the column's type name ({@link #DATABRICKS})</td><td>as the
 *       Databricks SQL connector gives Python's provider its Arrow (C-1's recorded tables); a
 *       name it does not hold, utf8 of the driver's text</td></tr>
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

    Object read(ResultSet rows, int at, Session session) throws SQLException;
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
    throw new java.sql.SQLDataException(
        "[SQLite] Type mismatch: a value of another type than its column's");
  }

  /**
   * A double in C's {@code %e} form, as ADBC SQLite writes a real into a text column: the double's
   * exact value to seven significant digits, an exact tie to even (glibc's printf), the exponent
   * signed and of two digits at least.
   */
  static String cExponent(double value) {
    if (Double.isInfinite(value)) {
      return value > 0 ? "inf" : "-inf";
    }
    if (Double.isNaN(value)) {
      return Double.doubleToRawLongBits(value) < 0 ? "-nan" : "nan";
    }
    String sign = (Double.doubleToRawLongBits(value) < 0) ? "-" : "";
    if (value == 0) {
      return sign + "0.000000e+00";
    }
    BigDecimal rounded =
        new BigDecimal(Math.abs(value)).round(new java.math.MathContext(7, java.math.RoundingMode.HALF_EVEN));
    String digits = rounded.unscaledValue().toString();
    int exponent = digits.length() - 1 - rounded.scale();
    digits = (digits + "000000").substring(0, 7);
    return sign
        + digits.charAt(0)
        + "."
        + digits.substring(1)
        + "e"
        + (exponent < 0 ? "-" : "+")
        + (Math.abs(exponent) < 10 ? "0" : "")
        + Math.abs(exponent);
  }

  // --- Databricks ----------------------------------------------------------------------

  /**
   * The Databricks column, by the type name the Databricks JDBC driver gives a column ({@code
   * getColumnTypeName}, its parameters left out), as the Databricks SQL connector gives Python's
   * provider its Arrow ({@code fetchall_arrow}, C-1's recorded tables): each type at its own
   * width, a null a null whatever the getter reads. The driver's own types are seen only live
   * (B7); the proposed JDBC probe checks this column against them.
   */
  public static final Map<String, String> DATABRICKS = Map.ofEntries(
      Map.entry("TINYINT", "int8"),
      Map.entry("SMALLINT", "int16"),
      Map.entry("INT", "int32 (INTEGER too)"),
      Map.entry("BIGINT", "int64"),
      Map.entry("FLOAT", "float32"),
      Map.entry("DOUBLE", "float64, a NaN kept"),
      Map.entry("DECIMAL", "decimal128(p, s), the driver's precision and scale, exact"),
      Map.entry("STRING", "utf8 (VARCHAR and CHAR too)"),
      Map.entry("BOOLEAN", "bool"),
      Map.entry("DATE", "date32"),
      Map.entry("TIMESTAMP", "timestamp[us]: the instant, as its wall time in UTC"),
      Map.entry("TIMESTAMP_NTZ", "timestamp[us]: its wall time"),
      Map.entry("BINARY", "binary"),
      Map.entry("ARRAY", "list of its element's column, for an element of a type above but"
          + " DECIMAL and the time stamps; else utf8 of the driver's text"),
      Map.entry("VOID", "null"),
      Map.entry("any other type (STRUCT, MAP, INTERVAL, VARIANT)", "utf8: the driver's text"));

  /** How one Databricks column is read: its Arrow type and each value's reading. */
  private interface DatabricksReading {
    Object read(ResultSet rows, int at) throws SQLException;
  }

  private record DatabricksColumn(ArrowType type, ArrowType element, DatabricksReading reading) {
    Field field(String name) {
      return element == null
          ? ArrowColumns.field(name, type, null)
          : ArrowColumns.list(name, ArrowColumns.field("item", element, null));
    }
  }

  private static final java.util.Calendar UTC =
      java.util.Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"));

  /** A Databricks result read whole by the Databricks column ({@link #DATABRICKS}). */
  static Result databricks(ResultSet rows) throws SQLException {
    return databricks(rows, List.of());
  }

  /** Apply only the output conversions explicitly selected by the core. */
  static Result databricks(ResultSet rows, List<Map<String, Object>> conversions)
      throws SQLException {
    ResultSetMetaData meta = rows.getMetaData();
    int width = meta.getColumnCount();
    List<DatabricksColumn> columns = new ArrayList<>();
    List<Field> fields = new ArrayList<>();
    List<List<Object>> values = new ArrayList<>();
    for (int at = 1; at <= width; at++) {
      DatabricksColumn column =
          databricksColumn(meta.getColumnTypeName(at), meta.getPrecision(at), meta.getScale(at));
      columns.add(column);
      fields.add(column.field(meta.getColumnLabel(at)));
      values.add(new ArrayList<>());
    }
    java.util.Set<Integer> converted = new java.util.HashSet<>();
    for (Map<String, Object> conversion : conversions) {
      if (!"timestamp_micros".equals(conversion.get("conversion"))
          || !(conversion.get("column") instanceof Number position)) {
        throw new SQLException("an unsupported SQL result conversion");
      }
      int at = position.intValue();
      if (at < 0 || at >= width || !converted.add(at)
          || !meta.getColumnLabel(at + 1).equals(conversion.get("name"))
          || !java.util.Set.of("BIGINT", "LONG").contains(bare(meta.getColumnTypeName(at + 1)))) {
        throw new SQLException("a SQL result conversion does not match its output column");
      }
      DatabricksColumn column = new DatabricksColumn(ArrowColumns.TIMESTAMP_US, null,
          (result, index) -> {
            long micros = result.getLong(index);
            return result.wasNull() ? null : micros;
          });
      columns.set(at, column);
      fields.set(at, column.field(meta.getColumnLabel(at + 1)));
    }
    while (rows.next()) {
      for (int at = 0; at < width; at++) {
        values.get(at).add(columns.get(at).reading().read(rows, at + 1));
      }
    }
    return new Result(fields, values);
  }

  /** The bare name of a Databricks type: upper case, its parameters left out. */
  private static String bare(String type) {
    String name = type == null ? "" : type.strip().toUpperCase(java.util.Locale.ROOT);
    int cut = name.length();
    for (char c : new char[] {'(', '<', ' '}) {
      int found = name.indexOf(c);
      if (found >= 0) {
        cut = Math.min(cut, found);
      }
    }
    return name.substring(0, cut);
  }

  private static DatabricksColumn databricksColumn(String type, int precision, int scale) {
    String name = bare(type);
    ArrowType scalar = databricksScalar(name, precision, scale);
    if (scalar != null) {
      return new DatabricksColumn(scalar, null, (rows, at) -> databricksValue(name, scale, rows, at));
    }
    int open = type == null ? -1 : type.indexOf('<');
    int close = type == null ? -1 : type.lastIndexOf('>');
    if ("ARRAY".equals(name) && open >= 0 && close > open) {
      String element = bare(type.substring(open + 1, close));
      if (ARRAY_ELEMENTS.contains(element)) {
        return new DatabricksColumn(
            ArrowType.List.INSTANCE,
            databricksScalar(element, 0, 0),
            (rows, at) -> databricksArray(element, rows, at));
      }
    }
    if ("VOID".equals(name) || "NULL".equals(name)) {
      return new DatabricksColumn(ArrowColumns.NULL, null, (rows, at) -> null);
    }
    return new DatabricksColumn(ArrowColumns.UTF8, null, (rows, at) -> rows.getString(at));
  }

  /** The element types an array is read as a list of; any other array is the driver's text. */
  private static final java.util.Set<String> ARRAY_ELEMENTS = java.util.Set.of(
      "TINYINT", "BYTE", "SMALLINT", "SHORT", "INT", "INTEGER", "BIGINT", "LONG", "FLOAT", "REAL",
      "DOUBLE", "STRING", "VARCHAR", "CHAR", "BOOLEAN", "DATE", "BINARY");

  /** The Arrow type of a Databricks scalar type, or null for one that is not a scalar. */
  private static ArrowType databricksScalar(String name, int precision, int scale) {
    return switch (name) {
      case "TINYINT", "BYTE" -> ArrowColumns.INT8;
      case "SMALLINT", "SHORT" -> ArrowColumns.INT16;
      case "INT", "INTEGER" -> ArrowColumns.INT32;
      case "BIGINT", "LONG" -> ArrowColumns.INT64;
      case "FLOAT", "REAL" -> ArrowColumns.FLOAT32;
      case "DOUBLE" -> ArrowColumns.FLOAT64;
      case "DECIMAL", "DEC", "NUMERIC" -> ArrowColumns.decimal(precision, scale);
      case "STRING", "VARCHAR", "CHAR" -> ArrowColumns.UTF8;
      case "BOOLEAN" -> ArrowColumns.BOOL;
      case "DATE" -> ArrowColumns.DATE32;
      case "TIMESTAMP", "TIMESTAMP_LTZ", "TIMESTAMP_NTZ" -> ArrowColumns.TIMESTAMP_US;
      case "BINARY" -> ArrowColumns.BINARY;
      default -> null;
    };
  }

  /** One Databricks scalar, as the Java value its Arrow column holds; null for a null. */
  private static Object databricksValue(String name, int scale, ResultSet rows, int at)
      throws SQLException {
    Object value =
        switch (name) {
          case "TINYINT", "BYTE" -> rows.getByte(at);
          case "SMALLINT", "SHORT" -> rows.getShort(at);
          case "INT", "INTEGER" -> rows.getInt(at);
          case "BIGINT", "LONG" -> rows.getLong(at);
          case "FLOAT", "REAL" -> rows.getFloat(at);
          case "DOUBLE" -> rows.getDouble(at);
          case "DECIMAL", "DEC", "NUMERIC" -> {
            BigDecimal read = rows.getBigDecimal(at);
            yield read == null ? null : read.setScale(scale, java.math.RoundingMode.UNNECESSARY);
          }
          case "STRING", "VARCHAR", "CHAR" -> rows.getString(at);
          case "BOOLEAN" -> rows.getBoolean(at);
          case "DATE" -> {
            java.time.LocalDate read = rows.getObject(at, java.time.LocalDate.class);
            yield read == null ? null : (int) read.toEpochDay();
          }
          case "TIMESTAMP", "TIMESTAMP_LTZ" -> {
            java.sql.Timestamp read = rows.getTimestamp(at, (java.util.Calendar) UTC.clone());
            yield read == null
                ? null
                : Math.floorDiv(read.getTime(), 1000L) * 1_000_000L + read.getNanos() / 1_000L;
          }
          case "TIMESTAMP_NTZ" -> {
            // Databricks JDBC has no LocalDateTime typed getter. Its supported
            // calendar getter assigns the driver's wall fields to UTC, matching
            // Arrow's timezone-free microsecond representation without using
            // the JVM's offset to compute that representation.
            java.sql.Timestamp read = rows.getTimestamp(at, (java.util.Calendar) UTC.clone());
            yield read == null
                ? null
                : Math.floorDiv(read.getTime(), 1000L) * 1_000_000L + read.getNanos() / 1_000L;
          }
          case "BINARY" -> rows.getBytes(at);
          default -> throw new SQLException("no Databricks scalar of " + name + " is read here");
        };
    return rows.wasNull() ? null : value;
  }

  /** One Databricks array of scalars, each element as its Arrow column holds it. */
  private static Object databricksArray(String element, ResultSet rows, int at)
      throws SQLException {
    Object value = rows.getObject(at);
    if (value == null || rows.wasNull()) {
      return null;
    }
    java.sql.Array read = databricksNativeArray(element, value);
    try {
      Object[] items = (Object[]) read.getArray();
      List<Object> out = new ArrayList<>(items.length);
      for (Object item : items) {
        out.add(item == null ? null : databricksElement(element, item));
      }
      return out;
    } finally {
      read.free();
    }
  }

  /**
   * JDBC 3.4.2 returns native arrays with complex support enabled and JSON text by default.
   * Its public parser (under {@code api.impl}) is the same native conversion the driver uses
   * for inline results. Keep that version-specific dependency here: the other JDBC providers
   * load and run without the optional Databricks driver, and enabled arrays avoid the driver's
   * lossy string serialization of embedded quotes.
   */
  private static java.sql.Array databricksNativeArray(String element, Object value)
      throws SQLException {
    if (value instanceof java.sql.Array array) {
      return array;
    }
    if (value instanceof String text) {
      return new com.databricks.jdbc.api.impl.ComplexDataTypeParser()
          .parseJsonStringToDbArray(text, "ARRAY<" + element + ">");
    }
    throw new java.sql.SQLDataException(
        "Databricks ARRAY has unsupported JDBC value type: " + value.getClass().getName(), "22000");
  }

  private static Object databricksElement(String element, Object item) {
    return switch (element) {
      case "TINYINT", "BYTE" -> ((Number) item).byteValue();
      case "SMALLINT", "SHORT" -> ((Number) item).shortValue();
      case "INT", "INTEGER" -> ((Number) item).intValue();
      case "BIGINT", "LONG" -> ((Number) item).longValue();
      case "FLOAT", "REAL" -> ((Number) item).floatValue();
      case "DOUBLE" -> ((Number) item).doubleValue();
      case "BOOLEAN" -> (Boolean) item;
      case "DATE" -> (int) ((java.time.LocalDate) item).toEpochDay();
      case "BINARY" -> (byte[]) item;
      default -> item.toString();
    };
  }

  // --- Postgres ------------------------------------------------------------------------

  /**
   * The Postgres column, by the name psycopg's registry gives its type, as the Python provider
   * reads each through psycopg ({@code _rust/providers/postgres.py}), every value as the server's
   * text, as psycopg reads it: its {@code _SCALARS}, {@code numeric}, the times, the interval,
   * JSON; any other type its registry names as Python's {@code str} of what psycopg read (a range
   * as {@code [1, 5)}); an array of a type it names as a list of its element ({@link #array}); a
   * type it does not name (a user's enum or composite, or an array of one) as the server's text.
   */
  public static final Map<String, String> POSTGRES = Map.ofEntries(
      Map.entry("bool", "bool"),
      Map.entry("int2", "int16"),
      Map.entry("int4", "int32"),
      Map.entry("int8", "int64"),
      Map.entry("oid", "int64"),
      Map.entry("float4", "double of the server's text of the value, as psycopg reads it"),
      Map.entry("float8", "double"),
      Map.entry("bytea", "binary"),
      Map.entry("date", "date32; years 1 to 9999, else refused"),
      Map.entry("numeric", "decimal128(p, s) to 38 digits, decimal256(p, s) to 76, else exact"
          + " text marked sdm:exact_number (no precision, a negative scale) or"
          + " sdm:declared_scale (past 76 digits)"),
      Map.entry("timestamp", "timestamp[us]; years 1 to 9999, else refused"),
      Map.entry("timestamptz", "timestamp[us, tz=UTC], the instant; years 1 to 9999 in the"
          + " session's zone, else refused"),
      Map.entry("time", "time64[us]; 24:00:00 refused"),
      Map.entry("timetz", "time64[us], its offset left out; 24:00:00 refused"),
      Map.entry("interval", "month_day_nano_interval of psycopg's timedelta (a month as 30 days),"
          + " tagged sdm:interval; another IntervalStyle than postgres, or past a timedelta,"
          + " refused"),
      Map.entry("json", "utf8 tagged arrow.json: the core's JSON text of the value, a JSON null"
          + " null"),
      Map.entry("jsonb", "utf8 tagged arrow.json: the core's JSON text of the value, a JSON null"
          + " null"),
      Map.entry("int4range, int8range, numrange, daterange, tsrange, tstzrange", "utf8: Python's"
          + " str of psycopg's Range"),
      Map.entry("int4multirange and the other multiranges", "utf8: Python's str of psycopg's"
          + " Multirange"),
      Map.entry("record", "refused (register JV-2)"),
      Map.entry("any other type psycopg's registry names, or none", "utf8: the server's text"));

  /** A type psycopg's registry names: its name, and its array type's OID. */
  record Registered(String name, int array) {}

  /**
   * psycopg's registry of the server's own types (psycopg 3, its default adapters, by OID), as the
   * Python provider reads a column's type: a type by its OID, an array by its array OID.
   */
  static final Map<Integer, Registered> REGISTRY = registry(
      16, "bool", 1000, 17, "bytea", 1001, 18, "\"char\"", 1002, 19, "name", 1003,
      20, "int8", 1016, 21, "int2", 1005, 22, "int2vector", 1006, 23, "int4", 1007,
      24, "regproc", 1008, 25, "text", 1009, 26, "oid", 1028, 27, "tid", 1010, 28, "xid", 1011,
      29, "cid", 1012, 30, "oidvector", 1013, 114, "json", 199, 142, "xml", 143,
      600, "point", 1017, 601, "lseg", 1018, 602, "path", 1019, 603, "box", 1020,
      604, "polygon", 1027, 628, "line", 629, 650, "cidr", 651, 700, "float4", 1021,
      701, "float8", 1022, 718, "circle", 719, 774, "macaddr8", 775, 790, "money", 791,
      829, "macaddr", 1040, 869, "inet", 1041, 1033, "aclitem", 1034, 1042, "bpchar", 1014,
      1043, "varchar", 1015, 1082, "date", 1182, 1083, "time", 1183, 1114, "timestamp", 1115,
      1184, "timestamptz", 1185, 1186, "interval", 1187, 1266, "timetz", 1270,
      1560, "bit", 1561, 1562, "varbit", 1563, 1700, "numeric", 1231, 1790, "refcursor", 2201,
      2202, "regprocedure", 2207, 2203, "regoper", 2208, 2204, "regoperator", 2209,
      2205, "regclass", 2210, 2206, "regtype", 2211, 2249, "record", 2287, 2950, "uuid", 2951,
      2970, "txid_snapshot", 2949, 3220, "pg_lsn", 3221, 3614, "tsvector", 3643,
      3615, "tsquery", 3645, 3642, "gtsvector", 3644, 3734, "regconfig", 3735,
      3769, "regdictionary", 3770, 3802, "jsonb", 3807, 3904, "int4range", 3905,
      3906, "numrange", 3907, 3908, "tsrange", 3909, 3910, "tstzrange", 3911,
      3912, "daterange", 3913, 3926, "int8range", 3927, 4072, "jsonpath", 4073,
      4089, "regnamespace", 4090, 4096, "regrole", 4097, 4191, "regcollation", 4192,
      4451, "int4multirange", 6150, 4532, "nummultirange", 6151, 4533, "tsmultirange", 6152,
      4534, "tstzmultirange", 6153, 4535, "datemultirange", 6155, 4536, "int8multirange", 6157,
      5069, "xid8", 271);

  /** Each array type psycopg's registry names, by its OID: its element's name. */
  static final Map<Integer, String> ARRAYS = arrays();

  private static Map<Integer, Registered> registry(Object... given) {
    Map<Integer, Registered> registry = new LinkedHashMap<>();
    for (int at = 0; at < given.length; at += 3) {
      registry.put((Integer) given[at], new Registered((String) given[at + 1], (Integer) given[at + 2]));
    }
    return Map.copyOf(registry);
  }

  private static Map<Integer, String> arrays() {
    Map<Integer, String> arrays = new LinkedHashMap<>();
    for (Registered each : REGISTRY.values()) {
      arrays.put(each.array(), each.name());
    }
    return Map.copyOf(arrays);
  }

  /** The range types psycopg reads as a {@code Range}, by name: how each bound is written. */
  private static final Map<String, PostgresText.Bound> RANGES = Map.of(
      "int4range", ConversionTable::pyInteger,
      "int8range", ConversionTable::pyInteger,
      "numrange", ConversionTable::pyDecimal,
      "daterange", (text, session) -> PostgresText.pyDate(PostgresText.date(text)),
      "tsrange", (text, session) -> PostgresText.pyDatetime(PostgresText.timestamp(text)),
      "tstzrange", ConversionTable::pyZoned);

  /** The multirange types, by name: their range's name. */
  private static final Map<String, String> MULTIRANGES = Map.of(
      "int4multirange", "int4range",
      "int8multirange", "int8range",
      "nummultirange", "numrange",
      "datemultirange", "daterange",
      "tsmultirange", "tsrange",
      "tstzmultirange", "tstzrange");

  private static String pyInteger(String text, Session session) {
    return new java.math.BigInteger(text).toString();
  }

  /**
   * Python's {@code str} of a {@code Decimal} of the server's text: a NaN or an infinity as the
   * server spells it, as {@code Decimal} writes it back.
   */
  private static String pyDecimal(String text) {
    return switch (text) {
      case "NaN", "Infinity", "-Infinity" -> text;
      default -> new BigDecimal(text).toString();
    };
  }

  private static String pyDecimal(String text, Session session) {
    return pyDecimal(text);
  }

  /**
   * Python's {@code str} of the {@code datetime} psycopg reads a {@code timestamptz} as: the
   * instant the server wrote, in the zone psycopg resolves the session's {@code TimeZone} to
   * ({@link Session#zone}).
   */
  private static String pyZoned(String text, Session session) throws SQLException {
    PostgresText.Zoned zoned = PostgresText.zoned(text);
    long micros = zoned.micros();
    int offset = session.offsetSeconds(micros);
    java.time.LocalDateTime utc = wall(micros, 0);
    java.time.LocalDateTime wall = wall(micros, offset);
    if (!inPythonYears(utc) || !inPythonYears(wall)) {
      // psycopg moves the instant to UTC, then to the session's zone; where either leaves a
      // datetime's years (1 to 9999), it keeps the server's own wall time and offset.
      return PostgresText.pyDatetime(zoned.wall()) + PostgresText.pyOffset(zoned.offsetSeconds());
    }
    return PostgresText.pyDatetime(wall) + PostgresText.pyOffset(offset);
  }

  /** The wall time of the instant {@code micros} at {@code offset} seconds east of UTC. */
  private static java.time.LocalDateTime wall(long micros, int offset) {
    return java.time.LocalDateTime.ofEpochSecond(
        Math.floorDiv(micros, 1_000_000L) + offset,
        (int) Math.floorMod(micros, 1_000_000L) * 1_000,
        java.time.ZoneOffset.UTC);
  }

  /** Whether a Python {@code datetime} holds {@code wall}'s year. */
  private static boolean inPythonYears(java.time.LocalDateTime wall) {
    return wall.getYear() >= 1 && wall.getYear() <= 9999;
  }

  private static Map<String, PostgresType> postgres() {
    Map<String, PostgresType> types = new LinkedHashMap<>();
    types.put("bool", (p, s) -> plain(ArrowColumns.BOOL, ConversionTable::bool));
    types.put("int2", (p, s) -> plain(ArrowColumns.INT16, texts(Short::valueOf)));
    types.put("int4", (p, s) -> plain(ArrowColumns.INT32, texts(Integer::valueOf)));
    types.put("int8", (p, s) -> plain(ArrowColumns.INT64, texts(Long::valueOf)));
    types.put("oid", (p, s) -> plain(ArrowColumns.INT64, texts(Long::valueOf)));
    // Read as the reference reads it, a double of the value the server writes (its shortest
    // text), never the float32 the server holds.
    types.put("float4", (p, s) -> plain(ArrowColumns.FLOAT64, texts(ConversionTable::parseDouble)));
    types.put("float8", (p, s) -> plain(ArrowColumns.FLOAT64, texts(ConversionTable::parseDouble)));
    types.put("bytea", (p, s) -> plain(ArrowColumns.BINARY, (r, i, c) -> r.getBytes(i)));
    types.put("date", (p, s) -> plain(ArrowColumns.DATE32, texts(ConversionTable::days)));
    types.put("numeric", ConversionTable::numeric);
    types.put(
        "timestamp",
        (p, s) ->
            plain(
                ArrowColumns.TIMESTAMP_US,
                texts(t -> PostgresText.micros(PostgresText.timestamp(t).toInstant(ZoneOffset.UTC)))));
    types.put(
        "timestamptz",
        (p, s) -> plain(ArrowColumns.TIMESTAMP_US_UTC, texts(t -> PostgresText.zoned(t).micros())));
    types.put(
        "time",
        (p, s) -> plain(ArrowColumns.TIME64_US, texts(t -> PostgresText.time(t).toNanoOfDay() / 1_000L)));
    // A timetz's time of day, its offset left out, as the reference reads a time.
    types.put(
        "timetz",
        (p, s) ->
            plain(
                ArrowColumns.TIME64_US,
                texts(t -> PostgresText.zonedTime(t).time().toNanoOfDay() / 1_000L)));
    types.put(
        "interval",
        (p, s) ->
            tagged(
                ArrowColumns.MONTH_DAY_NANO,
                INTERVAL_TAG,
                (r, i, c) -> {
                  String text = r.getString(i);
                  return text == null ? null : monthDayNano(PostgresText.interval(text, c.intervalStyle()));
                }));
    types.put("json", (p, s) -> tagged(ArrowColumns.UTF8, JSON_TAG, texts(ConversionTable::jsonText)));
    types.put("jsonb", (p, s) -> tagged(ArrowColumns.UTF8, JSON_TAG, texts(ConversionTable::jsonText)));
    return types;
  }

  /**
   * What a reading asks of the session: its {@code IntervalStyle}, read once, when a value needs
   * it.
   */
  public interface Session {
    String intervalStyle() throws SQLException;

    /**
     * The UTC offset, in seconds, at {@code micros} in the zone psycopg reads a {@code
     * timestamptz} in: the session's {@code TimeZone} as Python's {@code ZoneInfo} finds it, in
     * Python's zone data, else UTC, as psycopg's {@code get_tzinfo} falls back ({@link
     * com.nvidia.sdm.internal.NativeCore#sessionZoneOffset}).
     */
    int offsetSeconds(long micros) throws SQLException;
  }

  /** The Postgres column of the type with OID {@code oid}, by psycopg's registry. */
  public Column postgres(int oid, int precision, int scale) {
    Registered registered = REGISTRY.get(oid);
    if (registered != null) {
      return postgres(registered.name(), precision, scale);
    }
    String element = ARRAYS.get(oid);
    if (element != null) {
      return array(element);
    }
    // A type psycopg's registry does not name (a user's enum or composite, or an array of one),
    // which it reads as the server's text.
    return plain(ArrowColumns.UTF8, (r, i, c) -> r.getString(i));
  }

  /** The Postgres column of the type psycopg's registry names {@code name}. */
  public Column postgres(String name, int precision, int scale) {
    PostgresType type = postgres.get(name);
    if (type != null) {
      return type.column(precision, scale);
    }
    SessionText written = textOf(name);
    return plain(
        ArrowColumns.UTF8,
        (rows, at, session) -> {
          String given = rows.getString(at);
          return given == null ? null : written.read(given, session);
        });
  }

  /**
   * How a value of a type psycopg's registry names, and this table does not read into a type of
   * its own, is written: Python's {@code str} of what psycopg read.
   */
  private static SessionText textOf(String name) {
    PostgresText.Bound range = RANGES.get(name);
    if (range != null) {
      return (text, session) -> PostgresText.pyRange(text, range, session);
    }
    String multirange = MULTIRANGES.get(name);
    if (multirange != null) {
      PostgresText.Bound bound = RANGES.get(multirange);
      return (text, session) -> PostgresText.pyMultirange(text, bound, session);
    }
    if ("record".equals(name)) {
      // psycopg reads an anonymous record, which no table's column holds, as a tuple of texts,
      // and the provider writes Python's repr of it; refused here (register JV-2).
      return (text, session) -> {
        throw PostgresText.refused("an anonymous record");
      };
    }
    return (text, session) -> text;
  }

  /** A reading of one value's text in the session it is read in. */
  @FunctionalInterface
  interface SessionText {
    Object read(String text, Session session) throws SQLException;
  }

  /** A reading of one value, given the session it is read in. */
  @FunctionalInterface
  interface Reading {
    Object read(ResultSet rows, int at, Session session) throws SQLException;
  }

  /** A reading of one value's text, the server's, never null. */
  @FunctionalInterface
  interface Text {
    Object read(String text) throws SQLException;
  }

  /** The reading of each value's text by {@code text}, a null null. */
  private static Reading texts(Text text) {
    return (rows, at, session) -> {
      String given = rows.getString(at);
      return given == null ? null : text.read(given);
    };
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
      public Object read(ResultSet rows, int at, Session session) throws SQLException {
        return reading.read(rows, at, session);
      }
    };
  }

  private static Object bool(ResultSet rows, int at, Session session) throws SQLException {
    String text = rows.getString(at);
    return text == null ? null : "t".equals(text) || "true".equals(text);
  }

  private static Double parseDouble(String text) {
    return switch (text) {
      case "NaN" -> Double.NaN;
      case "Infinity" -> Double.POSITIVE_INFINITY;
      case "-Infinity" -> Double.NEGATIVE_INFINITY;
      default -> Double.valueOf(text);
    };
  }

  private static Object days(String text) throws SQLException {
    return (int) PostgresText.date(text).toEpochDay();
  }

  /** psycopg's timedelta of an interval, as Arrow's {@code MonthDayNano} of it: no months. */
  private static ArrowColumns.MonthDayNano monthDayNano(PostgresText.Delta delta) {
    return new ArrowColumns.MonthDayNano(0, delta.days(), delta.micros() * 1_000L);
  }

  /**
   * A JSON value as the reference's provider hands it over, the core's text of it ({@link
   * NativeCore#jsonText}), a JSON null null; one the core does not read (nested past its 2,000
   * levels; the server writes no other) as the server's text, which the core then refuses by name
   * as it refuses Python's (register JV-3).
   */
  private static Object jsonText(String text) {
    try {
      return NativeCore.jsonText(text);
    } catch (IllegalArgumentException e) {
      return text;
    }
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
      return tagged(ArrowColumns.UTF8, marked, texts(ConversionTable::numericText));
    }
    return plain(
        ArrowColumns.decimal(precision, scale),
        texts(text -> "NaN".equals(text) ? null : new BigDecimal(text)));
  }

  /** A numeric's exact text, as Python's {@code str} of the driver's {@code Decimal} writes it. */
  private static Object numericText(String text) {
    return switch (text) {
      case "NaN", "Infinity", "-Infinity" -> text;
      default -> pyDecimal(text);
    };
  }

  /**
   * An array column of a type psycopg's registry names ({@code element}): a list of its element's
   * type, each element read as the Python provider reads one ({@code _element_type}, {@code
   * _cell}), a multi-dimensional array's in row-major order: an element of its {@code
   * _SCALARS} in that type; a {@code numeric} as a double; a {@code timestamp} and a {@code time}
   * in theirs; a JSON element as the core's text of it, marked as JSON; any other as Python's
   * {@code str} of what psycopg read (a {@code timestamptz} in the session's zone as {@code
   * 2024-01-01 12:00:00+05:30}, an {@code interval} as {@code 1 day, 2:00:00}, a {@code timetz}
   * as {@code 12:00:00+02:00}, a range as {@code [1, 5)}).
   */
  private static Column array(String element) {
    Field item;
    ElementReading reading;
    switch (element) {
      case "json", "jsonb" -> {
        item = ArrowColumns.field("item", ArrowColumns.UTF8, JSON_TAG);
        reading = (value, text, session) -> jsonText(text);
      }
      case "bool" -> {
        item = ArrowColumns.field("item", ArrowColumns.BOOL, null);
        reading = (value, text, session) -> "t".equals(text) || "true".equals(text);
      }
      case "int2" -> {
        item = ArrowColumns.field("item", ArrowColumns.INT16, null);
        reading = (value, text, session) -> Short.valueOf(text);
      }
      case "int4" -> {
        item = ArrowColumns.field("item", ArrowColumns.INT32, null);
        reading = (value, text, session) -> Integer.valueOf(text);
      }
      case "int8", "oid" -> {
        item = ArrowColumns.field("item", ArrowColumns.INT64, null);
        reading = (value, text, session) -> Long.valueOf(text);
      }
      case "float4", "float8", "numeric" -> {
        // A numeric element, whose array carries no scale of its own, as a double, as the
        // reference reads one.
        item = ArrowColumns.field("item", ArrowColumns.FLOAT64, null);
        reading = (value, text, session) -> parseDouble(text);
      }
      case "bytea" -> {
        item = ArrowColumns.field("item", ArrowColumns.BINARY, null);
        reading =
            (value, text, session) ->
                value instanceof byte[] bytes
                    ? bytes
                    : text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
      }
      case "date" -> {
        item = ArrowColumns.field("item", ArrowColumns.DATE32, null);
        reading = (value, text, session) -> days(text);
      }
      case "timestamp" -> {
        item = ArrowColumns.field("item", ArrowColumns.TIMESTAMP_US, null);
        reading =
            (value, text, session) ->
                PostgresText.micros(PostgresText.timestamp(text).toInstant(ZoneOffset.UTC));
      }
      case "time" -> {
        item = ArrowColumns.field("item", ArrowColumns.TIME64_US, null);
        reading = (value, text, session) -> PostgresText.time(text).toNanoOfDay() / 1_000L;
      }
      case "timestamptz" -> {
        item = ArrowColumns.field("item", ArrowColumns.UTF8, null);
        reading = (value, text, session) -> pyZoned(text, session);
      }
      case "timetz" -> {
        item = ArrowColumns.field("item", ArrowColumns.UTF8, null);
        reading =
            (value, text, session) -> {
              PostgresText.ZonedTime read = PostgresText.zonedTime(text);
              return PostgresText.pyTime(read.time()) + PostgresText.pyOffset(read.offsetSeconds());
            };
      }
      case "interval" -> {
        item = ArrowColumns.field("item", ArrowColumns.UTF8, null);
        reading =
            (value, text, session) ->
                PostgresText.pyTimedelta(PostgresText.interval(text, session.intervalStyle()));
      }
      default -> {
        // Any other element as Python's str of what psycopg read.
        item = ArrowColumns.field("item", ArrowColumns.UTF8, null);
        SessionText written = textOf(element);
        reading = (value, text, session) -> written.read(text, session);
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
      public Object read(ResultSet rows, int at, Session session) throws SQLException {
        Array array = rows.getArray(at);
        if (array == null) {
          return null;
        }
        List<Object> out = new ArrayList<>();
        collect(array, out, read, session);
        return out;
      }
    };
  }

  /** One element's reading, from the driver's object and the server's text of it. */
  @FunctionalInterface
  private interface ElementReading {
    Object read(Object value, String text, Session session) throws SQLException;
  }

  /** {@code array}'s elements, a nested array's in row-major order, each read. */
  private static void collect(Array array, List<Object> out, ElementReading read, Session session)
      throws SQLException {
    try (ResultSet elements = array.getResultSet()) {
      while (elements.next()) {
        Object value = elements.getObject(2);
        if (value instanceof Array nested) {
          collect(nested, out, read, session);
          continue;
        }
        String text = elements.getString(2);
        out.add(value == null ? null : read.read(value, text, session));
      }
    } finally {
      array.free();
    }
  }

  /** A result's column as the driver describes it: its type's OID, precision and scale. */
  public record Described(int oid, int precision, int scale) {}

  /**
   * The columns of a Postgres result: each type's OID, as the driver names the type (its name
   * read back to the OID the server sent), and its precision and scale.
   */
  public static List<Described> described(java.sql.Connection connection, ResultSetMetaData meta)
      throws SQLException {
    List<Described> columns = new ArrayList<>();
    for (int at = 1; at <= meta.getColumnCount(); at++) {
      int scale = meta.getScale(at);
      // pgjdbc reads a negative scale's 11-bit field as unsigned (numeric(5, -2): 2046).
      if (scale > 1000) {
        scale = ((scale + 1024) & 2047) - 1024;
      }
      columns.add(
          new Described(
              PgSession.oid(connection, meta.getColumnTypeName(at)), meta.getPrecision(at), scale));
    }
    return columns;
  }
}
