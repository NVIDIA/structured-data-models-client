// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Locale;
import java.util.TimeZone;
import net.snowflake.client.api.resultset.SnowflakeResultSetMetaData;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

/** Lossless public JDBC getters to Arrow; warehouse conformance remains in Rust. */
final class SnowflakeValues {
  private SnowflakeValues() {}

  private record Column(ArrowType type, Reading reading, List<Field> children) {
    Column(ArrowType type, Reading reading) {
      this(type, reading, List.of());
    }
  }

  @FunctionalInterface
  private interface Reading {
    Object read(ResultSet rows, int index) throws SQLException;
  }

  static ConversionTable.Result read(ResultSet rows, Connection connection) throws SQLException {
    ResultSetMetaData metadata = rows.getMetaData();
    String zone = null;
    for (int i = 1; i <= metadata.getColumnCount(); i++) {
      String name = metadata.getColumnTypeName(i).replace("_", "");
      if (name.equalsIgnoreCase("TIMESTAMPLTZ") || name.equalsIgnoreCase("TIMESTAMPTZ")) {
        zone = timezone(connection);
        break;
      }
    }
    Calendar utc = Calendar.getInstance(TimeZone.getTimeZone("UTC"));
    List<Column> columns = new ArrayList<>();
    List<Field> fields = new ArrayList<>();
    List<List<Object>> values = new ArrayList<>();
    for (int i = 1; i <= metadata.getColumnCount(); i++) {
      Column column =
          metadata.getColumnTypeName(i).equalsIgnoreCase("VECTOR")
              ? vector(metadata, i)
              : column(
                  metadata.getColumnTypeName(i),
                  metadata.getPrecision(i),
                  metadata.getScale(i),
                  zone,
                  utc);
      columns.add(column);
      fields.add(
          new Field(
              metadata.getColumnLabel(i), FieldType.nullable(column.type()), column.children()));
      values.add(new ArrayList<>());
    }
    while (rows.next()) {
      for (int i = 0; i < columns.size(); i++)
        values.get(i).add(columns.get(i).reading().read(rows, i + 1));
    }
    return new ConversionTable.Result(fields, values);
  }

  private static Column vector(ResultSetMetaData metadata, int index) throws SQLException {
    var snowflake = metadata.unwrap(SnowflakeResultSetMetaData.class);
    int width = snowflake.getVectorDimension(index);
    var fields = snowflake.getColumnFields(index);
    if (width <= 0 || fields.size() != 1)
      throw new SQLException("invalid Snowflake vector metadata");
    String element = fields.get(0).getTypeName().toUpperCase(Locale.ROOT);
    ArrowType type =
        switch (element) {
          case "FLOAT" -> ArrowColumns.FLOAT32;
          case "INT", "INTEGER", "NUMBER", "FIXED" -> ArrowColumns.INT32;
          default -> throw new SQLException("unsupported Snowflake vector element type");
        };
    return new Column(
        new ArrowType.FixedSizeList(width),
        (rows, at) -> {
          var array = rows.getArray(at);
          if (array == null) return null;
          try {
            return java.util.Arrays.asList((Object[]) array.getArray());
          } finally {
            array.free();
          }
        },
        List.of(new Field("element", new FieldType(false, type, null), List.of())));
  }

  static long ticks(Instant instant, long factor) {
    return java.math.BigInteger.valueOf(instant.getEpochSecond())
        .multiply(java.math.BigInteger.valueOf(factor))
        .add(java.math.BigInteger.valueOf(instant.getNano() / (1_000_000_000L / factor)))
        .longValueExact();
  }

  private static String timezone(Connection connection) throws SQLException {
    try (var statement = connection.createStatement();
        var rows = statement.executeQuery("SHOW PARAMETERS LIKE 'TIMEZONE' IN SESSION")) {
      if (!rows.next()) throw new SQLException("Snowflake session timezone is absent");
      String zone = rows.getString("value");
      if (zone == null || zone.isBlank())
        throw new SQLException("Snowflake session timezone is absent");
      return zone;
    }
  }

  private static Column column(String name, int precision, int scale, String zone, Calendar utc)
      throws SQLException {
    String type = name.toUpperCase(Locale.ROOT).replace("_", "");
    return switch (type) {
      case "NUMBER", "DECIMAL", "NUMERIC" ->
          new Column(ArrowColumns.decimal(precision, scale), ResultSet::getBigDecimal);
      case "FLOAT", "DOUBLE", "REAL" ->
          new Column(
              ArrowColumns.FLOAT64,
              (r, i) -> {
                double value = r.getDouble(i);
                return r.wasNull() ? null : value;
              });
      case "BOOLEAN" ->
          new Column(
              ArrowColumns.BOOL,
              (r, i) -> {
                boolean value = r.getBoolean(i);
                return r.wasNull() ? null : value;
              });
      case "BINARY", "VARBINARY" -> new Column(ArrowColumns.BINARY, ResultSet::getBytes);
      case "DATE" ->
          new Column(
              ArrowColumns.DATE32,
              (r, i) -> {
                var date = r.getDate(i, utc);
                return date == null
                    ? null
                    : Math.toIntExact(Math.floorDiv(date.getTime(), 86_400_000L));
              });
      case "TIME" ->
          new Column(
              new ArrowType.Time(TimeUnit.NANOSECOND, 64),
              (r, i) -> {
                var value = r.getTimestamp(i, utc);
                if (value == null) return null;
                Instant instant = value.toInstant();
                // JDBC exposes TIME through a timestamp on the epoch day. Ignore its date
                // explicitly.
                return Math.floorMod(instant.getEpochSecond(), 86_400L) * 1_000_000_000L
                    + instant.getNano();
              });
      case "TIMESTAMP", "TIMESTAMPNTZ", "TIMESTAMPLTZ", "TIMESTAMPTZ" -> {
        boolean aware = type.equals("TIMESTAMPLTZ") || type.equals("TIMESTAMPTZ");
        TimeUnit unit =
            scale == 0
                ? TimeUnit.SECOND
                : scale <= 3
                    ? TimeUnit.MILLISECOND
                    : scale <= 6 ? TimeUnit.MICROSECOND : TimeUnit.NANOSECOND;
        yield new Column(
            new ArrowType.Timestamp(unit, aware ? zone : null),
            (r, i) -> {
              var value = r.getTimestamp(i, utc);
              if (value == null) return null;
              Instant instant = value.toInstant();
              long factor =
                  switch (unit) {
                    case SECOND -> 1L;
                    case MILLISECOND -> 1_000L;
                    case MICROSECOND -> 1_000_000L;
                    case NANOSECOND -> 1_000_000_000L;
                  };
              return ticks(instant, factor);
            });
      }
      // SHOW/DESCRIBE use VARCHAR. Unsupported warehouse types are described and omitted by Rust.
      default -> new Column(ArrowColumns.UTF8, ResultSet::getString);
    };
  }
}
