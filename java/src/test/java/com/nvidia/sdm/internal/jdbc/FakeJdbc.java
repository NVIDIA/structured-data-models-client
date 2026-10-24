// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.math.BigDecimal;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;
import java.util.function.Function;

/**
 * A JDBC connection standing in for the Databricks JDBC driver's (T3.13), fed answers as the
 * driver gives them: each result a list of columns, each its type name as the driver names it
 * ({@code getColumnTypeName}), its precision and scale, and its values as the driver's getters
 * read them; a failure a {@link SQLException} with the driver's state and message. It records
 * every statement run and every value bound, in order. Only what the provider calls is answered;
 * anything else fails the test.
 */
final class FakeJdbc {
  private FakeJdbc() {}

  /** One column of an answer: its name, the driver's type name, precision, scale and values. */
  record Column(String name, String type, int precision, int scale, List<Object> values) {
    static Column of(String name, String type, Object... values) {
      List<Object> list = new ArrayList<>();
      Collections.addAll(list, values);
      return new Column(name, type, 0, 0, list);
    }

    static Column decimal(String name, int precision, int scale, Object... values) {
      List<Object> list = new ArrayList<>();
      Collections.addAll(list, values);
      return new Column(name, "DECIMAL", precision, scale, list);
    }
  }

  /** A statement the provider ran: its text, and the values bound to it, in order. */
  record Ran(String text, List<Object> bound) {}

  /** The connection, what it ran, and how it answers each text. */
  static final class Database {
    final List<Ran> ran = Collections.synchronizedList(new ArrayList<>());
    private final Function<String, Object> answers;
    int statementsOpen;

    /** Answers each text with {@code answers}: a list of columns, or a {@link SQLException}. */
    Database(Function<String, Object> answers) {
      this.answers = answers;
    }

    Connection connection() {
      return (Connection)
          proxy(
              Connection.class,
              (self, method, args) ->
                  switch (method.getName()) {
                    case "createStatement" -> statement(null);
                    case "prepareStatement" -> statement((String) args[0]);
                    case "getAutoCommit" -> true;
                    case "isClosed" -> false;
                    case "close" -> null;
                    case "toString" -> "FakeJdbc.Database";
                    case "hashCode" -> System.identityHashCode(self);
                    case "equals" -> self == args[0];
                    default -> throw new AssertionError("Connection." + method.getName());
                  });
    }

    private Object statement(String prepared) {
      statementsOpen++;
      Map<Integer, Object> bound = new LinkedHashMap<>();
      Class<?> type = prepared == null ? Statement.class : PreparedStatement.class;
      return proxy(
          type,
          (self, method, args) -> {
            String name = method.getName();
            if (name.startsWith("set") && prepared != null && args != null && args.length >= 2
                && args[0] instanceof Integer at) {
              bound.put(at, args[1]);
              return null;
            }
            return switch (name) {
              case "execute" -> {
                run(prepared == null ? (String) args[0] : prepared, bound);
                yield false;
              }
              case "executeQuery" -> run(prepared == null ? (String) args[0] : prepared, bound);
              case "setEscapeProcessing", "setFetchSize" -> null;
              case "close" -> {
                statementsOpen--;
                yield null;
              }
              case "toString" -> "FakeJdbc.Statement";
              case "hashCode" -> System.identityHashCode(self);
              case "equals" -> self == args[0];
              default -> throw new AssertionError("Statement." + name);
            };
          });
    }

    @SuppressWarnings("unchecked")
    private ResultSet run(String text, Map<Integer, Object> bound) throws SQLException {
      List<Object> values = new ArrayList<>();
      for (int at = 1; at <= bound.size(); at++) {
        values.add(bound.get(at));
      }
      ran.add(new Ran(text, values));
      Object answer = answers.apply(text);
      if (answer instanceof SQLException failed) {
        throw failed;
      }
      return result((List<Column>) answer);
    }
  }

  private static Object proxy(Class<?> type, InvocationHandler handler) {
    return Proxy.newProxyInstance(FakeJdbc.class.getClassLoader(), new Class<?>[] {type}, handler);
  }

  /** A result set over {@code columns}, read as the Databricks driver's getters read a value. */
  static ResultSet result(List<Column> columns) {
    int rows = columns.isEmpty() ? 0 : columns.get(0).values().size();
    int[] at = {-1};
    boolean[] lastNull = {false};
    ResultSetMetaData meta =
        (ResultSetMetaData)
            proxy(
                ResultSetMetaData.class,
                (self, method, args) -> {
                  Column column = args == null ? null : columns.get((Integer) args[0] - 1);
                  return switch (method.getName()) {
                    case "getColumnCount" -> columns.size();
                    case "getColumnLabel", "getColumnName" -> column.name();
                    case "getColumnTypeName" -> column.type();
                    case "getPrecision" -> column.precision();
                    case "getScale" -> column.scale();
                    default -> throw new AssertionError("ResultSetMetaData." + method.getName());
                  };
                });
    return (ResultSet)
        proxy(
            ResultSet.class,
            (self, method, args) -> {
              String name = method.getName();
              switch (name) {
                case "next":
                  at[0]++;
                  return at[0] < rows;
                case "getMetaData":
                  return meta;
                case "wasNull":
                  return lastNull[0];
                case "close":
                  return null;
                case "toString":
                  return "FakeJdbc.ResultSet";
                case "hashCode":
                  return System.identityHashCode(self);
                case "equals":
                  return self == args[0];
                default:
                  break;
              }
              if (!name.startsWith("get") || args == null || !(args[0] instanceof Integer index)) {
                throw new AssertionError("ResultSet." + name);
              }
              Object value = columns.get(index - 1).values().get(at[0]);
              lastNull[0] = value == null;
              return read(name, value, args);
            });
  }

  /** A value as the getter {@code name} reads it, a null as the getter's zero. */
  private static Object read(String name, Object value, Object[] args) throws SQLException {
    switch (name) {
      case "getLong":
        return value == null ? 0L : ((Number) value).longValue();
      case "getInt":
        return value == null ? 0 : ((Number) value).intValue();
      case "getShort":
        return value == null ? (short) 0 : ((Number) value).shortValue();
      case "getByte":
        return value == null ? (byte) 0 : ((Number) value).byteValue();
      case "getDouble":
        return value == null ? 0.0 : ((Number) value).doubleValue();
      case "getFloat":
        return value == null ? 0.0f : ((Number) value).floatValue();
      case "getBoolean":
        return value != null && (Boolean) value;
      case "getBigDecimal":
        return (BigDecimal) value;
      case "getString":
        return value == null ? null : value.toString();
      case "getBytes":
        return (byte[]) value;
      case "getArray":
        return value == null ? null : array((List<?>) value);
      case "getTimestamp":
        return timestamp(value, args.length > 1 ? (Calendar) args[1] : null);
      case "getObject":
        return args.length > 1 ? as(value, (Class<?>) args[1]) : value;
      default:
        throw new AssertionError("ResultSet." + name);
    }
  }

  /**
   * A timestamp as the driver gives one: a {@code TIMESTAMP}'s instant ({@link Instant}), a
   * {@code TIMESTAMP_NTZ}'s wall time ({@link LocalDateTime}), each read in the calendar's zone, or
   * the JVM's.
   */
  private static Timestamp timestamp(Object value, Calendar calendar) {
    if (value == null) {
      return null;
    }
    TimeZone zone = calendar == null ? TimeZone.getDefault() : calendar.getTimeZone();
    if (value instanceof Instant instant) {
      return Timestamp.from(instant);
    }
    // The released driver first constructs a Timestamp in the JVM's zone,
    // even when its calendar getter is used afterwards.
    LocalDateTime wall = Timestamp.valueOf((LocalDateTime) value).toLocalDateTime();
    return Timestamp.from(wall.atZone(zone.toZoneId()).toInstant());
  }

  private static Object as(Object value, Class<?> type) throws SQLException {
    if (value == null) {
      return null;
    }
    // Databricks JDBC 3.4.2 has no LocalDateTime typed-getter branch.
    // Its fallback is TimestampConverter.toString, including its JVM-zone conversion.
    if (type == LocalDateTime.class) {
      return timestamp(value, null).toString();
    }
    if (type.isInstance(value)) {
      return value;
    }
    if (type == LocalDate.class && value instanceof LocalDate date) {
      return date;
    }
    throw new SQLException("cannot read " + value.getClass().getSimpleName() + " as " + type);
  }

  private static java.sql.Array array(List<?> values) {
    return (java.sql.Array)
        proxy(
            java.sql.Array.class,
            (self, method, args) ->
                switch (method.getName()) {
                  case "getArray" -> values.toArray();
                  case "free" -> null;
                  default -> throw new AssertionError("Array." + method.getName());
                });
  }
}
