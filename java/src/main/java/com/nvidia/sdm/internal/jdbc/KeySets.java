// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.ArrowExchange;
import com.nvidia.sdm.internal.Trace;
import java.math.BigInteger;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Types;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;

/**
 * How each driver binds a key set a plan names (the seeds of a {@code neighbors} or {@code
 * lookup} request), the Arrow attachment the core sent with the step, as the Python provider for
 * that driver binds it: DuckDB's registered as Arrow (kept as a temporary table, since a
 * registered stream is read once), SQLite's ingested as a temporary table of the types ADBC
 * gives it, Postgres's the statement's one parameter, JSON text. Not part of the API.
 */
abstract class KeySets {
  /** A key set read whole: its Arrow IPC bytes, its fields and each column's values. */
  record Keys(byte[] ipc, List<Field> fields, List<List<Object>> columns) {
    int rows() {
      return columns.isEmpty() ? 0 : columns.get(0).size();
    }

    /** {@code reader} read whole; it is closed. */
    static Keys read(ArrowReader reader) throws Exception {
      byte[] ipc = ArrowData.ipc(reader);
      List<Field> fields;
      List<List<Object>> columns = new ArrayList<>();
      try (ArrowReader again = ArrowData.reader(ipc, ArrowData.allocator())) {
        VectorSchemaRoot root = again.getVectorSchemaRoot();
        fields = root.getSchema().getFields();
        for (int at = 0; at < fields.size(); at++) {
          columns.add(new ArrayList<>());
        }
        while (again.loadNextBatch()) {
          for (int at = 0; at < fields.size(); at++) {
            FieldVector vector = root.getVector(at);
            for (int row = 0; row < root.getRowCount(); row++) {
              columns.get(at).add(vector.isNull(row) ? null : value(vector, row));
            }
          }
        }
      }
      return new Keys(ipc, fields, columns);
    }
  }

  /** One cell of a key set as Java holds it. */
  static Object value(FieldVector vector, int row) {
    if (vector instanceof BigIntVector v) {
      return v.get(row);
    }
    if (vector instanceof IntVector v) {
      return (long) v.get(row);
    }
    if (vector instanceof SmallIntVector v) {
      return (long) v.get(row);
    }
    if (vector instanceof TinyIntVector v) {
      return (long) v.get(row);
    }
    if (vector instanceof UInt1Vector v) {
      return (long) UInt1Vector.getNoOverflow(v.getDataBuffer(), row);
    }
    if (vector instanceof UInt2Vector v) {
      return (long) v.get(row);
    }
    if (vector instanceof UInt4Vector v) {
      return UInt4Vector.getNoOverflow(v.getDataBuffer(), row);
    }
    if (vector instanceof UInt8Vector v) {
      return UInt8Vector.getNoOverflow(v.getDataBuffer(), row);
    }
    if (vector instanceof Float8Vector v) {
      return v.get(row);
    }
    if (vector instanceof Float4Vector v) {
      return (double) v.get(row);
    }
    if (vector instanceof BitVector v) {
      return v.get(row) == 1;
    }
    if (vector instanceof VarCharVector v) {
      return new String(v.get(row), java.nio.charset.StandardCharsets.UTF_8);
    }
    if (vector instanceof VarBinaryVector v) {
      return v.get(row);
    }
    if (vector instanceof DecimalVector v) {
      return v.getObject(row);
    }
    if (vector instanceof DateDayVector v) {
      return LocalDate.ofEpochDay(v.get(row));
    }
    if (vector instanceof TimeStampVector v) {
      return new Stamp(v.get(row), ((ArrowType.Timestamp) v.getField().getType()));
    }
    throw new IllegalArgumentException(
        "a key set column of " + vector.getField().getType() + " is not one a provider binds");
  }

  /** A timestamp cell: its count and its Arrow type. */
  record Stamp(long count, ArrowType.Timestamp type) {
    Instant instant() {
      return switch (type.getUnit()) {
        case SECOND -> Instant.ofEpochSecond(count);
        case MILLISECOND -> Instant.ofEpochMilli(count);
        case MICROSECOND -> Instant.ofEpochSecond(
            Math.floorDiv(count, 1_000_000L), Math.floorMod(count, 1_000_000L) * 1_000L);
        case NANOSECOND -> Instant.ofEpochSecond(
            Math.floorDiv(count, 1_000_000_000L), Math.floorMod(count, 1_000_000_000L));
      };
    }
  }

  static KeySets of(JdbcProvider.Driver driver) {
    return switch (driver) {
      case DUCKDB -> new DuckDbKeys();
      case SQLITE -> new SqliteKeys();
      case POSTGRES -> new PostgresKeys();
    };
  }

  /** Binds {@code keys} as the relation {@code name} on {@code on}. */
  abstract void bind(Connection on, String name, Keys keys, BufferAllocator allocator)
      throws Exception;

  static String quoted(String name) {
    return "\"" + name.replace("\"", "\"\"") + "\"";
  }

  // --- DuckDB ---------------------------------------------------------------------------

  /**
   * DuckDB's: the key set's Arrow registered on the cursor under a name of this module's, and
   * kept as a temporary table of the relation's name, so its rows reach the statement as Arrow,
   * never as text, and a statement may read it more than once.
   */
  static final class DuckDbKeys extends KeySets {
    private static final AtomicLong NEXT = new AtomicLong();

    @Override
    void bind(Connection on, String name, Keys keys, BufferAllocator allocator) throws Exception {
      String registered = "__sdm_keys_" + NEXT.incrementAndGet();
      String sql =
          "CREATE TEMPORARY TABLE " + quoted(name) + " AS SELECT * FROM " + quoted(registered);
      try (ArrowArrayStream stream =
          ArrowExchange.export(allocator, ArrowData.reader(keys.ipc(), allocator))) {
        DuckDb.register(on, registered, stream);
        Trace trace = Trace.installed();
        if (trace != null) {
          trace.bound(0, name, sql, keys.ipc());
        }
        try (Statement statement = on.createStatement()) {
          statement.execute(sql);
        }
      }
    }
  }

  // --- SQLite ---------------------------------------------------------------------------

  /**
   * SQLite's, as ADBC's {@code adbc_ingest(..., temporary=True)} makes one: a temporary table of
   * the relation's name, each column of the type ADBC declares for its Arrow type, created, or
   * replaced only where this module made that name on the connection before (a relation the
   * caller made is never replaced, and one of the same name fails the bind).
   */
  static final class SqliteKeys extends KeySets {
    private static final Map<Connection, Set<String>> MADE =
        Collections.synchronizedMap(new WeakHashMap<>());

    private static final DateTimeFormatter STAMP =
        DateTimeFormatter.ofPattern("uuuu-MM-dd'T'HH:mm:ss.SSSSSS");

    @Override
    void bind(Connection on, String name, Keys keys, BufferAllocator allocator) throws Exception {
      Set<String> made = MADE.computeIfAbsent(on, c -> Collections.synchronizedSet(new HashSet<>()));
      List<String> columns = new ArrayList<>();
      for (Field field : keys.fields()) {
        String declared = declared(field.getType());
        columns.add(quoted(field.getName()) + (declared.isEmpty() ? "" : " " + declared));
      }
      String create = "CREATE TEMPORARY TABLE " + quoted(name) + " (" + String.join(", ", columns) + ")";
      Trace trace = Trace.installed();
      if (trace != null) {
        trace.bound(0, name, create, keys.ipc());
      }
      try (Statement statement = on.createStatement()) {
        if (made.contains(name)) {
          statement.execute("DROP TABLE IF EXISTS temp." + quoted(name));
        }
        statement.execute(create);
      }
      made.add(name);
      String marks = String.join(", ", Collections.nCopies(keys.fields().size(), "?"));
      try (PreparedStatement insert =
          on.prepareStatement("INSERT INTO temp." + quoted(name) + " VALUES (" + marks + ")")) {
        for (int row = 0; row < keys.rows(); row++) {
          for (int at = 0; at < keys.fields().size(); at++) {
            set(insert, at + 1, keys.columns().get(at).get(row));
          }
          insert.addBatch();
        }
        insert.executeBatch();
      }
    }

    /** The type ADBC SQLite declares a column of {@code type} with. */
    private static String declared(ArrowType type) throws SQLException {
      if (type instanceof ArrowType.Int || type instanceof ArrowType.Bool) {
        return "INTEGER";
      }
      if (type instanceof ArrowType.FloatingPoint) {
        return "REAL";
      }
      if (type instanceof ArrowType.Utf8 || type instanceof ArrowType.LargeUtf8
          || type instanceof ArrowType.Date) {
        return "TEXT";
      }
      if (type instanceof ArrowType.Binary || type instanceof ArrowType.LargeBinary) {
        return "BLOB";
      }
      if (type instanceof ArrowType.Timestamp) {
        return "";
      }
      throw new SQLException("a key set column of " + type + " ADBC SQLite does not ingest");
    }

    private static void set(PreparedStatement insert, int at, Object value) throws SQLException {
      if (value == null) {
        insert.setNull(at, Types.NULL);
      } else if (value instanceof Long whole) {
        insert.setLong(at, whole);
      } else if (value instanceof BigInteger big) {
        // An unsigned 64-bit value past a signed one, which ADBC refuses too.
        insert.setLong(at, big.longValueExact());
      } else if (value instanceof Boolean flag) {
        insert.setLong(at, flag ? 1 : 0);
      } else if (value instanceof Double real) {
        insert.setDouble(at, real);
      } else if (value instanceof String text) {
        insert.setString(at, text);
      } else if (value instanceof byte[] bytes) {
        insert.setBytes(at, bytes);
      } else if (value instanceof LocalDate date) {
        insert.setString(at, date.toString());
      } else if (value instanceof Stamp stamp) {
        insert.setString(at, LocalDateTime.ofInstant(stamp.instant(), ZoneOffset.UTC).format(STAMP));
      } else {
        throw new SQLException("a key set value ADBC SQLite does not ingest");
      }
    }
  }

  // --- Postgres -------------------------------------------------------------------------

  /** Postgres's: bound as the statement's one parameter ({@link #postgresText}). */
  static final class PostgresKeys extends KeySets {
    @Override
    void bind(Connection on, String name, Keys keys, BufferAllocator allocator) {
      throw new UnsupportedOperationException("a Postgres key set is a statement's parameter");
    }
  }

  /**
   * A key set as the Postgres dialect binds it: a JSON array of its rows, each an array of its
   * cells' texts in the key set's column order, {@code null} for a null, each cell Arrow's text of
   * the value (pyarrow's cast to text, {@link ArrowText}), written as Python's {@code
   * json.dumps(rows, separators=(',', ':'), ensure_ascii=False)} writes it. The core hands over
   * the cells the reference spells otherwise (a time without a zone, binary) already as text.
   */
  static String postgresText(Keys keys) {
    StringBuilder out = new StringBuilder("[");
    for (int row = 0; row < keys.rows(); row++) {
      if (row > 0) {
        out.append(',');
      }
      out.append('[');
      for (int at = 0; at < keys.fields().size(); at++) {
        if (at > 0) {
          out.append(',');
        }
        Object value = keys.columns().get(at).get(row);
        out.append(value == null ? "null" : jsonString(ArrowText.of(value, keys.fields().get(at))));
      }
      out.append(']');
    }
    return out.append(']').toString();
  }

  /** Text as {@code json.dumps(..., ensure_ascii=False)} writes it. */
  static String jsonString(String text) {
    StringBuilder out = new StringBuilder("\"");
    for (int at = 0; at < text.length(); at++) {
      char c = text.charAt(at);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        case '\b' -> out.append("\\b");
        case '\f' -> out.append("\\f");
        default -> {
          if (c < 0x20) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    return out.append('"').toString();
  }

  /**
   * A statement binding its key set as {@code $1}, as JDBC reads a statement with a parameter:
   * each {@code ?} outside quotes doubled (pgjdbc reads it as a placeholder), and the parameter,
   * the one {@code $1} outside a quoted identifier ({@code "..."}) and a text literal ({@code
   * '...'}, or {@code E'...'} with its backslash escapes), the only quoting the dialect writes, as
   * {@code ?}. A {@code $1} inside quotes is text and stays.
   *
   * @throws JdbcProvider.PlanNotRun the text holds no such {@code $1}, or more than one
   */
  static String jdbcText(String text) throws JdbcProvider.PlanNotRun {
    StringBuilder out = new StringBuilder();
    int found = 0;
    int at = 0;
    while (at < text.length()) {
      char c = text.charAt(at);
      if (c == '"' || c == '\'') {
        boolean escaped = c == '\'' && at > 0 && "Ee".indexOf(text.charAt(at - 1)) >= 0;
        int end = at + 1;
        while (end < text.length()) {
          if (escaped && text.charAt(end) == '\\') {
            end += 2;
            continue;
          }
          if (text.charAt(end) == c) {
            if (end + 1 < text.length() && text.charAt(end + 1) == c) {
              end += 2;
              continue;
            }
            break;
          }
          end++;
        }
        out.append(text, at, Math.min(end + 1, text.length()));
        at = end + 1;
      } else if (text.startsWith("$1", at)
          && !(at + 2 < text.length() && Character.isDigit(text.charAt(at + 2)))) {
        out.append('?');
        found++;
        at += 2;
      } else {
        out.append(c == '?' ? "??" : String.valueOf(c));
        at++;
      }
    }
    if (found != 1) {
      throw new JdbcProvider.PlanNotRun("a statement whose parameter is not told apart");
    }
    return out.toString();
  }
}
