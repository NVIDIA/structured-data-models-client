// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.bridge.kumo;

import ai.kumo.graph.Column;
import ai.kumo.graph.Values;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BaseIntVector;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.impl.UnionListWriter;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.Text;

/** Arrow answers built from plain Java rows, and the value conversions both ways. */
final class Answers {
  private Answers() {}

  /** The column types of the stock prototype; a modified copy may add others. */
  private static final Set<String> KNOWN_TYPES =
      Set.of("STRING", "INT64", "FLOAT64", "BOOL", "TIMESTAMP", "STRINGLIST");

  /**
   * The type the bridge reads a column as: its own type when the stock prototype has it, else
   * {@code STRING}. A caller's copy of the prototype may add types (an id type, say) that this
   * code cannot name; their values are read and compared as text.
   */
  static Column.Type readAs(Column.Type type) {
    return Column.Type.valueOf(readAs(type.name()));
  }

  /** {@link #readAs(Column.Type)} by type name. */
  static String readAs(String typeName) {
    return KNOWN_TYPES.contains(typeName) ? typeName : "STRING";
  }

  /** The Arrow field of a prototype column; any type the bridge does not know is text. */
  static Field field(String name, Column.Type type) {
    ArrowType arrow =
        switch (type) {
          case INT64 -> new ArrowType.Int(64, true);
          case FLOAT64 -> new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
          case BOOL -> ArrowType.Bool.INSTANCE;
          case TIMESTAMP -> new ArrowType.Timestamp(TimeUnit.MICROSECOND, null);
          case STRINGLIST -> ArrowType.List.INSTANCE;
          default -> ArrowType.Utf8.INSTANCE;
        };
    List<Field> children =
        type == Column.Type.STRINGLIST
            ? List.of(new Field("item", FieldType.nullable(ArrowType.Utf8.INSTANCE), null))
            : null;
    return new Field(name, FieldType.nullable(arrow), children);
  }

  /** A reader over {@code rows}, each holding one value per field, in order. */
  static ArrowReader reader(Schema schema, List<Object[]> rows, BufferAllocator allocator)
      throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
        ArrowStreamWriter writer = new ArrowStreamWriter(root, null, Channels.newChannel(out))) {
      writer.start();
      root.allocateNew();
      List<FieldVector> vectors = root.getFieldVectors();
      for (int row = 0; row < rows.size(); row++) {
        Object[] values = rows.get(row);
        for (int column = 0; column < vectors.size(); column++) {
          set(vectors.get(column), row, values[column]);
        }
      }
      for (FieldVector vector : vectors) {
        vector.setValueCount(rows.size());
      }
      root.setRowCount(rows.size());
      writer.writeBatch();
      writer.end();
    }
    return new ArrowStreamReader(new ByteArrayInputStream(out.toByteArray()), allocator);
  }

  private static void set(FieldVector vector, int row, Object value) {
    if (value == null) {
      if (vector instanceof ListVector list) {
        list.setNull(row);
      } else {
        vector.setNull(row);
      }
      return;
    }
    if (vector instanceof BigIntVector v) {
      v.setSafe(row, number(value).longValue());
    } else if (vector instanceof IntVector v) {
      v.setSafe(row, number(value).intValue());
    } else if (vector instanceof Float8Vector v) {
      v.setSafe(row, number(value).doubleValue());
    } else if (vector instanceof BitVector v) {
      v.setSafe(row, bool(value) ? 1 : 0);
    } else if (vector instanceof VarCharVector v) {
      v.setSafe(row, text(value).getBytes(StandardCharsets.UTF_8));
    } else if (vector instanceof TimeStampVector v) {
      long micros = micros(value);
      v.setSafe(row, ((ArrowType.Timestamp) v.getField().getType()).getUnit() == TimeUnit.NANOSECOND
          ? Math.multiplyExact(micros, 1000L) : micros);
    } else if (vector instanceof ListVector v) {
      UnionListWriter writer = v.getWriter();
      writer.setPosition(row);
      writer.startList();
      for (Object item : list(value)) {
        if (item == null) {
          writer.writeNull();
        } else {
          byte[] bytes = item.toString().getBytes(StandardCharsets.UTF_8);
          try (var buffer = v.getAllocator().buffer(bytes.length)) {
            buffer.setBytes(0, bytes);
            writer.writeVarChar(0, bytes.length, buffer);
          }
        }
      }
      writer.endList();
    } else if (vector instanceof BaseIntVector v) {
      v.setWithPossibleTruncate(row, number(value).longValue());
    } else {
      throw new IllegalStateException("no conversion into " + vector.getField());
    }
  }

  private static Number number(Object value) {
    if (value instanceof Number number) {
      return number;
    }
    if (value instanceof Boolean flag) {
      return flag ? 1 : 0;
    }
    return new BigDecimal(value.toString().trim());
  }

  private static boolean bool(Object value) {
    if (value instanceof Boolean flag) {
      return flag;
    }
    if (value instanceof Number number) {
      return number.doubleValue() != 0;
    }
    return Boolean.parseBoolean(value.toString().trim());
  }

  private static String text(Object value) {
    if (value instanceof byte[] bytes) {
      return java.util.Base64.getEncoder().encodeToString(bytes);
    }
    return value.toString();
  }

  private static List<?> list(Object value) {
    if (value instanceof List<?> list) {
      return list;
    }
    if (value instanceof Object[] array) {
      return List.of(array);
    }
    String text = value.toString().trim();
    if (text.startsWith("[")) {
      Object parsed = ai.kumo.client.Json.read(text);
      if (parsed instanceof List<?> list) {
        return list;
      }
    }
    return List.of(text);
  }

  /** A timestamp as microseconds since the epoch; text and zone-less times are UTC. */
  static long micros(Object value) {
    Instant instant;
    if (value instanceof Number number) {
      return number.longValue();
    } else if (value instanceof Instant given) {
      instant = given;
    } else if (value instanceof LocalDateTime local) {
      instant = local.toInstant(ZoneOffset.UTC);
    } else if (value instanceof OffsetDateTime offset) {
      instant = offset.toInstant();
    } else if (value instanceof ZonedDateTime zoned) {
      instant = zoned.toInstant();
    } else if (value instanceof LocalDate date) {
      instant = date.atStartOfDay(ZoneOffset.UTC).toInstant();
    } else if (value instanceof java.sql.Timestamp timestamp) {
      instant = timestamp.toLocalDateTime().toInstant(ZoneOffset.UTC);
    } else if (value instanceof java.util.Date date) {
      instant = date.toInstant();
    } else {
      instant = Values.instant(value);
    }
    return Values.micros(instant);
  }

  /** An Arrow cell as a plain Java value: text as {@link String}, the rest as Arrow gives it. */
  static Object plain(FieldVector vector, int row) {
    if (vector.isNull(row)) {
      return null;
    }
    if (vector instanceof TimeStampVector timestamps) {
      long raw = timestamps.get(row);
      TimeUnit unit = ((ArrowType.Timestamp) vector.getField().getType()).getUnit();
      return switch (unit) {
        case SECOND -> Math.multiplyExact(raw, 1_000_000L);
        case MILLISECOND -> Math.multiplyExact(raw, 1000L);
        case MICROSECOND -> raw;
        case NANOSECOND -> Math.floorDiv(raw, 1000L);
      };
    }
    Object value = vector.getObject(row);
    if (value instanceof Text text) {
      return text.toString();
    }
    if (value instanceof List<?> list) {
      List<Object> items = new ArrayList<>();
      for (Object item : list) {
        items.add(item instanceof Text text ? text.toString() : item);
      }
      return items;
    }
    return value;
  }
}
