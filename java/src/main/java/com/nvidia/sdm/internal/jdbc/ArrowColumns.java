// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.BitVectorHelper;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.Decimal256Vector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.IntervalMonthDayNanoVector;
import org.apache.arrow.vector.NullVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TimeMicroVector;
import org.apache.arrow.vector.TimeStampMicroTZVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.IntervalUnit;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Arrow columns built from the values a driver read, each already the Java value its Arrow type
 * holds ({@link Value}): the one place the host writes Arrow it made itself (a result read from
 * rows, a key set or the entities a call names). Not part of the API.
 */
public final class ArrowColumns {
  private ArrowColumns() {}

  /** The Arrow types the host builds, by their Arrow name. */
  public static final ArrowType INT8 = new ArrowType.Int(8, true);

  public static final ArrowType INT16 = new ArrowType.Int(16, true);

  public static final ArrowType INT32 = new ArrowType.Int(32, true);
  public static final ArrowType INT64 = new ArrowType.Int(64, true);
  public static final ArrowType BOOL = ArrowType.Bool.INSTANCE;
  public static final ArrowType FLOAT32 = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
  public static final ArrowType FLOAT64 = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
  public static final ArrowType UTF8 = ArrowType.Utf8.INSTANCE;
  public static final ArrowType BINARY = ArrowType.Binary.INSTANCE;
  public static final ArrowType DATE32 = new ArrowType.Date(DateUnit.DAY);
  public static final ArrowType TIMESTAMP_US = new ArrowType.Timestamp(TimeUnit.MICROSECOND, null);
  public static final ArrowType TIMESTAMP_US_UTC =
      new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC");
  public static final ArrowType TIME64_US = new ArrowType.Time(TimeUnit.MICROSECOND, 64);
  public static final ArrowType MONTH_DAY_NANO = new ArrowType.Interval(IntervalUnit.MONTH_DAY_NANO);
  public static final ArrowType NULL = ArrowType.Null.INSTANCE;

  /** A decimal of {@code precision} and {@code scale}: 128 bits to 38 digits, 256 beyond. */
  public static ArrowType decimal(int precision, int scale) {
    return new ArrowType.Decimal(precision, scale, precision <= 38 ? 128 : 256);
  }

  /** A month, day and nanosecond interval, as Arrow's {@code MonthDayNano} holds one. */
  public record MonthDayNano(int months, int days, long nanos) {}

  /** One batch of {@code columns}' values, each a list as long as every other. */
  public static VectorSchemaRoot batch(
      Schema schema, List<List<Object>> columns, BufferAllocator allocator) {
    List<FieldVector> vectors = new ArrayList<>();
    int rows = columns.isEmpty() ? 0 : columns.get(0).size();
    try {
      for (int at = 0; at < schema.getFields().size(); at++) {
        vectors.add(column(schema.getFields().get(at), columns.get(at), allocator));
      }
    } catch (RuntimeException e) {
      vectors.forEach(FieldVector::close);
      throw e;
    }
    return new VectorSchemaRoot(schema.getFields(), vectors, rows);
  }

  /** {@code field}'s column of {@code values}. */
  public static FieldVector column(Field field, List<Object> values, BufferAllocator allocator) {
    FieldVector vector = field.createVector(allocator);
    try {
      fill(vector, values);
    } catch (RuntimeException e) {
      vector.close();
      throw e;
    }
    return vector;
  }

  private static void fill(FieldVector vector, List<Object> values) {
    int rows = values.size();
    if (vector instanceof NullVector nulls) {
      nulls.setValueCount(rows);
      return;
    }
    if (vector instanceof ListVector list) {
      fillList(list, values);
      return;
    }
    vector.setInitialCapacity(rows);
    vector.allocateNew();
    for (int at = 0; at < rows; at++) {
      Object value = values.get(at);
      if (value == null) {
        vector.setNull(at);
        continue;
      }
      set(vector, at, value);
    }
    vector.setValueCount(rows);
  }

  private static void set(FieldVector vector, int at, Object value) {
    if (vector instanceof BigIntVector v) {
      v.setSafe(at, ((Number) value).longValue());
    } else if (vector instanceof IntVector v) {
      v.setSafe(at, ((Number) value).intValue());
    } else if (vector instanceof SmallIntVector v) {
      v.setSafe(at, ((Number) value).shortValue());
    } else if (vector instanceof TinyIntVector v) {
      v.setSafe(at, ((Number) value).byteValue());
    } else if (vector instanceof Float4Vector v) {
      v.setSafe(at, ((Number) value).floatValue());
    } else if (vector instanceof BitVector v) {
      v.setSafe(at, Boolean.TRUE.equals(value) ? 1 : 0);
    } else if (vector instanceof Float8Vector v) {
      v.setSafe(at, ((Number) value).doubleValue());
    } else if (vector instanceof VarCharVector v) {
      v.setSafe(at, ((String) value).getBytes(StandardCharsets.UTF_8));
    } else if (vector instanceof VarBinaryVector v) {
      v.setSafe(at, (byte[]) value);
    } else if (vector instanceof DateDayVector v) {
      v.setSafe(at, ((Number) value).intValue());
    } else if (vector instanceof TimeStampMicroTZVector v) {
      v.setSafe(at, ((Number) value).longValue());
    } else if (vector instanceof TimeStampMicroVector v) {
      v.setSafe(at, ((Number) value).longValue());
    } else if (vector instanceof TimeMicroVector v) {
      v.setSafe(at, ((Number) value).longValue());
    } else if (vector instanceof IntervalMonthDayNanoVector v) {
      MonthDayNano interval = (MonthDayNano) value;
      v.setSafe(at, interval.months(), interval.days(), interval.nanos());
    } else if (vector instanceof DecimalVector v) {
      v.setSafe(at, scaled((BigDecimal) value, v.getScale()));
    } else if (vector instanceof Decimal256Vector v) {
      v.setSafe(at, scaled((BigDecimal) value, v.getScale()));
    } else {
      throw new IllegalArgumentException(
          "no Arrow column of " + vector.getField().getType() + " is built here");
    }
  }

  /** {@code value} at {@code scale}: a decimal's digits are never rounded. */
  private static BigDecimal scaled(BigDecimal value, int scale) {
    return value.setScale(scale, RoundingMode.UNNECESSARY);
  }

  private static void fillList(ListVector list, List<Object> values) {
    int rows = values.size();
    List<Object> flat = new ArrayList<>();
    for (Object value : values) {
      if (value != null) {
        flat.addAll((List<?>) value);
      }
    }
    list.setInitialCapacity(Math.max(rows, 1));
    list.allocateNew();
    while (list.getValueCapacity() < rows) {
      list.reAlloc();
    }
    // The elements as one column, then each row's span of them.
    fill(list.getDataVector(), flat);
    int offset = 0;
    list.getOffsetBuffer().setInt(0, 0);
    for (int at = 0; at < rows; at++) {
      Object value = values.get(at);
      if (value != null) {
        BitVectorHelper.setBit(list.getValidityBuffer(), at);
        offset += ((List<?>) value).size();
      }
      list.getOffsetBuffer().setInt((long) (at + 1) * ListVector.OFFSET_WIDTH, offset);
    }
    list.setLastSet(rows - 1);
    list.setValueCount(rows);
  }

  /** A double as Arrow's cast to text writes it ({@link ArrowText#floating}). */
  public static String floatText(double value) {
    return ArrowText.floating(value);
  }

  /** A float32 as Arrow's cast to text writes it, at its own width. */
  public static String floatText(float value) {
    return ArrowText.floating32(value);
  }

  /** A field named {@code name} of {@code type}, nullable, with {@code metadata} or none. */
  public static Field field(String name, ArrowType type, Map<String, String> metadata) {
    return new Field(name, new FieldType(true, type, null, metadata), null);
  }

  /** A list field named {@code name} of the element field {@code element}. */
  public static Field list(String name, Field element) {
    return new Field(name, FieldType.nullable(ArrowType.List.INSTANCE), List.of(element));
  }
}
