// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.internal.jdbc.ArrowColumns;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * The entities a call names, as one Arrow column {@code indices}, of the type their Java values
 * are, as the Python host builds it of the type pyarrow infers from the values ({@code
 * _rust/materialize.py}'s {@code _indices}): whole numbers an {@code int64}, with a {@code Double}
 * among them a {@code double}, text {@code utf8}, booleans {@code bool}, decimals a {@code
 * decimal128} as wide as the widest needs, nulls alone the null type; values of more than one
 * kind, or a whole number past 64 bits, each as its text ({@link #text}), a null null. The core
 * casts the column to the key's type by its own rule table and refuses an entity that does not
 * cast (T1.32a); this decides nothing of it. Not part of the API.
 */
public final class Indices {
  private Indices() {}

  /** {@code values} as the one-column table the core reads. */
  public static ArrowReader column(List<Object> values, BufferAllocator allocator) {
    ArrowType type = type(values);
    List<Object> written = new ArrayList<>();
    for (Object value : values) {
      written.add(value == null ? null : cell(value, type));
    }
    Field field = ArrowColumns.field("indices", type, null);
    Schema schema = new Schema(List.of(field));
    VectorSchemaRoot batch = ArrowColumns.batch(schema, List.of(written), allocator);
    return ArrowData.reader(schema, List.of(batch), allocator);
  }

  private static boolean whole(Object value) {
    return value instanceof Long || value instanceof Integer || value instanceof Short
        || value instanceof Byte
        || value instanceof BigInteger big && big.bitLength() < 64;
  }

  private static ArrowType type(List<Object> values) {
    boolean any = false;
    boolean wholes = true;
    boolean reals = true;
    boolean real = false;
    boolean texts = true;
    boolean flags = true;
    boolean decimals = true;
    int integers = 0;
    int scale = 0;
    for (Object value : values) {
      if (value == null) {
        continue;
      }
      any = true;
      wholes &= whole(value);
      reals &= whole(value) || value instanceof Double || value instanceof Float;
      real |= value instanceof Double || value instanceof Float;
      texts &= value instanceof String;
      flags &= value instanceof Boolean;
      decimals &= value instanceof BigDecimal;
      if (value instanceof BigDecimal decimal) {
        integers = Math.max(integers, decimal.precision() - decimal.scale());
        scale = Math.max(scale, decimal.scale());
      }
    }
    if (!any) {
      return ArrowColumns.NULL;
    }
    if (wholes) {
      return ArrowColumns.INT64;
    }
    if (reals && real) {
      return ArrowColumns.FLOAT64;
    }
    if (flags) {
      return ArrowColumns.BOOL;
    }
    if (decimals) {
      int precision = Math.max(1, integers + scale);
      if (precision <= 38) {
        return ArrowColumns.decimal(precision, scale);
      }
    }
    return ArrowColumns.UTF8;
  }

  private static Object cell(Object value, ArrowType type) {
    if (type == ArrowColumns.INT64) {
      return ((Number) value).longValue();
    }
    if (type == ArrowColumns.FLOAT64) {
      return ((Number) value).doubleValue();
    }
    if (type == ArrowColumns.UTF8) {
      return text(value);
    }
    return value;
  }

  /**
   * A value as its text, a boolean as Arrow writes one and a float as the core writes a float
   * entity (pyarrow's cast of it to text, at its own width), so the core reads back the number
   * the host was given.
   */
  static String text(Object value) {
    if (value instanceof Boolean flag) {
      return flag ? "true" : "false";
    }
    if (value instanceof Float single) {
      return com.nvidia.sdm.internal.jdbc.ArrowColumns.floatText(single);
    }
    if (value instanceof Double real) {
      return com.nvidia.sdm.internal.jdbc.ArrowColumns.floatText(real);
    }
    return String.valueOf(value);
  }
}
