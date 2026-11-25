// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.*;

/** Public Spark Row values mapped to Arrow without pandas or floating-point integer conversion. */
final class SparkValues {
  private SparkValues() {}
  static Schema schema(StructType schema, String zone) {
    List<Field> fields = new ArrayList<>();
    for (StructField field : schema.fields()) fields.add(field(field.name(), field.dataType(), field.nullable(), zone));
    return new Schema(fields);
  }
  private static Field field(String name, DataType type, boolean nullable, String zone) {
    ArrowType arrow;
    List<Field> children = List.of();
    if (type instanceof ArrayType array) {
      arrow = ArrowType.List.INSTANCE;
      children = List.of(field("item", array.elementType(), array.containsNull(), zone));
    } else if (type instanceof StructType structure) {
      arrow = ArrowType.Struct.INSTANCE;
      children = schema(structure, zone).getFields();
    } else if (type instanceof MapType map) {
      arrow = new ArrowType.Map(false);
      children = List.of(new Field("entries", FieldType.notNullable(ArrowType.Struct.INSTANCE),
          List.of(field("key", map.keyType(), false, zone), field("value", map.valueType(), map.valueContainsNull(), zone))));
    } else if (type instanceof DecimalType decimal) {
      arrow = ArrowColumns.decimal(decimal.precision(), decimal.scale());
    } else {
      arrow = switch (type.typeName()) {
        case "byte" -> ArrowColumns.INT8;
        case "short" -> ArrowColumns.INT16;
        case "integer" -> ArrowColumns.INT32;
        case "long" -> ArrowColumns.INT64;
        case "float" -> ArrowColumns.FLOAT32;
        case "double" -> ArrowColumns.FLOAT64;
        case "boolean" -> ArrowColumns.BOOL;
        case "string", "char", "varchar" -> ArrowColumns.UTF8;
        case "binary" -> ArrowColumns.BINARY;
        case "date" -> ArrowColumns.DATE32;
        case "timestamp" -> new ArrowType.Timestamp(TimeUnit.MICROSECOND, zone);
        case "timestamp_ntz" -> ArrowColumns.TIMESTAMP_US;
        case "void" -> ArrowColumns.NULL;
        default -> throw new IllegalArgumentException("Unsupported Spark Arrow type: " + type.typeName());
      };
    }
    return new Field(name, new FieldType(nullable, arrow, null), children);
  }

  static Object value(Object value, DataType type) {
    if (value == null) return null;
    if (type instanceof StructType structure) {
      Row row = (Row) value;
      Map<String, Object> fields = new LinkedHashMap<>();
      for (int i = 0; i < structure.size(); i++) fields.put(structure.apply(i).name(), value(row.get(i), structure.apply(i).dataType()));
      return fields;
    }
    if (type instanceof ArrayType array) {
      Iterable<?> items = value instanceof scala.collection.Seq<?> sequence
          ? scala.jdk.javaapi.CollectionConverters.asJava(sequence) : (Iterable<?>) value;
      List<Object> values = new ArrayList<>();
      for (Object item : items) values.add(value(item, array.elementType()));
      return values;
    }
    if (type instanceof MapType map) {
      Map<?, ?> items = value instanceof scala.collection.Map<?, ?> scalaMap
          ? scala.jdk.javaapi.CollectionConverters.asJava(scalaMap) : (Map<?, ?>) value;
      List<Object> entries = new ArrayList<>();
      for (var entry : items.entrySet()) {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("key", value(entry.getKey(), map.keyType()));
        fields.put("value", value(entry.getValue(), map.valueType()));
        entries.add(fields);
      }
      return entries;
    }
    if (type.equals(DataTypes.DateType)) {
      return Math.toIntExact((value instanceof java.sql.Date date ? date.toLocalDate() : (java.time.LocalDate) value).toEpochDay());
    }
    if (type.equals(DataTypes.TimestampType)) {
      java.time.Instant instant = value instanceof java.sql.Timestamp timestamp ? timestamp.toInstant() : (java.time.Instant) value;
      return Math.addExact(Math.multiplyExact(instant.getEpochSecond(), 1_000_000L), instant.getNano() / 1000L);
    }
    if (type.equals(DataTypes.TimestampNTZType)) {
      java.time.LocalDateTime local = value instanceof java.sql.Timestamp timestamp ? timestamp.toLocalDateTime() : (java.time.LocalDateTime) value;
      return Math.addExact(Math.multiplyExact(local.toEpochSecond(ZoneOffset.UTC), 1_000_000L), local.getNano() / 1000L);
    }
    return value;
  }
}
