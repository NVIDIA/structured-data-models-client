// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;

/** Mechanical decoding of the declared Arrow/PrimitiveOption representations. */
public final class ItemValues {
  private ItemValues() {}

  public static StructuredDataException invalid(String message) {
    return new StructuredDataException(message, "INTERNAL_ERROR", Map.of(), null);
  }

  public static Object primitive(String text) {
    try {
      return primitive(Json.read(text.getBytes(StandardCharsets.UTF_8)));
    } catch (Json.JsonException error) {
      throw invalid("projected result value is not valid JSON");
    }
  }

  private static Object primitive(Object encoded) {
    if (!(encoded instanceof Map<?, ?> map) || map.size() != 1)
      throw invalid("invalid projected primitive");
    if (map.containsKey("value")) return Wording.scalar(map.get("value"));
    if (map.get("integer") instanceof String text) {
      BigInteger n = new BigInteger(text);
      return n.bitLength() < 64 ? n.longValue() : n;
    }
    if (map.get("array") instanceof List<?> values) {
      List<Object> result = new ArrayList<>();
      for (Object v : values) result.add(primitive(v));
      return result;
    }
    if (map.get("object") instanceof List<?> values) {
      Map<Object, Object> result = new LinkedHashMap<>();
      for (Object value : values) {
        if (!(value instanceof List<?> pair)
            || pair.size() != 2
            || !(pair.get(0) instanceof String)) throw invalid("invalid projected object");
        result.put(pair.get(0), primitive(pair.get(1)));
      }
      return result;
    }
    throw invalid("unknown projected primitive");
  }

  public static Map<Object, Object> mapping(Object value) {
    if (value == null) return null;
    Map<Object, Object> result = new LinkedHashMap<>();
    if (value instanceof Map<?, ?> map) {
      map.forEach(result::put);
      return result;
    }
    if (!(value instanceof List<?> pairs))
      throw invalid("projected mapping is not pairs or object");
    for (Object pair : pairs) {
      if (!(pair instanceof List<?> entries) || entries.size() != 2)
        throw invalid("projected mapping pair has wrong length");
      result.put(entries.get(0), entries.get(1));
    }
    return result;
  }

  public static Object nativeValue(Field field, Object value) {
    if (value == null) return null;
    if (field.getType() instanceof ArrowType.Map) {
      if (!(value instanceof List<?> entries)) throw invalid("Arrow map has invalid entries");
      Field struct = field.getChildren().get(0);
      Field key = struct.getChildren().get(0), item = struct.getChildren().get(1);
      Map<Object, Object> result = new LinkedHashMap<>();
      for (Object entry : entries) {
        if (!(entry instanceof Map<?, ?> map)) throw invalid("Arrow map entry is not a struct");
        result.put(
            nativeValue(key, map.get(key.getName())), nativeValue(item, map.get(item.getName())));
      }
      return result;
    }
    if (field.getType() instanceof ArrowType.List
        || field.getType() instanceof ArrowType.LargeList) {
      if (!(value instanceof List<?> values)) throw invalid("Arrow list has invalid values");
      List<Object> result = new ArrayList<>();
      for (Object item : values) result.add(nativeValue(field.getChildren().get(0), item));
      return result;
    }
    if (value instanceof org.apache.arrow.vector.util.Text) return value.toString();
    return value;
  }

  public static Object freeze(Object value) {
    if (value instanceof Map<?, ?> map) {
      Map<Object, Object> copy = new LinkedHashMap<>();
      map.forEach((key, item) -> copy.put(freeze(key), freeze(item)));
      return Collections.unmodifiableMap(copy);
    }
    if (value instanceof List<?> list) {
      List<Object> copy = new ArrayList<>();
      for (Object item : list) copy.add(freeze(item));
      return Collections.unmodifiableList(copy);
    }
    return value;
  }
}
