// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Owned JSON-family values and their lossless native option representation. */
public final class DirectOptions {
  private DirectOptions() {}

  @SuppressWarnings("unchecked")
  public static Map<String, Object> snapshot(Map<String, ?> values) {
    try { return (Map<String, Object>) freeze(Json.read(Json.write(values))); }
    catch (Json.JsonException e) {
      throw new StructuredDataException("options must contain JSON values", "INVALID_REQUEST", Map.of(), e);
    }
  }

  private static Object freeze(Object value) {
    if (value instanceof Map<?, ?> map) {
      Map<String, Object> copy = new LinkedHashMap<>();
      map.forEach((key, item) -> copy.put((String) key, freeze(item)));
      return Collections.unmodifiableMap(copy);
    }
    if (value instanceof List<?> list) {
      List<Object> copy = new ArrayList<>();
      list.forEach(item -> copy.add(freeze(item)));
      return Collections.unmodifiableList(copy);
    }
    return value;
  }

  public static Map<String, Object> encode(Map<String, Object> values) {
    return Map.of("entries", entries(values));
  }

  private static List<Object> entries(Map<?, ?> values) {
    List<Object> entries = new ArrayList<>();
    values.forEach((key, value) -> entries.add(List.of(key, encodeValue(value))));
    return entries;
  }

  private static Object encodeValue(Object value) {
    if (value == null) return Map.of("value", Map.of("t", "null"));
    if (value instanceof Boolean) return Map.of("value", Map.of("t", "bool", "v", value));
    if (value instanceof String) return Map.of("value", Map.of("t", "utf8", "v", value));
    if (value instanceof Map<?, ?> map) return Map.of("object", entries(map));
    if (value instanceof List<?> list) return Map.of("array", list.stream().map(DirectOptions::encodeValue).toList());
    if (value instanceof java.math.BigDecimal || value instanceof Double || value instanceof Float)
      return Map.of("value", Map.of("t", "f64", "v", value.toString()));
    return Map.of("integer", value.toString());
  }
}
