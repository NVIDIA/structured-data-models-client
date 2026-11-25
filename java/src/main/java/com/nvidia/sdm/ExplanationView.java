// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowFrames;
import com.nvidia.sdm.internal.ArrowTabular;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Json;
import java.util.*;
import java.nio.charset.StandardCharsets;

/** Immutable snapshot of an explanation; field validation is demand-driven. */
public final class ExplanationView {
  private final Map<String, Object> raw;

  public ExplanationView(Map<String, ?> raw) {
    this.raw = mapping(freeze(raw, "$", new IdentityHashMap<>(), 0), "$");
  }

  private static StructuredDataException invalid(String path) {
    return new StructuredDataException("Malformed explanation field", "INVALID_REQUEST", Map.of("path", path), null);
  }

  private static Object freeze(Object value, String path, IdentityHashMap<Object, Boolean> seen, int depth) {
    if (depth > 128 || seen.containsKey(value)) throw invalid(path);
    if (value == null || value instanceof String || value instanceof Boolean
        || value instanceof Byte || value instanceof Short || value instanceof Integer
        || value instanceof Long || value instanceof Float || value instanceof Double
        || value instanceof java.math.BigInteger || value instanceof java.math.BigDecimal) return value;
    seen.put(value, true);
    try {
      if (value instanceof Map<?, ?> map) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
          if (!(entry.getKey() instanceof String key)) throw invalid(path);
          copy.put(key, freeze(entry.getValue(), path + "." + key, seen, depth + 1));
        }
        return Collections.unmodifiableMap(copy);
      }
      if (value instanceof List<?> list) {
        List<Object> copy = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) copy.add(freeze(list.get(i), path + "[" + i + "]", seen, depth + 1));
        return Collections.unmodifiableList(copy);
      }
      throw invalid(path);
    } finally { seen.remove(value); }
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> mapping(Object value, String path) {
    if (!(value instanceof Map<?, ?>)) throw invalid(path);
    return (Map<String, Object>) value;
  }

  public Map<String, Object> raw() { return raw; }
  public Map<String, Object> details() {
    Object value = raw.get("details");
    return value == null ? null : mapping(value, "$.details");
  }

  private record Field(Object value, String path) {}
  private Field prose(String... names) {
    for (String name : names) if (raw.containsKey(name)) return new Field(raw.get(name), "$." + name);
    if (raw.get("details") instanceof Map<?, ?> nested) {
      for (String name : names) if (nested.containsKey(name)) return new Field(nested.get(name), "$.details." + name);
    }
    return new Field(null, "$");
  }

  public String summary() {
    Field field = prose("summary");
    if (field.value() != null && !(field.value() instanceof String)) throw invalid(field.path());
    return (String) field.value();
  }

  public List<String> warnings() {
    Field field = prose("warnings", "warning");
    if (field.value() == null) return null;
    if (field.path().endsWith(".warning")) {
      if (!(field.value() instanceof String text)) throw invalid(field.path());
      return List.of(text);
    }
    if (!(field.value() instanceof List<?> list)) throw invalid(field.path());
    List<String> values = new ArrayList<>();
    for (Object value : list) {
      if (!(value instanceof String text)) throw invalid(field.path());
      values.add(text);
    }
    return List.copyOf(values);
  }

  private List<?> sequence(String name) {
    boolean nested = "kumo_rfm_v2_1".equals(raw.get("format"));
    Map<String, Object> owner = nested ? details() : raw;
    Object value = owner == null ? null : owner.get(name);
    if (value != null && !(value instanceof List<?>)) throw invalid((nested ? "$.details." : "$.") + name);
    return (List<?>) value;
  }

  public List<?> cohorts() { return sequence("cohorts"); }
  public List<?> subgraphs() { return sequence("subgraphs"); }

  public String toText() {
    List<String> lines = new ArrayList<>();
    String summary = summary(); if (summary != null) lines.add(summary);
    List<String> warnings = warnings();
    if (warnings != null) for (String warning : warnings) lines.add("Warning: " + warning);
    return String.join("\n", lines);
  }

  /** Aggregate supported scores lazily with the shared numerical policy. */
  public FeatureImportance featureImportance(boolean normalize) {
    if (!"kumo_rfm_v2_1".equals(raw.get("format"))) return null;
    List<?> graphs = subgraphs(); if (graphs == null) return null;
    List<Object> cells = new ArrayList<>();
    for (int i = 0; i < graphs.size(); i++) {
      String path = "$.details.subgraphs[" + i + "]";
      Map<String, Object> tables = mapping(mapping(graphs.get(i), path).get("tables"), path + ".tables");
      for (var table : tables.entrySet()) {
        String tablePath = path + ".tables." + table.getKey();
        for (var node : mapping(table.getValue(), tablePath).entrySet()) {
          String nodePath = tablePath + "." + node.getKey();
          Map<String, Object> columns = mapping(mapping(node.getValue(), nodePath).get("cells"), nodePath + ".cells");
          for (var column : columns.entrySet()) {
            String cellPath = nodePath + ".cells." + column.getKey();
            Object score = mapping(column.getValue(), cellPath).get("score");
            if (score == null) continue;
            String encoded;
            // Nonfinite spellings remain distinguishable and are rejected by Rust.
            if (score instanceof Number) encoded = score.toString();
            else try { encoded = new String(Json.write(score instanceof String || score instanceof Boolean ? score : Map.of("invalid", true)), StandardCharsets.UTF_8); }
            catch (Json.JsonException e) { throw invalid(cellPath + ".score"); }
            cells.add(Map.of("table", table.getKey(), "column", column.getKey(), "score", encoded, "path", cellPath + ".score"));
          }
        }
      }
    }
    try (Host.Finished finished = ArrowFrames.run(Map.of("op", "feature_importance", "input", 1),
        Map.of(1, Host.attachment(Map.of("cells", cells, "normalize", normalize))), Map.of(), Host.Call.none())) {
      Map<String, Object> output = ArrowTabular.value(finished);
      List<FeatureImportanceRow> rows = new ArrayList<>();
      for (Object value : (List<?>) output.get("rows")) {
        Map<String, Object> row = mapping(value, "rows");
        rows.add(new FeatureImportanceRow((String) row.get("table"), (String) row.get("column"), ((Number) row.get("score")).doubleValue()));
      }
      return new FeatureImportance(rows, normalize);
    }
  }
}
