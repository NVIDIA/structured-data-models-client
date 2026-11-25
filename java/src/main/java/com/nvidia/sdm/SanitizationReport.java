// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.List;
import java.util.Map;

/** Immutable graph preparation facts; unavailable does not mean zero drops. */
public record SanitizationReport(String status, List<TableSanitizationReport> tables) {
  public SanitizationReport {
    tables = List.copyOf(tables);
    if (!List.of("available", "unavailable", "not_applicable").contains(status)
        || (!status.equals("available") && !tables.isEmpty())) {
      throw new IllegalArgumentException("invalid sanitization report status or tables");
    }
  }

  static SanitizationReport unavailable() { return new SanitizationReport("unavailable", List.of()); }
  static SanitizationReport notApplicable() { return new SanitizationReport("not_applicable", List.of()); }

  static SanitizationReport prepared(Object value) {
    return new SanitizationReport("available", list(value).stream().map(item -> {
      Map<?, ?> row = object(item);
      return new TableSanitizationReport(text(row.get("table")),
          count(row, "input_rows"), count(row, "output_rows"), count(row, "null_primary_key_rows"),
          count(row, "duplicate_primary_key_rows"), count(row, "null_time_rows"));
    }).toList());
  }

  private static long count(Map<?, ?> row, String key) {
    Object value = row.get(key);
    if (!(value instanceof Number number) || !(value instanceof Long || value instanceof Integer) || number.longValue() < 0) {
      throw new IllegalArgumentException("invalid sanitization count");
    }
    return number.longValue();
  }

  Map<String, Object> saved() {
    return Map.of("status", status, "tables", tables.stream().map(row -> Map.of(
        "table", row.table(), "input_rows", row.inputRows(), "output_rows", row.outputRows(),
        "null_primary_key_rows", row.nullPrimaryKeyRows(),
        "duplicate_primary_key_rows", row.duplicatePrimaryKeyRows(), "null_time_rows", row.nullTimeRows())).toList());
  }

  private static List<?> list(Object value) {
    if (!(value instanceof List<?> list)) throw new IllegalArgumentException("invalid sanitization tables");
    return list;
  }

  private static Map<?, ?> object(Object value) {
    if (!(value instanceof Map<?, ?> map)) throw new IllegalArgumentException("invalid sanitization report");
    return map;
  }

  private static String text(Object value) {
    if (!(value instanceof String text)) throw new IllegalArgumentException("invalid sanitization text");
    return text;
  }

  static SanitizationReport restored(Object value) throws java.io.IOException {
    Map<String, Object> saved = Snapshots.object(value);
    try {
      return new SanitizationReport(Snapshots.text(saved.get("status")), prepared(saved.get("tables")).tables());
    } catch (IllegalArgumentException error) {
      throw Snapshots.invalid("invalid sanitization report", error);
    }
  }
}
