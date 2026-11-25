// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * How a request's context examples were drawn from the source; unavailable where the request does
 * not say (context the caller gave, or a request saved before reports were kept).
 *
 * @param status {@code available} or {@code unavailable}
 * @param parts one draw per part: one for a request, one per forecast step or ranking question
 */
public record ContextReport(String status, List<ContextDraw> parts) {
  public ContextReport {
    parts = List.copyOf(parts);
    if (!List.of("available", "unavailable").contains(status)
        || (!status.equals("available") && !parts.isEmpty())) {
      throw new IllegalArgumentException("invalid context report status or parts");
    }
  }

  static ContextReport unavailable() {
    return new ContextReport("unavailable", List.of());
  }

  /** The core's report of each part, or unavailable where it says nothing. */
  static ContextReport reported(Object value) {
    if (value == null) {
      return unavailable();
    }
    if (!(value instanceof List<?> list)) {
      throw new IllegalArgumentException("invalid context report");
    }
    if (list.isEmpty()) {
      return unavailable();
    }
    return new ContextReport("available", list.stream().map(ContextReport::draw).toList());
  }

  private static ContextDraw draw(Object item) {
    if (!(item instanceof Map<?, ?> row)) {
      throw new IllegalArgumentException("invalid context draw");
    }
    if (!(row.get("stopped") instanceof String stopped)
        || !(row.get("filtered_at_source") instanceof Boolean filtered)) {
      throw new IllegalArgumentException("invalid context draw");
    }
    return new ContextDraw(
        count(row, "draws"),
        count(row, "rows_drawn"),
        count(row, "candidates"),
        count(row, "kept"),
        count(row, "asked"),
        stopped,
        filtered);
  }

  private static long count(Map<?, ?> row, String key) {
    Object value = row.get(key);
    if (!(value instanceof Number number)
        || !(value instanceof Long || value instanceof Integer)
        || number.longValue() < 0) {
      throw new IllegalArgumentException("invalid context draw count");
    }
    return number.longValue();
  }

  Map<String, Object> saved() {
    return Map.of(
        "status",
        status,
        "parts",
        parts.stream()
            .map(
                part -> {
                  Map<String, Object> row = new LinkedHashMap<>();
                  row.put("draws", part.draws());
                  row.put("rows_drawn", part.rowsDrawn());
                  row.put("candidates", part.candidates());
                  row.put("kept", part.kept());
                  row.put("asked", part.asked());
                  row.put("stopped", part.stopped());
                  row.put("filtered_at_source", part.filteredAtSource());
                  return row;
                })
            .toList());
  }

  static ContextReport restored(Object value) {
    if (!(value instanceof Map<?, ?> saved) || !(saved.get("status") instanceof String status)) {
      throw new IllegalArgumentException("invalid saved context report");
    }
    if (status.equals("unavailable")) {
      return unavailable();
    }
    return reported(saved.get("parts"));
  }
}
