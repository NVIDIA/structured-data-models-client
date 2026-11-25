// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.util.*;

/** Test-only presentation of public semantic items alongside retained raw Arrow evidence. */
final class ParityResults {
  private ParityResults() {}

  static final List<String> FIELDS =
      List.of(
          "row_index",
          "id",
          "forecast_step",
          "prediction",
          "probabilities",
          "scores",
          "rankings",
          "metadata",
          "explanation",
          "quantiles",
          "embeddings");

  static Map<String, Object> frame(PredictionResult result) {
    var rows = new ArrayList<Map<String, Object>>();
    var columns = new LinkedHashSet<String>();
    for (var item : result.items()) {
      var row = new LinkedHashMap<String, Object>();
      for (String field : List.of("row_index", "id", "forecast_step", "prediction")) {
        Object value = item.fields().get(field);
        if (value != null) row.put(field, ParityRunner.scalar(value));
      }
      for (String field : List.of("probabilities", "quantiles")) {
        Object value = item.fields().get(field);
        if (value instanceof Map<?, ?> mapping)
          mapping.forEach(
              (key, cell) ->
                  row.put(
                      (field.equals("probabilities") ? "probability_" : "quantile_") + key,
                      ParityRunner.scalar(cell)));
      }
      for (String field : List.of("scores", "embeddings", "rankings")) {
        Object value = item.fields().get(field);
        if (value != null && !(field.equals("rankings") && ((List<?>) value).isEmpty()))
          row.put(field, ParityRunner.scalar(value));
      }
      columns.addAll(row.keySet());
      rows.add(row);
    }
    rows.sort(Comparator.comparingLong(row -> ((Number) row.get("row_index")).longValue()));
    return Map.of(
        "columns",
        new ArrayList<>(columns),
        "rows",
        rows.stream().map(row -> columns.stream().map(row::get).toList()).toList());
  }

  static Map<String, Object> details(PredictionResult result) {
    var details = new LinkedHashMap<String, Object>();
    details.put("model", result.model());
    details.put("request_id", result.requestId());
    details.put("outputs", result.outputs());
    details.put(
        "items",
        result.items().stream()
            .map(
                item -> {
                  var fields = new LinkedHashMap<String, Object>();
                  FIELDS.forEach(
                      field -> fields.put(field, ParityRunner.scalar(item.fields().get(field))));
                  return fields;
                })
            .toList());
    details.put("metadata", ParityRunner.scalar(result.metadata()));
    details.put("explanation", ParityRunner.scalar(result.explanation()));
    details.put("unscored_candidates", ParityRunner.scalar(result.unscoredCandidates()));
    return details;
  }
}
