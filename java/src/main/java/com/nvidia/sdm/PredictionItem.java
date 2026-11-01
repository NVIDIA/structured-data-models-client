// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.ItemValues;
import java.util.List;
import java.util.Map;

/** One semantic answer, with original ranking candidate values and immutable nested values. */
public final class PredictionItem {
  private final Map<String, Object> fields;

  @SuppressWarnings("unchecked")
  PredictionItem(Map<String, Object> fields) {
    this.fields = (Map<String, Object>) ItemValues.freeze(fields);
  }

  public long rowIndex() {
    return ((Number) fields.get("row_index")).longValue();
  }

  public Object id() {
    return fields.get("id");
  }

  public Long forecastStep() {
    return fields.get("forecast_step") == null
        ? null
        : ((Number) fields.get("forecast_step")).longValue();
  }

  public Object prediction() {
    return fields.get("prediction");
  }

  public Map<Object, Object> probabilities() {
    return mapping("probabilities");
  }

  public Map<Object, Object> metadata() {
    return mapping("metadata");
  }

  public Map<Object, Object> explanation() {
    return mapping("explanation");
  }

  public Map<Object, Object> quantiles() {
    return mapping("quantiles");
  }

  @SuppressWarnings("unchecked")
  public List<Double> scores() {
    return (List<Double>) fields.get("scores");
  }

  @SuppressWarnings("unchecked")
  public List<Double> embeddings() {
    return (List<Double>) fields.get("embeddings");
  }

  @SuppressWarnings("unchecked")
  public List<Map<Object, Object>> rankings() {
    return (List<Map<Object, Object>>) fields.get("rankings");
  }

  /** All carried semantic fields; missing output fields are absent. */
  public Map<String, Object> fields() {
    return fields;
  }

  @SuppressWarnings("unchecked")
  private Map<Object, Object> mapping(String name) {
    return (Map<Object, Object>) fields.get(name);
  }
}
