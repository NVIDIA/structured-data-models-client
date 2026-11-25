// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Explicit meaning of supplied relational examples; the Rust engine validates it. */
public record RelationalTask(Kind kind, String entityTable, String linkTargetTable, Long stepSizeNs, Integer numForecasts) {
  public enum Kind { REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, TEMPORAL_LINK_PREDICTION, FORECASTING }
  public RelationalTask { Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(entityTable, "entityTable"); }
  public static RelationalTask regression(String entity) { return new RelationalTask(Kind.REGRESSION,entity,null,null,null); }
  public static RelationalTask binary(String entity) { return new RelationalTask(Kind.BINARY_CLASSIFICATION,entity,null,null,null); }
  public static RelationalTask multiclass(String entity) { return new RelationalTask(Kind.MULTICLASS_CLASSIFICATION,entity,null,null,null); }
  public static RelationalTask link(String entity,String destination) { return new RelationalTask(Kind.TEMPORAL_LINK_PREDICTION,entity,destination,null,null); }
  public static RelationalTask forecast(String entity,long stepSizeNs,int numForecasts) { return new RelationalTask(Kind.FORECASTING,entity,null,stepSizeNs,numForecasts); }
  Map<String,Object> contract() {
    Map<String,Object> value = new LinkedHashMap<>();
    value.put("kind",kind.name().toLowerCase(java.util.Locale.ROOT)); value.put("entity_table",entityTable);
    value.put("link_target_table",linkTargetTable); value.put("step_size_ns",stepSizeNs); value.put("num_forecasts",numForecasts);
    return value;
  }
}
