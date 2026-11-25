// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.Objects;
import java.util.Map;
import com.nvidia.sdm.internal.DirectOptions;

/** Optional schema, entity-table name and inference/task options for a direct Arrow Tabular request. */
public final class RequestOptions {
  private final Map<String, Object> inference;
  private final Map<String, Object> taskOptions;
  private final Schema schema;
  private final String entityTable;

  private RequestOptions(Builder builder) {
    inference = DirectOptions.snapshot(builder.inference);
    taskOptions = DirectOptions.snapshot(builder.taskOptions);
    schema = builder.schema;
    entityTable = builder.entityTable;
  }

  public static Builder builder() { return new Builder(); }
  public Map<String, Object> inference() { return inference; }
  public Map<String, Object> taskOptions() { return taskOptions; }
  public Schema schema() { return schema; }
  public String entityTable() { return entityTable; }

  public static final class Builder {
    private Map<String, ?> inference = Map.of();
    private Map<String, ?> taskOptions = Map.of();
    private Schema schema;
    private String entityTable = "instance";
    private Builder() {}
    /** Explicit immutable schema; Rust checks it against the supplied tables. */
    public Builder schema(Schema schema) { this.schema = Objects.requireNonNull(schema); return this; }
    public Builder entityTable(String name) { entityTable = Objects.requireNonNull(name); return this; }
    /** JSON-family values, copied on build; fractional values are sent as float64. */
    public Builder inference(Map<String, ?> values) { inference = Objects.requireNonNull(values); return this; }
    public Builder taskOptions(Map<String, ?> values) { taskOptions = Objects.requireNonNull(values); return this; }
    public RequestOptions build() { return new RequestOptions(this); }
  }
}
