// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.DirectOptions;
import java.util.*;

/** Named relational inference options. Rust supplies defaults and validates task-specific values. */
public final class InferenceConfig {
  private final Map<String,Object> values;
  private InferenceConfig(Builder builder) { values = DirectOptions.snapshot(builder.values); }
  public static Builder builder() { return new Builder(); }
  /** Immutable plain options; absent fields remain absent. */
  public Map<String,Object> toMap() { return values; }
  public static final class Builder {
    private final Map<String,Object> values = new LinkedHashMap<>();
    private Builder() {}
    public Builder numEstimators(int value) { values.put("num_estimators", value); return this; }
    public Builder columnShuffle(boolean value) { values.put("column_shuffle", value); return this; }
    public Builder categoryShuffle(boolean value) { values.put("category_shuffle", value); return this; }
    public Builder hopShuffle(boolean value) { values.put("hop_shuffle", value); return this; }
    public Builder classShuffle(boolean value) { values.put("class_shuffle", value); return this; }
    /** Null entries mean no transform; Rust checks transform names. */
    public Builder targetTransforms(List<String> value) { values.put("target_transforms", new ArrayList<>(Objects.requireNonNull(value))); return this; }
    public Builder outputType(String value) { values.put("output_type", Objects.requireNonNull(value)); return this; }
    public InferenceConfig build() { return new InferenceConfig(this); }
  }
}
