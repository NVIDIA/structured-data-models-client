// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.LinkedHashMap;
import java.util.Map;

/** Controls bounded ground-truth extraction without model inference. */
public final class LabelsOptions {
  private final long size;
  private final long maxIterations;
  private final Object anchorTime;
  private final Long randomSeed;

  private LabelsOptions(Builder builder) {
    size = builder.size;
    maxIterations = builder.maxIterations;
    anchorTime = builder.anchorTime;
    randomSeed = builder.randomSeed;
    if (size < 0 || maxIterations <= 0) throw new StructuredDataException(
        "size must be nonnegative and maxIterations positive", "INVALID_REQUEST", Map.of(), null);
  }

  public static Builder builder(long size) { return new Builder(size); }

  Map<String, Object> input(String query) {
    Map<String, Object> input = new LinkedHashMap<>();
    input.put("query", query);
    input.put("size", size);
    input.put("max_iterations", maxIterations);
    input.put("anchor_entity", anchorTime == MaterializeOptions.EACH_ENTITY);
    input.put("anchor_time", anchorTime == MaterializeOptions.EACH_ENTITY ? null : PredictiveQuery.timestamp(anchorTime));
    input.put("seed", PredictiveQuery.seed(randomSeed));
    input.put("unseeded", randomSeed == null);
    return input;
  }

  /** Defaults to historical anchor selection, seed 42 and ten candidate trials per output row. */
  public static final class Builder {
    private final long size;
    private long maxIterations = 10;
    private Object anchorTime;
    private Long randomSeed = 42L;
    private Builder(long size) { this.size = size; }
    public Builder maxIterations(long value) { maxIterations = value; return this; }
    /** A supported Java time value, interpreted as by MaterializeOptions.anchorTime. */
    public Builder anchorTime(Object value) { anchorTime = value; return this; }
    /** Use each entity's own timestamp, for static targets only. */
    public Builder entityTime() { anchorTime = MaterializeOptions.EACH_ENTITY; return this; }
    /** A nonnegative seed, or null for a fresh sample. */
    public Builder randomSeed(long value) { randomSeed = value; return this; }
    public Builder randomSeed(Long value) { randomSeed = value; return this; }
    public LabelsOptions build() { return new LabelsOptions(this); }
  }
}
