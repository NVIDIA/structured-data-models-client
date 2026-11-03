// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.DirectOptions;
import java.util.List;
import java.util.Map;

/** Sampling and inference options for supplied relational examples. */
public final class ExamplesOptions {
  final List<Integer> neighbors;
  final long seed;
  final Map<String,Object> inference;
  private ExamplesOptions(Builder b) { neighbors=b.neighbors==null?null:List.copyOf(b.neighbors); seed=b.seed; inference=DirectOptions.snapshot(b.inference); }
  public static Builder builder() { return new Builder(); }
  public static final class Builder {
    private List<Integer> neighbors;
    private long seed=42;
    private Map<String,Object> inference=Map.of();
    private Builder() {}
    public Builder numNeighbors(List<Integer> values) { neighbors=values; return this; }
    public Builder randomSeed(long value) { seed=value; return this; }
    public Builder inference(Map<String,Object> value) { inference=value; return this; }
    public ExamplesOptions build() { return new ExamplesOptions(this); }
  }
}
