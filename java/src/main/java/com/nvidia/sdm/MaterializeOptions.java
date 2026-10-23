// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * What {@link PredictiveQuery#materialize} gathers, as Python's {@code materialize(...)} takes it:
 * the entities to predict for ({@code indices}), the anchor time, the context size, and the
 * sampling options, each passed to the core as given, which takes or refuses it by name.
 *
 * <pre>{@code
 * MaterializeOptions options = MaterializeOptions.builder()
 *     .indices(List.of(1, 2, 3))
 *     .anchorTime(LocalDate.of(2024, 3, 1))
 *     .option("num_neighbors", List.of(8, 8))
 *     .build();
 * }</pre>
 *
 * <p>An option is a name and a plain value (a number, text, a boolean, null, a {@code List} or
 * {@code Map} of them), as Python's keyword arguments are: {@code random_seed}, {@code num_hops},
 * {@code num_neighbors}, {@code max_pq_iterations}, {@code use_prediction_time}, {@code
 * return_embeddings}, {@code exclude_cols_dict}, {@code prediction_statistic}, {@code
 * inference_config}, {@code rows_per_call}, {@code lag_timesteps}, {@code top_k}; one the core does
 * not read is refused by it, {@code INVALID_REQUEST}.
 */
public final class MaterializeOptions {
  /** The anchor time each entity is asked about at its own time ({@code anchor_time='entity'}). */
  static final Object EACH_ENTITY = new Object();

  private final List<Object> indices;
  private final Object anchorTime;
  private final Integer numContextSamples;
  private final Object contextAnchorTime;
  private final boolean contextAnchorGiven;
  private final Map<String, Object> options;

  private MaterializeOptions(Builder builder) {
    this.indices = builder.indices == null ? null : Collections.unmodifiableList(builder.indices);
    this.anchorTime = builder.anchorTime;
    this.numContextSamples = builder.numContextSamples;
    this.contextAnchorTime = builder.contextAnchorTime;
    this.contextAnchorGiven = builder.contextAnchorGiven;
    this.options = Collections.unmodifiableMap(new LinkedHashMap<>(builder.options));
  }

  public static Builder builder() {
    return new Builder();
  }

  List<Object> indices() {
    return indices;
  }

  Object anchorTime() {
    return anchorTime;
  }

  Integer numContextSamples() {
    return numContextSamples;
  }

  Object contextAnchorTime() {
    return contextAnchorTime;
  }

  boolean contextAnchorGiven() {
    return contextAnchorGiven;
  }

  Map<String, Object> options() {
    return options;
  }

  /** Builds {@link MaterializeOptions}. */
  public static final class Builder {
    private List<Object> indices;
    private Object anchorTime;
    private Integer numContextSamples;
    private Object contextAnchorTime;
    private boolean contextAnchorGiven;
    private final Map<String, Object> options = new LinkedHashMap<>();

    private Builder() {}

    /**
     * The entities to predict for, by their keys' values (a {@code Long}, text, a {@code
     * BigDecimal}, ...), as Python's {@code indices}; the core casts them to the key's type.
     */
    public Builder indices(List<?> indices) {
      this.indices = new ArrayList<>(Objects.requireNonNull(indices, "indices"));
      return this;
    }

    /** The entities to predict for. */
    public Builder indices(Object... indices) {
      return indices(Arrays.asList(indices));
    }

    /** The anchor time, a wall-clock time with no zone, as a {@code pandas.Timestamp} holds one. */
    public Builder anchorTime(LocalDateTime anchorTime) {
      this.anchorTime = Objects.requireNonNull(anchorTime, "anchorTime");
      return this;
    }

    /** The anchor time at the start of {@code anchorTime}'s day. */
    public Builder anchorTime(LocalDate anchorTime) {
      return anchorTime(anchorTime.atStartOfDay());
    }

    /** The anchor time, an instant in a zone. */
    public Builder anchorTime(ZonedDateTime anchorTime) {
      this.anchorTime = Objects.requireNonNull(anchorTime, "anchorTime");
      return this;
    }

    /** Each entity asked about at its own time, as Python's {@code anchor_time='entity'}. */
    public Builder anchorAtEachEntity() {
      this.anchorTime = EACH_ENTITY;
      return this;
    }

    /** The most labelled context rows to use. */
    public Builder numContextSamples(int numContextSamples) {
      this.numContextSamples = numContextSamples;
      return this;
    }

    /** The context's anchor time, a wall-clock time with no zone. */
    public Builder contextAnchorTime(LocalDateTime contextAnchorTime) {
      this.contextAnchorTime = Objects.requireNonNull(contextAnchorTime, "contextAnchorTime");
      this.contextAnchorGiven = true;
      return this;
    }

    /** The context's anchor time at the start of its day. */
    public Builder contextAnchorTime(LocalDate contextAnchorTime) {
      return contextAnchorTime(contextAnchorTime.atStartOfDay());
    }

    /** The context anchored at each entity's own time ({@code context_anchor_time='entity'}). */
    public Builder contextAnchorAtEachEntity() {
      this.contextAnchorTime = EACH_ENTITY;
      this.contextAnchorGiven = true;
      return this;
    }

    /** The sampling option {@code name}, with a plain value, passed to the core as given. */
    public Builder option(String name, Object value) {
      options.put(Objects.requireNonNull(name, "name"), value);
      return this;
    }

    /** The seed of the sample ({@code random_seed}). */
    public Builder randomSeed(long seed) {
      return option("random_seed", seed);
    }

    /** How many entities one call scores ({@code rows_per_call}). */
    public Builder rowsPerCall(long rows) {
      return option("rows_per_call", rows);
    }

    public MaterializeOptions build() {
      return new MaterializeOptions(this);
    }
  }
}
