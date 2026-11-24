// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.provider;

import java.util.EnumSet;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;

/**
 * What a {@link com.nvidia.sdm.Source} answers, and how.
 *
 * <p>Every provider answers {@code describe} and {@code scan}. Each other request is answered by
 * the provider when declared in {@link #optional}; otherwise the core answers it itself, in
 * memory, from the rows a {@code scan} reads, within {@link #fallback}'s limits, or refuses it
 * with {@code UNSUPPORTED_REQUEST} when no fallback is declared.
 *
 * @param optional the optional requests the provider answers itself
 * @param maxKeysPerRequest the most seeds one {@code neighbors} or {@code lookup} may carry; more
 *     reach the provider as several requests, whose answers the core joins. Empty is no limit.
 * @param sampling whether a sample drawn with the same seed is the same rows every time
 * @param maxSeed the largest sampling seed the provider takes
 * @param nullOrder where the provider puts nulls when it orders rows; empty, and the core never
 *     asks it for rows in an order
 * @param fallback the limits within which the core answers undeclared optional requests in
 *     memory; empty declares no fallback
 * @param filters the value filters the provider applies to a sample or scan; the core sends it no
 *     other kind, and checks every row it keeps itself, so a provider may return more rows than a
 *     value filter holds for but never fewer
 */
public record Capabilities(
    Set<OptionalRequest> optional,
    OptionalLong maxKeysPerRequest,
    Sampling sampling,
    long maxSeed,
    java.util.Optional<NullOrder> nullOrder,
    java.util.Optional<Fallback> fallback,
    Set<DataRequest.FilterKind> filters) {

  public Capabilities {
    optional = Set.copyOf(optional);
    filters = Set.copyOf(filters);
    Objects.requireNonNull(maxKeysPerRequest, "maxKeysPerRequest");
    Objects.requireNonNull(sampling, "sampling");
    Objects.requireNonNull(nullOrder, "nullOrder");
    Objects.requireNonNull(fallback, "fallback");
    if (maxKeysPerRequest.isPresent() && maxKeysPerRequest.getAsLong() < 1) {
      throw new IllegalArgumentException("maxKeysPerRequest must be at least 1");
    }
    if (maxSeed < 0) {
      throw new IllegalArgumentException("maxSeed must not be negative");
    }
  }

  /** Capabilities applying no value filter, as declared before value filters existed. */
  public Capabilities(
      Set<OptionalRequest> optional,
      OptionalLong maxKeysPerRequest,
      Sampling sampling,
      long maxSeed,
      java.util.Optional<NullOrder> nullOrder,
      java.util.Optional<Fallback> fallback) {
    this(optional, maxKeysPerRequest, sampling, maxSeed, nullOrder, fallback, Set.of());
  }

  /** Whether a sample drawn with the same seed is the same rows every time. */
  public enum Sampling {
    SEEDED_STABLE,
    UNSEEDED
  }

  /** Where nulls go when rows are ordered. */
  public enum NullOrder {
    FIRST,
    LAST
  }

  /** The most rows and bytes the core reads to answer a request in memory. */
  public record Fallback(long maxRows, long maxBytes) {
    public Fallback {
      if (maxRows < 1 || maxBytes < 1) {
        throw new IllegalArgumentException("fallback limits must be positive");
      }
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Starts from: no optional requests, no key limit, seeded-stable sampling, any non-negative
   * {@code long} seed, nulls last, no fallback, no value filters.
   */
  public static final class Builder {
    private final Set<OptionalRequest> optional = EnumSet.noneOf(OptionalRequest.class);
    private OptionalLong maxKeys = OptionalLong.empty();
    private Sampling sampling = Sampling.SEEDED_STABLE;
    private long maxSeed = Long.MAX_VALUE;
    private java.util.Optional<NullOrder> nullOrder = java.util.Optional.of(NullOrder.LAST);
    private java.util.Optional<Fallback> fallback = java.util.Optional.empty();
    private final Set<DataRequest.FilterKind> filters = EnumSet.noneOf(DataRequest.FilterKind.class);

    private Builder() {}

    /** Declares the value filters the provider applies to a sample or scan. */
    public Builder filters(DataRequest.FilterKind... kinds) {
      filters.addAll(java.util.List.of(kinds));
      return this;
    }

    public Builder answers(OptionalRequest... requests) {
      optional.addAll(java.util.List.of(requests));
      return this;
    }

    public Builder maxKeysPerRequest(long max) {
      maxKeys = OptionalLong.of(max);
      return this;
    }

    public Builder sampling(Sampling value) {
      sampling = Objects.requireNonNull(value);
      return this;
    }

    public Builder maxSeed(long value) {
      maxSeed = value;
      return this;
    }

    public Builder nullOrder(NullOrder value) {
      nullOrder = java.util.Optional.ofNullable(value);
      return this;
    }

    public Builder fallback(long maxRows, long maxBytes) {
      fallback = java.util.Optional.of(new Fallback(maxRows, maxBytes));
      return this;
    }

    public Capabilities build() {
      return new Capabilities(optional, maxKeys, sampling, maxSeed, nullOrder, fallback, filters);
    }
  }
}
