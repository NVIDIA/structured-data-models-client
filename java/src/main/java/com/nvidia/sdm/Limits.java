// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.LimitSettings;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The size limits a client holds a request to before it is sent, as Python's {@code
 * sdm_client.Limits}. Each mirrors a limit the serving layer enforces ({@code TFM_MAX_TRAIN_ROWS},
 * {@code TFM_MAX_TEST_ROWS}, {@code TFM_MAX_PAYLOAD_BYTES}), which a deployment sets, so each is a
 * setting with the serving layer's default.
 *
 * <pre>{@code
 * Limits limits = Limits.builder()
 *     .maxTabularContextRows(200_000)
 *     .maxRequestBytes(512L * 1024 * 1024)
 *     .build();
 * SdmClient client = SdmClient.connect(url, ClientOptions.builder().limits(limits).build());
 * Request request = query.materialize(MaterializeOptions.builder().indices(keys).limits(limits).build());
 * }</pre>
 *
 * <p>A setting not set here takes its environment variable, {@code SDM_CLIENT_} and its name in
 * capitals (such as {@code SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS}), else its default: 10,000 context
 * rows for either model, 10,000 rows scored per relational call, 200 per temporal link prediction
 * call, and 31,457,280 bytes (30 MiB) per call. The core resolves them, as it does for every host.
 */
public final class Limits {
  private final Map<String, Long> values;

  private Limits(Map<String, Long> values) {
    this.values = Collections.unmodifiableMap(new LinkedHashMap<>(values));
  }

  /** No setting set: each takes its environment variable, else its default. */
  public static Limits defaults() {
    return new Limits(Map.of());
  }

  public static Builder builder() {
    return new Builder();
  }

  /** The settings set here, by name, such as {@code max_request_bytes}; none for a default. */
  public Map<String, Long> values() {
    return values;
  }

  /**
   * What these limits come to, each setting by name: the value set here, else its environment
   * variable's, else its default, as the core resolves them for every request.
   *
   * @throws StructuredDataException {@code INVALID_CONFIGURATION} when an environment variable is
   *     not a whole number of at least 1
   */
  public Map<String, Long> resolved() {
    return LimitSettings.resolved(values);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Limits limits && limits.values.equals(values);
  }

  @Override
  public int hashCode() {
    return values.hashCode();
  }

  @Override
  public String toString() {
    return "Limits" + values;
  }

  /** Builds {@link Limits}; each value is a whole number of at least 1. */
  public static final class Builder {
    private final Map<String, Long> values = new LinkedHashMap<>();

    private Builder() {}

    /** The most context rows a relational request carries ({@code num_context_samples}' most). */
    public Builder maxRelationalContextRows(long rows) {
      return set("max_relational_context_rows", rows);
    }

    /** The most rows one relational call scores; more are asked in several calls. */
    public Builder maxRelationalPredictRows(long rows) {
      return set("max_relational_predict_rows", rows);
    }

    /** The most rows one temporal link prediction call scores. */
    public Builder maxTemporalLinkPredictRows(long rows) {
      return set("max_temporal_link_predict_rows", rows);
    }

    /** The most context rows a single-table request carries; materialize samples down to it. */
    public Builder maxTabularContextRows(long rows) {
      return set("max_tabular_context_rows", rows);
    }

    /**
     * The most bytes one call's body takes. A call past it that scores several rows is split into
     * smaller calls; one whose context alone is past it is refused.
     */
    public Builder maxRequestBytes(long bytes) {
      return set("max_request_bytes", bytes);
    }

    private Builder set(String name, long value) {
      if (value < 1) {
        throw new StructuredDataException(
            name + " is " + value + ", and it takes a whole number of at least 1",
            "INVALID_CONFIGURATION",
            Map.of("setting", name),
            null);
      }
      values.put(name, value);
      return this;
    }

    public Limits build() {
      return new Limits(values);
    }
  }
}
