// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.time.Duration;
import java.util.Objects;

/**
 * How {@link SdmClient#connect(String, ClientOptions)} reaches the endpoint, as Python's {@code
 * SDMClient(url, api_key, timeout=, max_retries=, limits=)}: an API key sent as {@code X-API-Key}
 * (refused over plaintext {@code http://} to a host other than this one), how long each attempt
 * may take (60 seconds), how many times a transient failure is tried again (3), and the size
 * limits requests are held to ({@link Limits}, each by default its environment variable's, else
 * the serving layer's default).
 */
public final class ClientOptions {
  private final String apiKey;
  private final Duration timeout;
  private final int maxRetries;
  private final Limits limits;

  private ClientOptions(Builder builder) {
    this.apiKey = builder.apiKey;
    this.timeout = builder.timeout;
    this.maxRetries = builder.maxRetries;
    this.limits = builder.limits;
  }

  public static ClientOptions defaults() {
    return builder().build();
  }

  public static Builder builder() {
    return new Builder();
  }

  String apiKey() {
    return apiKey;
  }

  Duration timeout() {
    return timeout;
  }

  int maxRetries() {
    return maxRetries;
  }

  Limits limits() {
    return limits;
  }

  /** Builds {@link ClientOptions}. */
  public static final class Builder {
    private String apiKey;
    private Duration timeout = Duration.ofSeconds(60);
    private int maxRetries = 3;
    private Limits limits = Limits.defaults();

    private Builder() {}

    /** The key an authenticating gateway in front of the endpoint takes. */
    public Builder apiKey(String apiKey) {
      this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
      return this;
    }

    /** How long each attempt may take, not the call as a whole. */
    public Builder timeout(Duration timeout) {
      this.timeout = Objects.requireNonNull(timeout, "timeout");
      return this;
    }

    /** How many times a transient failure (408, 429, 502, 503, 504, a dropped connection) is retried. */
    public Builder maxRetries(int maxRetries) {
      this.maxRetries = maxRetries;
      return this;
    }

    /** The size limits requests to the endpoint are held to. */
    public Builder limits(Limits limits) {
      this.limits = Objects.requireNonNull(limits, "limits");
      return this;
    }

    public ClientOptions build() {
      return new ClientOptions(this);
    }
  }
}
