// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.time.Duration;
import java.util.Objects;

/**
 * How {@link SdmClient#connect(String, ClientOptions)} reaches the endpoint, as Python's {@code
 * SDMClient(url, api_key, timeout=, max_retries=, exact_url=, bearer_token=)}: an API key sent as
 * {@code X-API-Key}, or as {@code Authorization: Bearer} (refused over plaintext {@code http://} to
 * a host other than this one), how long each attempt may take (60 seconds), how many times a
 * transient failure is tried again (3), and whether the URL is the endpoint's predictions URL
 * itself rather than a base.
 *
 * <p>A hosted model in the NVIDIA API catalog takes both {@code exactUrl} and {@code bearerToken}:
 *
 * <pre>{@code
 * SdmClient client = SdmClient.connect(
 *     "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions",
 *     ClientOptions.builder().apiKey(key).exactUrl(true).bearerToken(true).build());
 * }</pre>
 */
public final class ClientOptions {
  private final String apiKey;
  private final Duration timeout;
  private final int maxRetries;
  private final boolean exactUrl;
  private final boolean bearerToken;

  private ClientOptions(Builder builder) {
    this.apiKey = builder.apiKey;
    this.timeout = builder.timeout;
    this.maxRetries = builder.maxRetries;
    this.exactUrl = builder.exactUrl;
    this.bearerToken = builder.bearerToken;
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

  boolean exactUrl() {
    return exactUrl;
  }

  boolean bearerToken() {
    return bearerToken;
  }

  /** Builds {@link ClientOptions}. */
  public static final class Builder {
    private String apiKey;
    private Duration timeout = Duration.ofSeconds(60);
    private int maxRetries = 3;
    private boolean exactUrl;
    private boolean bearerToken;

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

    /**
     * Post predictions to the URL as it is, rather than append {@code /v1/predictions} to it: for an
     * endpoint, such as a hosted model in the NVIDIA API catalog, that serves predictions at one URL.
     * Such an endpoint has no sessions, model list or readiness route, so {@code fit}, {@code
     * models} and {@code healthReady} fail with a named error.
     */
    public Builder exactUrl(boolean exactUrl) {
      this.exactUrl = exactUrl;
      return this;
    }

    /** Send the API key as {@code Authorization: Bearer} rather than {@code X-API-Key}. */
    public Builder bearerToken(boolean bearerToken) {
      this.bearerToken = bearerToken;
      return this;
    }

    public ClientOptions build() {
      return new ClientOptions(this);
    }
  }
}
