// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.ModelEndpointException;
import com.nvidia.sdm.StructuredDataException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.time.Duration;
import java.util.Map;
import java.util.Set;

/**
 * The model transport over {@code java.net.http}, the Java twin of the Python host's {@code
 * Transport} ({@code sdm_client/core/transport.py}): it posts the core's call to {@code
 * /v1/predictions} as the bytes the core wrote, never parsing or re-encoding them (design choice
 * 10, "Model calls"), and gives back the endpoint's answer as the bytes it sent, which the core
 * reads. A transient failure (a status of 429, 502, 503 or 504, or a connection that failed) is
 * tried again with exponential backoff, a {@code Retry-After} the endpoint sends honoured up to a
 * minute; a status of 400 or more is the endpoint's refusal ({@link ModelEndpointException}); an
 * endpoint that cannot be reached, or an answer past 64 MiB or that is not a JSON object, is a
 * failed transport ({@link Failed}). Not part of the API.
 */
public final class HttpTransport {
  /** Where a prediction is posted. */
  public static final String PREDICTIONS = "/v1/predictions";

  /** The statuses tried again. */
  static final Set<Integer> RETRY_STATUS = Set.of(429, 502, 503, 504);

  /** The largest answer read. */
  static final int MAX_RESPONSE_BYTES = 64 * 1024 * 1024;

  /** The longest a {@code Retry-After} is waited. */
  static final long RETRY_AFTER_MAX_SECONDS = 60;

  private static final Set<String> LOCAL_HOSTS = Set.of("localhost", "127.0.0.1", "::1", "[::1]");

  private final String url;
  private final String apiKey;
  private final Duration timeout;
  private final int maxRetries;
  private final HttpClient client;

  /** The transport could not reach the endpoint or read its answer. */
  public static final class Failed extends Exception {
    private static final long serialVersionUID = 1L;

    Failed(String message) {
      super(message);
    }
  }

  public HttpTransport(String url, String apiKey, Duration timeout, int maxRetries) {
    this.url = validated(url, apiKey);
    this.apiKey = apiKey;
    if (timeout == null || timeout.isNegative() || timeout.isZero()) {
      throw configuration("timeout must be a positive duration");
    }
    if (maxRetries < 0) {
      throw configuration("maxRetries must be a non-negative integer, got " + maxRetries);
    }
    this.timeout = timeout;
    this.maxRetries = maxRetries;
    this.client = HttpClient.newBuilder().connectTimeout(timeout).build();
  }

  public String url() {
    return url;
  }

  private static StructuredDataException configuration(String message) {
    return new StructuredDataException(message, "INVALID_CONFIGURATION", Map.of(), null);
  }

  /** {@code url} as a base endpoint: http or https, a host, no query or fragment. */
  private static String validated(String url, String apiKey) {
    if (url == null) {
      throw configuration("url must be a string");
    }
    URI parsed;
    try {
      parsed = new URI(url);
    } catch (URISyntaxException e) {
      throw configuration("url is not a URL");
    }
    String scheme = parsed.getScheme();
    if (!"http".equals(scheme) && !"https".equals(scheme)) {
      throw configuration("url must start with http:// or https://");
    }
    if (parsed.getHost() == null || parsed.getHost().isEmpty()) {
      throw configuration("url is missing a host");
    }
    if (parsed.getRawQuery() != null || parsed.getRawFragment() != null) {
      throw configuration(
          "url must be a base endpoint without a query string or fragment; pass a credential as"
              + " the API key rather than in the URL");
    }
    boolean credential = apiKey != null || parsed.getRawUserInfo() != null;
    if (credential && "http".equals(scheme)
        && !LOCAL_HOSTS.contains(parsed.getHost().toLowerCase(java.util.Locale.ROOT))) {
      throw configuration(
          "refusing to send a credential over plaintext HTTP; use an https:// URL (or a"
              + " localhost endpoint)");
    }
    return url.endsWith("/") ? url.substring(0, url.length() - 1) : url;
  }

  /**
   * Posts {@code body} to {@link #PREDICTIONS} as it is and gives back the answer's bytes, an
   * empty answer as {@code {}}.
   *
   * @throws ModelEndpointException the endpoint answered with a status of 400 or more
   * @throws Failed the endpoint could not be reached, or its answer could not be read
   */
  public byte[] predict(byte[] body) throws Failed {
    URI target = URI.create(url + PREDICTIONS);
    HttpRequest.Builder request =
        HttpRequest.newBuilder(target)
            .timeout(timeout)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofByteArray(body));
    if (apiKey != null) {
      request.header("X-API-Key", apiKey);
    }
    HttpRequest built = request.build();
    for (int attempt = 0; ; attempt++) {
      HttpResponse<InputStream> response;
      try {
        response = client.send(built, HttpResponse.BodyHandlers.ofInputStream());
      } catch (HttpTimeoutException e) {
        if (attempt < maxRetries) {
          backoff(attempt, null);
          continue;
        }
        throw new Failed("the request timed out after " + timeout + ", the timeout given");
      } catch (IOException e) {
        if (attempt < maxRetries) {
          backoff(attempt, null);
          continue;
        }
        throw new Failed("the request failed: " + e.getClass().getSimpleName());
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new Failed("the request was interrupted");
      }
      int status = response.statusCode();
      byte[] content = capped(response);
      if (RETRY_STATUS.contains(status) && attempt < maxRetries) {
        backoff(attempt, response.headers().firstValue("Retry-After").orElse(null));
        continue;
      }
      if (status >= 400) {
        throw ModelEndpointException.of(status, content);
      }
      if (content.length == 0) {
        return "{}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
      }
      checkObject(content);
      return content;
    }
  }

  /**
   * The answer must be a JSON object; one nested past this host's reader is handed over as sent,
   * for the core to refuse as too deep, as the Python host does (register P1B-22).
   */
  private static void checkObject(byte[] content) throws Failed {
    Object read;
    try {
      read = Json.read(content);
    } catch (Json.TooDeep e) {
      return;
    } catch (Json.JsonException e) {
      throw new Failed("the endpoint did not return JSON");
    }
    if (!(read instanceof Map<?, ?>)) {
      throw new Failed("expected a JSON object from the endpoint");
    }
  }

  private static byte[] capped(HttpResponse<InputStream> response) throws Failed {
    try (InputStream in = response.body()) {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      byte[] chunk = new byte[64 * 1024];
      int read;
      while ((read = in.read(chunk)) >= 0) {
        if (out.size() + read > MAX_RESPONSE_BYTES) {
          throw new Failed("the response body exceeds the " + MAX_RESPONSE_BYTES + " byte limit");
        }
        out.write(chunk, 0, read);
      }
      return out.toByteArray();
    } catch (IOException e) {
      throw new Failed("the response could not be read: " + e.getClass().getSimpleName());
    }
  }

  /** Waits before attempt {@code attempt + 1}: half a second doubled each time, or Retry-After. */
  private static void backoff(int attempt, String retryAfter) {
    long millis = (long) (500 * Math.pow(2, attempt));
    if (retryAfter != null) {
      try {
        millis = Math.min(Long.parseLong(retryAfter.strip()), RETRY_AFTER_MAX_SECONDS) * 1000;
      } catch (NumberFormatException e) {
        // A date or other form: the backoff stands.
      }
    }
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

}
