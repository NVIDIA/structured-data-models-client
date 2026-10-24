// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.databricks.sdk.WorkspaceClient;
import com.databricks.sdk.core.DatabricksConfig;
import com.databricks.sdk.core.UserAgent;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.HttpTransport;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Supplier;

/**
 * A Databricks Model Serving endpoint a client predicts through ({@link
 * SdmClient#forDatabricksServing}), as Python's {@code SDMClient.for_databricks_serving}: the
 * endpoint is named, not addressed, inside the workspace at {@code host}, and called with the
 * official SDK's unified authentication, or a token supplied by the caller.
 *
 * <p>The core writes each call's envelope (the one {@code request_json} row the served model
 * reads), weighs it against {@link #maxRequestBytes}, and reads every answer and failure, as
 * Python's serving transport does; this posts the core's bytes as they are to the endpoint's
 * invocations and hands back what it was answered with. A call the endpoint throttles (429) or
 * finds unavailable (503), or one that cannot reach it, is tried again, as the Databricks SDK
 * tries one, until the timeout runs out.
 *
 * <pre>{@code
 * DatabricksServing serving =
 *     DatabricksServing.of(new WorkspaceClient(), "kumo-rfm");
 * try (SdmClient client = SdmClient.forDatabricksServing(serving)) {
 *   PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL);
 * }
 * }</pre>
 */
public final class DatabricksServing {
  /** The gateway's own limit on one request, in decimal megabytes, Python's default. */
  public static final long MAX_REQUEST_BYTES = 16L * 1000 * 1000;

  /** A serving endpoint scales to zero, and a cold start outlasts a minute: Python's default. */
  public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(300);

  private static final Set<String> LOCAL_HOSTS = Set.of("localhost", "127.0.0.1", "::1", "[::1]");

  private final String host;
  private final String endpoint;
  private final Supplier<String> token;
  private final SdkAuthentication sdk;
  private final long maxRequestBytes;
  private final Duration timeout;
  private final HttpClient client;

  private DatabricksServing(
      String host, String endpoint, Supplier<String> token, SdkAuthentication sdk,
      long maxRequestBytes, Duration timeout) {
    this.host = host;
    this.endpoint = endpoint;
    this.token = token;
    this.sdk = sdk;
    this.maxRequestBytes = maxRequestBytes;
    this.timeout = timeout;
    this.client = HttpClient.newBuilder().connectTimeout(timeout).build();
  }

  /**
   * Uses the official SDK's default unified authentication: environment, configuration profile,
   * or any other authentication method the SDK supports.
   */
  public static DatabricksServing of(String endpoint) {
    String name = name(endpoint);
    try {
      return of(new WorkspaceClient(), name);
    } catch (RuntimeException error) {
      throw configuration("Databricks SDK authentication could not be configured");
    }
  }

  /** Uses {@code workspace}'s official SDK configuration and refreshed authentication headers. */
  public static DatabricksServing of(WorkspaceClient workspace, String endpoint) {
    return of(workspace, endpoint, MAX_REQUEST_BYTES);
  }

  /**
   * Uses an existing official SDK workspace client, preserving all its authentication methods
   * and headers, without passing the core's request or response through the SDK's JSON codecs.
   * The SDK discovers or refreshes credentials at each attempt; the serving deadline includes
   * that work. The caller retains ownership of the SDK client and configuration.
   */
  public static DatabricksServing of(
      WorkspaceClient workspace, String endpoint, long maxRequestBytes) {
    Objects.requireNonNull(workspace, "workspace");
    String name = name(endpoint);
    positiveLimit(maxRequestBytes);
    DatabricksConfig config = Objects.requireNonNull(workspace.config(), "workspace.config");
    return new DatabricksServing(
        null, name, null, new SdkAuthentication(config), maxRequestBytes, DEFAULT_TIMEOUT);
  }

  /** The endpoint {@code endpoint} in the workspace at {@code host}, with the gateway's limit. */
  public static DatabricksServing of(String host, String endpoint, Supplier<String> token) {
    return of(host, endpoint, token, MAX_REQUEST_BYTES);
  }

  /**
   * The endpoint {@code endpoint} (its name, not its URL) in the workspace at {@code host} (its
   * base URL, {@code https://...}), called with the token {@code token} gives, each request at
   * most {@code maxRequestBytes} once wrapped in the endpoint's envelope.
   *
   * @throws StructuredDataException {@code INVALID_CONFIGURATION} for a host that is not a base
   *     URL or would send the token over plaintext HTTP to another host, an endpoint that is not
   *     a name (refused without being repeated, since a URL can carry a token), or a limit that
   *     is not positive
   */
  public static DatabricksServing of(
      String host, String endpoint, Supplier<String> token, long maxRequestBytes) {
    Objects.requireNonNull(token, "token");
    String base = base(host);
    String name = name(endpoint);
    positiveLimit(maxRequestBytes);
    return new DatabricksServing(base, name, token, null, maxRequestBytes, DEFAULT_TIMEOUT);
  }

  /** This endpoint, each attempt and its retries bounded by {@code timeout}. */
  public DatabricksServing timeout(Duration timeout) {
    if (timeout == null || timeout.isNegative() || timeout.isZero()) {
      throw configuration("timeout must be a positive duration");
    }
    return new DatabricksServing(host, endpoint, token, sdk, maxRequestBytes, timeout);
  }

  /** The serving endpoint's name. */
  public String endpoint() {
    return endpoint;
  }

  /** The most bytes one request may take once wrapped in the endpoint's envelope. */
  public long maxRequestBytes() {
    return maxRequestBytes;
  }

  /** The longest one call, its retries included, may take. */
  public Duration timeout() {
    return timeout;
  }

  /** Where requests go: the endpoint's name, since it has no URL of its own. */
  String url() {
    return "databricks-serving:" + endpoint;
  }

  /** This endpoint as the core is told of it ({@code Predict.model_endpoint}). */
  Map<String, Object> modelEndpoint() {
    Map<String, Object> named = new LinkedHashMap<>();
    named.put("endpoint", "serving_endpoint");
    named.put("name", endpoint);
    named.put("max_request_bytes", maxRequestBytes);
    return named;
  }

  /**
   * Posts {@code body}, the core's envelope, as it is, and gives back the endpoint's answer's
   * bytes. A 429 or 503, or an attempt that cannot reach the endpoint, is tried again after the
   * endpoint's {@code Retry-After} (a second where it gives none) or a growing pause, as the SDK
   * tries one, until the timeout runs out, which is the failure of a status of 504.
   *
   * @throws Host.EndpointFailed the endpoint failed, with its status and body
   * @throws Host.TransportFailed the call was interrupted
   */
  byte[] post(byte[] body) throws Host.EndpointFailed, Host.TransportFailed {
    long deadline = System.nanoTime() + timeout.toNanos();
    for (int attempt = 1; ; attempt++) {
      Authentication authentication = sdk == null
          ? new Authentication(host, Map.of("Authorization", "Bearer " + token.get()))
          : authenticate(deadline);
      long remaining = deadline - System.nanoTime();
      if (remaining <= 0) {
        throw new Host.EndpointFailed(504, new byte[0]);
      }
      // Native SDK authentication can discover the workspace host. Validate it afterwards,
      // before any credential is sent, and on every attempt if the caller edits its config.
      URI target = URI.create(base(authentication.host())
          + "/serving-endpoints/" + endpoint + "/invocations");
      HttpRequest request;
      try {
        HttpRequest.Builder builder = HttpRequest.newBuilder(target)
            .timeout(Duration.ofNanos(remaining));
        authentication.headers().forEach(builder::header);
        request = builder
            .setHeader("Content-Type", "application/json")
            .setHeader("User-Agent", userAgent())
            .POST(HttpRequest.BodyPublishers.ofByteArray(body))
            .build();
      } catch (IllegalArgumentException error) {
        // Header validation diagnostics can quote the invalid credential.
        throw new Host.TransportFailed();
      }
      long pause;
      try {
        HttpResponse<byte[]> response = HttpTransport.exchange(
            client, request, HttpResponse.BodyHandlers.ofByteArray(),
            Math.max(0, deadline - System.nanoTime()));
        if (System.nanoTime() - deadline >= 0) {
          throw new Host.EndpointFailed(504, new byte[0]);
        }
        int status = response.statusCode();
        if (status >= 200 && status < 300) {
          return response.body();
        }
        if (status != 429 && status != 503) {
          throw new Host.EndpointFailed(status, response.body());
        }
        pause = retryAfter(response.headers().firstValue("Retry-After").orElse(null));
      } catch (HttpTimeoutException e) {
        throw new Host.EndpointFailed(504, new byte[0]);
      } catch (IOException e) {
        pause = Math.min(10, attempt) * 1000L;
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new Host.TransportFailed();
      }
      pause += ThreadLocalRandom.current().nextLong(1000);
      if (System.nanoTime() + pause * 1_000_000L >= deadline) {
        // The SDK's retries end as its own timeout, which Python reports as 504.
        throw new Host.EndpointFailed(504, new byte[0]);
      }
      try {
        Thread.sleep(pause);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new Host.TransportFailed();
      }
    }
  }

  private record Authentication(String host, Map<String, String> headers) {}

  /** Shared by timeout copies: a noninterruptible SDK refresh never accumulates more workers. */
  private static final class SdkAuthentication {
    final DatabricksConfig config;
    final Semaphore permit = new Semaphore(1, true);

    SdkAuthentication(DatabricksConfig config) {
      this.config = config;
    }
  }

  /** SDK refresh may perform IO: cancellation must bound it as well as the serving body. */
  private Authentication authenticate(long deadline)
      throws Host.EndpointFailed, Host.TransportFailed {
    long remaining = deadline - System.nanoTime();
    if (remaining <= 0) {
      throw new Host.EndpointFailed(504, new byte[0]);
    }
    try {
      if (!sdk.permit.tryAcquire(remaining, TimeUnit.NANOSECONDS)) {
        throw new Host.EndpointFailed(504, new byte[0]);
      }
    } catch (InterruptedException error) {
      Thread.currentThread().interrupt();
      throw new Host.TransportFailed();
    }
    FutureTask<Authentication> pending = new FutureTask<>(() -> {
      Map<String, String> headers = Map.copyOf(sdk.config.authenticate());
      return new Authentication(sdk.config.getHost(), headers);
    });
    Thread refresh = new Thread(() -> {
      try {
        pending.run();
      } finally {
        sdk.permit.release();
      }
    }, "sdm-databricks-auth-" + Integer.toHexString(System.identityHashCode(sdk)));
    refresh.setDaemon(true);
    try {
      refresh.start();
    } catch (RuntimeException | Error error) {
      sdk.permit.release();
      throw error;
    }
    try {
      return pending.get(Math.max(0, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
    } catch (TimeoutException error) {
      throw new Host.EndpointFailed(504, new byte[0]);
    } catch (InterruptedException error) {
      Thread.currentThread().interrupt();
      throw new Host.TransportFailed();
    } catch (ExecutionException error) {
      // SDK diagnostics can carry credentials and configuration. They stay internal.
      throw new Host.TransportFailed();
    } finally {
      pending.cancel(true);
    }
  }

  /** Keep the SDK's identity without mutating its process-wide product setting. */
  private static String userAgent() {
    String version = DatabricksServing.class.getPackage().getImplementationVersion();
    return "structured-data-models-client/" + (version == null ? "development" : version)
        + " " + UserAgent.asString();
  }

  private static void positiveLimit(long maxRequestBytes) {
    if (maxRequestBytes <= 0) {
      throw configuration(
          "max_request_bytes must be a positive integer, got " + maxRequestBytes);
    }
  }

  /** The pause a {@code Retry-After} asks for, in milliseconds: its seconds, else one. */
  private static long retryAfter(String given) {
    if (given != null) {
      try {
        return Math.max(0, Long.parseLong(given.strip())) * 1000L;
      } catch (NumberFormatException e) {
        // A date or another form: a second, as the SDK waits.
      }
    }
    return 1000L;
  }

  private static StructuredDataException configuration(String message) {
    return new StructuredDataException(message, "INVALID_CONFIGURATION", Map.of(), null);
  }

  /** {@code host} as a base URL: http or https, a host, no query or fragment. */
  private static String base(String host) {
    if (host == null) {
      throw configuration("host must be the workspace's URL");
    }
    URI parsed;
    try {
      parsed = new URI(host);
    } catch (URISyntaxException e) {
      throw configuration("host is not a URL");
    }
    String scheme = parsed.getScheme();
    if (!"http".equals(scheme) && !"https".equals(scheme)) {
      throw configuration("host must start with http:// or https://");
    }
    if (parsed.getHost() == null || parsed.getHost().isEmpty()) {
      throw configuration("host is missing a host");
    }
    if (parsed.getRawQuery() != null
        || parsed.getRawFragment() != null
        || parsed.getRawUserInfo() != null) {
      throw configuration(
          "host must be the workspace's base URL, without a query string, fragment or user;"
              + " pass the token through the token supplier");
    }
    if ("http".equals(scheme)
        && !LOCAL_HOSTS.contains(parsed.getHost().toLowerCase(Locale.ROOT))) {
      throw configuration(
          "refusing to send a token over plaintext HTTP; use an https:// URL (or a localhost"
              + " workspace)");
    }
    return host.endsWith("/") ? host.substring(0, host.length() - 1) : host;
  }

  /**
   * {@code endpoint} as a serving endpoint's name, refused without being repeated: a pasted URL
   * can carry a token, so the scan for one runs before any check that would echo it.
   */
  private static String name(String endpoint) {
    if (endpoint == null || endpoint.isBlank()) {
      throw configuration("endpoint must be a non-empty serving endpoint name");
    }
    for (String marker : new String[] {"://", "/", "?", "#", "@"}) {
      if (endpoint.contains(marker)) {
        throw configuration("endpoint must be a serving endpoint name, not a URL");
      }
    }
    if (!endpoint.equals(endpoint.strip())) {
      throw configuration("endpoint name has surrounding whitespace");
    }
    return endpoint;
  }

  @Override
  public String toString() {
    return "DatabricksServing(" + endpoint + ")";
  }
}
