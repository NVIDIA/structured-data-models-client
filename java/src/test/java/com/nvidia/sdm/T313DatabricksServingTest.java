// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.databricks.sdk.WorkspaceClient;
import com.databricks.sdk.core.CredentialsProvider;
import com.databricks.sdk.core.DatabricksConfig;
import com.databricks.sdk.core.DatabricksException;
import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Json;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.time.Duration;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Predicting through a Databricks Model Serving endpoint from Java (T3.13): {@link
 * DatabricksServing} posts the core's bytes as they are to the endpoint's invocations, with the
 * caller's token, and hands back what it was answered with; the core writes the envelope,
 * weighs it against the caller's limit and reads every answer and failure, so each outcome is
 * Python's core route's: the result, the refusal under the endpoint's status and its own code
 * ({@link ModelEndpointException}), a failure the served model answered with, the limit, a call
 * tried again, and an endpoint that cannot be reached; each sentence the core's, a name quoted as
 * Java writes a string.
 */
class T313DatabricksServingTest {
  private static final String ENDPOINT = "kumo-rfm-serving";
  private static final String INVOCATIONS = "/serving-endpoints/" + ENDPOINT + "/invocations";
  private static final String SECRET = "Sup3rSecret-CustomerValue-9f2a";
  private static final String BINARY =
      "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id";

  /** What the workspace answers one call with. */
  record Answer(int status, byte[] body, Map<String, String> headers) {
    static Answer of(int status, String body) {
      return new Answer(status, body.getBytes(StandardCharsets.UTF_8), Map.of());
    }
  }

  /** A workspace answering each call by {@code answers} in turn, then by the served model. */
  static final class Workspace implements AutoCloseable {
    final List<byte[]> bodies = Collections.synchronizedList(new ArrayList<>());
    final List<String> paths = Collections.synchronizedList(new ArrayList<>());
    final List<Map<String, String>> headers = Collections.synchronizedList(new ArrayList<>());
    final List<String> userAgents = Collections.synchronizedList(new ArrayList<>());
    final List<String> managementTokens = Collections.synchronizedList(new ArrayList<>());
    private final HttpServer server;

    @SafeVarargs
    Workspace(Function<byte[], Answer>... answers) throws IOException {
      AtomicInteger call = new AtomicInteger();
      server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
      server.createContext(
          "/",
          exchange -> {
            if (exchange.getRequestURI().getPath().equals("/.well-known/databricks-config")) {
              byte[] metadata = "{}".getBytes(StandardCharsets.UTF_8);
              exchange.sendResponseHeaders(200, metadata.length);
              try (OutputStream out = exchange.getResponseBody()) {
                out.write(metadata);
              }
              return;
            }
            byte[] body;
            try (InputStream in = exchange.getRequestBody()) {
              body = in.readAllBytes();
            }
            bodies.add(body);
            paths.add(exchange.getRequestMethod() + " " + exchange.getRequestURI().getPath());
            Map<String, String> seen = new LinkedHashMap<>();
            for (String name : List.of("Authorization", "Content-Type")) {
              seen.put(name, exchange.getRequestHeaders().getFirst(name));
            }
            headers.add(seen);
            userAgents.add(exchange.getRequestHeaders().getFirst("User-Agent"));
            managementTokens.add(
                exchange.getRequestHeaders().getFirst("X-Databricks-Azure-SP-Management-Token"));
            int at = call.getAndIncrement();
            Answer answer =
                at < answers.length ? answers[at].apply(body) : served(body);
            answer.headers().forEach((k, v) -> exchange.getResponseHeaders().add(k, v));
            exchange.sendResponseHeaders(answer.status(), answer.body().length);
            try (OutputStream out = exchange.getResponseBody()) {
              out.write(answer.body());
            }
          });
      server.start();
    }

    String host() {
      return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    DatabricksServing serving(long limit) {
      return DatabricksServing.of(host(), ENDPOINT, () -> "dapi-test", limit);
    }

    @Override
    public void close() {
      server.stop(0);
    }
  }

  @Test
  void officialSdkAuthenticationRefreshesEveryHeaderWithoutRewritingTheCoreBytes()
      throws Exception {
    AtomicInteger refreshes = new AtomicInteger();
    try (Workspace workspace = new Workspace(body -> Answer.of(200, "{}"))) {
      DatabricksConfig config = new DatabricksConfig().setHost(workspace.host())
          .setCredentialsProvider(CredentialsProvider.from("sdm-test", ignored -> () -> Map.of(
              "Authorization", "Bearer sdk-" + refreshes.incrementAndGet(),
              "X-Databricks-Azure-SP-Management-Token", "management-scope")));
      DatabricksServing serving = DatabricksServing.of(new WorkspaceClient(config), ENDPOINT);
      byte[] body = "{\"opaque\":\"\\u00e9\"}".getBytes(StandardCharsets.UTF_8);
      serving.post(body);
      serving.post(body);
      assertEquals("Bearer sdk-1", workspace.headers.get(0).get("Authorization"));
      assertEquals("Bearer sdk-2", workspace.headers.get(1).get("Authorization"));
      assertEquals(List.of("management-scope", "management-scope"), workspace.managementTokens);
      assertArrayEquals(body, workspace.bodies.get(0));
      assertArrayEquals(body, workspace.bodies.get(1));
      for (String agent : workspace.userAgents) {
        assertTrue(agent.contains("structured-data-models-client/"), agent);
        assertTrue(agent.contains("databricks-sdk-java/"), agent);
      }
    }
  }

  @Test
  void aNamedOfficialSdkProfileSuppliesTheCredentials(@TempDir Path temporary) throws Exception {
    try (Workspace workspace = new Workspace(body -> Answer.of(200, "{}"))) {
      Path profile = temporary.resolve("databrickscfg");
      Files.writeString(profile,
          "[serving-test]\nhost = " + workspace.host() + "\ntoken = profile-token\n");
      WorkspaceClient client = new WorkspaceClient(new DatabricksConfig()
          .setConfigFile(profile.toString()).setProfile("serving-test"));
      DatabricksServing.of(client, ENDPOINT, 1234).timeout(Duration.ofSeconds(2))
          .post(new byte[] {'{', '}'});
      assertEquals("Bearer profile-token", workspace.headers.get(0).get("Authorization"));
    }
  }

  @Test
  void sdkAuthenticationFailuresDoNotExposeTheirCredentialDiagnostic() throws Exception {
    try (Workspace workspace = new Workspace(body -> Answer.of(200, "{}"))) {
      WorkspaceClient client = new WorkspaceClient(new DatabricksConfig().setHost(workspace.host())
          .setCredentialsProvider(CredentialsProvider.from("sdm-test", ignored -> () -> {
            throw new DatabricksException(SECRET);
          })));
      Host.TransportFailed error = assertThrows(Host.TransportFailed.class,
          () -> DatabricksServing.of(client, ENDPOINT).post(new byte[] {'{', '}'}));
      assertFalse(String.valueOf(error).contains(SECRET));
      assertTrue(workspace.bodies.isEmpty());
    }
  }

  @Test
  void theServingDeadlineIncludesSdkCredentialRefresh() throws Exception {
    CountDownLatch refreshing = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    ExecutorService caller = Executors.newSingleThreadExecutor();
    try (Workspace workspace = new Workspace(body -> Answer.of(200, "{}"))) {
      WorkspaceClient client = new WorkspaceClient(new DatabricksConfig().setHost(workspace.host())
          .setCredentialsProvider(CredentialsProvider.from("sdm-test", ignored -> () -> {
            refreshing.countDown();
            try {
              release.await(5, TimeUnit.SECONDS);
            } catch (InterruptedException error) {
              Thread.currentThread().interrupt();
            }
            return Map.of("Authorization", "Bearer late-token");
          })));
      DatabricksServing serving = DatabricksServing.of(client, ENDPOINT)
          .timeout(Duration.ofMillis(150));
      Future<Host.EndpointFailed> result = caller.submit(() -> assertThrows(
          Host.EndpointFailed.class, () -> serving.post(new byte[] {'{', '}'})));
      assertTrue(refreshing.await(1, TimeUnit.SECONDS));
      assertEquals(504, result.get(1, TimeUnit.SECONDS).status());
      assertTrue(workspace.bodies.isEmpty());
    } finally {
      release.countDown();
      caller.shutdownNow();
      assertTrue(caller.awaitTermination(3, TimeUnit.SECONDS));
    }
  }

  /** The call one envelope carries. */
  @SuppressWarnings("unchecked")
  static byte[] call(byte[] envelope) {
    try {
      Map<String, Object> read = (Map<String, Object>) Json.read(envelope);
      List<Object> records = (List<Object>) read.get("dataframe_records");
      assertEquals(1, records.size());
      String request = (String) ((Map<String, Object>) records.get(0)).get("request_json");
      return request.getBytes(StandardCharsets.UTF_8);
    } catch (Json.JsonException e) {
      throw new AssertionError(e);
    }
  }

  /** The endpoint's reply carrying {@code inner} as its one {@code response_json} row. */
  static Answer row(String inner) {
    try {
      Map<String, Object> row = Map.of("response_json", inner);
      return new Answer(200, Json.write(Map.of("predictions", List.of(row))), Map.of());
    } catch (Json.JsonException e) {
      throw new AssertionError(e);
    }
  }

  /** The served model's answer to an envelope, by the scripted rule, in its row. */
  static Answer served(byte[] envelope) {
    return row(new String(T312Shop.scripted(call(envelope)), StandardCharsets.UTF_8));
  }

  @SuppressWarnings("unchecked")
  static Map<String, Object> withoutId(byte[] call) throws Json.JsonException {
    Map<String, Object> read = new LinkedHashMap<>((Map<String, Object>) Json.read(call));
    Map<String, Object> metadata = new LinkedHashMap<>((Map<String, Object>) read.get("metadata"));
    metadata.remove("request_id");
    read.put("metadata", metadata);
    return read;
  }

  private static Request asked(Connection connection) {
    return PredictiveQuery.parse(BINARY)
        .materialize(
            Schema.fromSource(DuckDbSource.of(connection)),
            MaterializeOptions.builder()
                .indices(List.of(0L, 1L))
                .anchorTime(LocalDate.of(2024, 3, 1))
                .build());
  }

  private static List<Object> predictions(PredictionResult result) throws Exception {
    List<Object> out = new ArrayList<>();
    try (ArrowReader reader = result.read(ArrowData.allocator())) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot rows = reader.getVectorSchemaRoot();
        for (int at = 0; at < rows.getRowCount(); at++) {
          out.add(String.valueOf(rows.getVector("prediction").getObject(at)));
        }
      }
    }
    return out;
  }

  @Test
  void theCoresEnvelopeIsPostedAsItIsAndItsRowRead() throws Exception {
    try (Connection connection = T312Shop.open("duckdb");
        Workspace workspace = new Workspace();
        T312Shop.Endpoint nim = new T312Shop.Endpoint();
        SdmClient serving = SdmClient.forDatabricksServing(workspace.serving(16_000_000));
        SdmClient direct = SdmClient.connect(nim.url())) {
      Request request = asked(connection);
      PredictionResult result = serving.predict(request, SdmClient.KUMO_RELATIONAL);
      PredictionResult expected = direct.predict(request, SdmClient.KUMO_RELATIONAL);
      assertEquals(expected.model(), result.model());
      assertEquals(expected.rows(), result.rows());
      assertEquals(predictions(expected), predictions(result));
      assertEquals(expected.outputs(), result.outputs());
      assertEquals("databricks-serving:" + ENDPOINT, serving.url());

      assertEquals(List.of("POST " + INVOCATIONS), workspace.paths);
      assertEquals(
          Map.of("Authorization", "Bearer dapi-test", "Content-Type", "application/json"),
          workspace.headers.get(0));
      byte[] sent = workspace.bodies.get(0);
      String text = new String(sent, StandardCharsets.UTF_8);
      assertTrue(text.startsWith("{\"dataframe_records\":[{\"request_json\":\"{"), text);
      assertTrue(text.chars().allMatch(c -> c < 0x80), "ASCII, as Python's writer");
      assertEquals(withoutId(nim.received.get(0)), withoutId(call(sent)));
    }
  }

  @Test
  void theFourBaseJ2ShapesUseTheSameCoreWithDirectAndServingEndpoints() throws Exception {
    // J2's products/order details become shop users/orders with the same query shapes.
    // These exercise the language-neutral paths; warehouse/JDBC types are covered separately.
    String regression = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";
    String inactivity = "PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH users.user_id";
    String multiclass = "PREDICT users.country FOR EACH users.user_id";
    String[] queries = {regression, regression, inactivity, multiclass};
    LocalDate latest = LocalDate.of(2024, 3, 1);
    LocalDate[] anchors = {latest, latest.minusDays(180), latest, latest};
    try (Connection connection = T312Shop.open("duckdb");
        Workspace workspace = new Workspace();
        T312Shop.Endpoint nim = new T312Shop.Endpoint();
        SdmClient serving = SdmClient.forDatabricksServing(workspace.serving(16_000_000));
        SdmClient direct = SdmClient.connect(nim.url())) {
      Schema schema = Schema.fromSource(DuckDbSource.of(connection));
      for (int at = 0; at < queries.length; at++) {
        Request request = PredictiveQuery.parse(queries[at]).materialize(schema,
            MaterializeOptions.builder().indices(List.of(0L, 1L))
                .anchorTime(anchors[at]).build());
        PredictionResult expected = direct.predict(request, SdmClient.KUMO_RELATIONAL);
        PredictionResult result = serving.predict(request, SdmClient.KUMO_RELATIONAL);
        assertEquals(2, result.rows(), queries[at]);
        assertEquals(predictions(expected), predictions(result), queries[at]);
        assertEquals(withoutId(nim.received.get(at)), withoutId(call(workspace.bodies.get(at))));
      }
    }
  }

  private static ModelEndpointException refused(Answer answer) throws Exception {
    try (Connection connection = T312Shop.open("duckdb");
        Workspace workspace = new Workspace(body -> answer);
        SdmClient client = SdmClient.forDatabricksServing(workspace.serving(16_000_000))) {
      Request request = asked(connection);
      StructuredDataException error =
          assertThrows(
              StructuredDataException.class,
              () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      String shown = error.getMessage() + error.details();
      assertFalse(shown.contains(SECRET), shown);
      return assertInstanceOf(ModelEndpointException.class, error);
    }
  }

  @Test
  void anEndpointsFailureIsPythonsRefusal() throws Exception {
    ModelEndpointException coded =
        refused(
            Answer.of(
                400,
                "{\"error_code\":\"BAD_REQUEST\",\"message\":\"INVALID_SCHEMA: row " + SECRET
                    + "\"}"));
    assertEquals(400, coded.status());
    assertEquals("INVALID_SCHEMA", coded.code());
    assertEquals(
        "serving endpoint \"" + ENDPOINT + "\" refused the request (INVALID_SCHEMA)",
        coded.getMessage());
    assertEquals(ENDPOINT, coded.details().get("endpoint"));
    assertEquals("BAD_REQUEST", coded.details().get("provider_code"));

    ModelEndpointException fault =
        refused(
            Answer.of(
                500,
                "{\"error_code\":\"INTERNAL_ERROR\",\"message\":\"failed to collect any context"
                    + " examples for " + SECRET + "\"}"));
    assertEquals(503, fault.status());
    assertEquals("INVALID_REQUEST", fault.code());
    assertEquals("EMPTY_CONTEXT", fault.details().get("diagnosis"));
    assertEquals("INTERNAL_ERROR", fault.details().get("endpoint_code"));

    ModelEndpointException missing = refused(Answer.of(404, ""));
    assertEquals(404, missing.status());
    assertNull(missing.code());
    assertEquals("serving endpoint \"" + ENDPOINT + "\" request failed", missing.getMessage());

    ModelEndpointException inBand =
        refused(
            row("{\"status\":422,\"code\":\"UNSUPPORTED_TASK\",\"detail\":\"" + SECRET + "\"}"));
    assertEquals(422, inBand.status());
    assertEquals("UNSUPPORTED_TASK", inBand.code());
  }

  @Test
  void anAnswerWithNoRowIsATransportError() throws Exception {
    try (Connection connection = T312Shop.open("duckdb");
        Workspace workspace = new Workspace(body -> Answer.of(200, "{\"predictions\":[]}"));
        SdmClient client = SdmClient.forDatabricksServing(workspace.serving(16_000_000))) {
      Request request = asked(connection);
      StructuredDataException error =
          assertThrows(
              StructuredDataException.class,
              () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      assertEquals("TRANSPORT_ERROR", error.code());
      assertEquals(
          "serving endpoint \"" + ENDPOINT + "\" did not return exactly one prediction row",
          error.getMessage());
    }
  }

  @Test
  void aCallPastTheCallersLimitIsRefusedBeforeItIsSent() throws Exception {
    try (Connection connection = T312Shop.open("duckdb");
        Workspace workspace = new Workspace();
        SdmClient client = SdmClient.forDatabricksServing(workspace.serving(100))) {
      Request request = asked(connection);
      StructuredDataException error =
          assertThrows(
              StructuredDataException.class,
              () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      assertEquals("INVALID_REQUEST", error.code());
      assertEquals(100L, ((Number) error.details().get("limit")).longValue());
      assertTrue(error.getMessage().contains("above the 100 byte limit it accepts"), error.getMessage());
      assertEquals(List.of(), workspace.bodies);
    }
  }

  @Test
  void aCallTheEndpointThrottlesIsTriedAgainWithTheSameBytes() throws Exception {
    Answer busy =
        new Answer(
            429,
            "{\"error_code\":\"REQUEST_LIMIT_EXCEEDED\",\"message\":\"slow down\"}"
                .getBytes(StandardCharsets.UTF_8),
            Map.of("Retry-After", "0"));
    try (Connection connection = T312Shop.open("duckdb");
        Workspace workspace = new Workspace(body -> busy, body -> Answer.of(503, ""));
        SdmClient client = SdmClient.forDatabricksServing(workspace.serving(16_000_000))) {
      PredictionResult result = client.predict(asked(connection), SdmClient.KUMO_RELATIONAL);
      assertEquals(2, result.rows());
      assertEquals(3, workspace.bodies.size());
      assertArrayEquals(workspace.bodies.get(0), workspace.bodies.get(1));
      assertArrayEquals(workspace.bodies.get(0), workspace.bodies.get(2));
    }
  }

  @Test
  void anEndpointThatCannotBeReachedIsAFailureAtItsDeadline() throws Exception {
    int port;
    try (ServerSocket free = new ServerSocket(0)) {
      port = free.getLocalPort();
    }
    DatabricksServing nowhere =
        DatabricksServing.of("http://127.0.0.1:" + port, ENDPOINT, () -> "dapi-test", 16_000_000)
            .timeout(Duration.ofSeconds(1));
    try (Connection connection = T312Shop.open("duckdb");
        SdmClient client = SdmClient.forDatabricksServing(nowhere)) {
      Request request = asked(connection);
      ModelEndpointException error =
          assertThrows(
              ModelEndpointException.class,
              () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      assertEquals(504, error.status());
      assertNull(error.code());
    }
  }

  @Test
  void aResponseBodyThatStallsAfterHeadersIsBoundedByTheCallDeadline() throws Exception {
    stalledBody(false);
  }

  @Test
  void aRetryHasOnlyTheRemainingCallBudgetToReadItsBody() throws Exception {
    stalledBody(true);
  }

  /** Keep the body open until cleanup, so headers alone cannot complete the call. */
  private static void stalledBody(boolean retry) throws Exception {
    CountDownLatch headers = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    AtomicInteger calls = new AtomicInteger();
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    ExecutorService caller = Executors.newSingleThreadExecutor();
    server.createContext("/", exchange -> {
      try (exchange) {
        exchange.getRequestBody().readAllBytes();
        if (retry && calls.incrementAndGet() == 1) {
          try {
            Thread.sleep(1500);
          } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return;
          }
          exchange.getResponseHeaders().add("Retry-After", "0");
          exchange.sendResponseHeaders(503, -1);
          return;
        }
        exchange.sendResponseHeaders(200, 2);
        exchange.getResponseBody().flush();
        headers.countDown();
        try {
          release.await(8, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
          Thread.currentThread().interrupt();
        }
      }
    });
    server.start();
    try {
      DatabricksServing serving = DatabricksServing.of(
          "http://127.0.0.1:" + server.getAddress().getPort(), ENDPOINT, () -> "dapi-test")
          .timeout(Duration.ofSeconds(3));
      long started = System.nanoTime();
      Future<Host.EndpointFailed> result = caller.submit(() -> assertThrows(
          Host.EndpointFailed.class, () -> serving.post(new byte[] {'{', '}'})));
      assertTrue(headers.await(3, TimeUnit.SECONDS), "the endpoint must have sent headers");
      long remaining = Duration.ofMillis(3800).toNanos() - (System.nanoTime() - started);
      Host.EndpointFailed error = result.get(Math.max(1, remaining), TimeUnit.NANOSECONDS);
      assertEquals(504, error.status());
      if (retry) {
        assertEquals(2, calls.get(), "the second attempt must have started");
      }
    } finally {
      release.countDown();
      server.stop(0);
      caller.shutdownNow();
      assertTrue(caller.awaitTermination(3, TimeUnit.SECONDS));
    }
  }

  @Test
  void aServingConfigurationThatCannotBeUsedIsRefusedWithoutRepeatingIt() {
    String[] endpoints = {"https://host/serving-endpoints/x", "a/b", "x?token=" + SECRET, "", " x"};
    for (String endpoint : endpoints) {
      StructuredDataException error =
          assertThrows(
              StructuredDataException.class,
              () -> DatabricksServing.of("https://host", endpoint, () -> "t", 16_000_000));
      assertEquals("INVALID_CONFIGURATION", error.code(), endpoint);
      if (!endpoint.isBlank()) {
        assertFalse(error.getMessage().contains(endpoint.strip()), error.getMessage());
      }
    }
    for (long limit : new long[] {0, -1}) {
      assertEquals(
          "INVALID_CONFIGURATION",
          assertThrows(
                  StructuredDataException.class,
                  () -> DatabricksServing.of("https://host", ENDPOINT, () -> "t", limit))
              .code());
    }
    for (String host : new String[] {"ftp://host", "http://workspace.example", "https://h/x?y=1"}) {
      assertEquals(
          "INVALID_CONFIGURATION",
          assertThrows(
                  StructuredDataException.class,
                  () -> DatabricksServing.of(host, ENDPOINT, () -> "t", 16_000_000))
              .code(),
          host);
    }
    assertEquals(
        16_000_000L,
        DatabricksServing.of("https://host", ENDPOINT, () -> "t").maxRequestBytes());
  }
}
