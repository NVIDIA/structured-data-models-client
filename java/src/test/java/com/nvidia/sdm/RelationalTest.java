// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Json;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

/** Public multi-table journeys through Java, JNI, the shared sampler and an HTTP endpoint. */
class RelationalTest {
  static final String SCALAR = "PREDICT users.value FOR EACH users.user_id";
  static final String TEMPORAL = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";
  static final String LINK = "PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) FOR EACH users.user_id";
  static final String RANK = "PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 2 FOR EACH users.user_id";
  static final String FORECAST = "PREDICT SUM(orders.price, 0, 30, days) FORECAST 2 TIMEFRAMES FOR EACH users.user_id";

  static Table users(RootAllocator allocator) {
    BigIntVector id = new BigIntVector("user_id", allocator);
    Float8Vector value = new Float8Vector("value", allocator);
    VarCharVector group = new VarCharVector("group", allocator); group.allocateNew();
    id.allocateNew(20); value.allocateNew(20);
    for (int i = 0; i < 20; i++) {
      id.setSafe(i, i + 1); value.setSafe(i, i * 1.25);
      group.setSafe(i, (i % 2 == 0 ? "a" : "b").getBytes(StandardCharsets.UTF_8));
    }
    id.setValueCount(20); value.setValueCount(20); group.setValueCount(20);
    return new Table(List.of(id, value, group));
  }

  static Table orders(RootAllocator allocator, boolean temporal, boolean links) {
    List<FieldVector> vectors = new ArrayList<>();
    BigIntVector order = new BigIntVector("order_id", allocator);
    BigIntVector user = new BigIntVector("user_id", allocator);
    Float8Vector price = new Float8Vector("price", allocator);
    order.allocateNew(400); user.allocateNew(400); price.allocateNew(400);
    for (int i = 0; i < 400; i++) {
      order.setSafe(i, i); user.setSafe(i, i % 20 + 1); price.setSafe(i, (i % 17) * 1.25);
    }
    order.setValueCount(400); user.setValueCount(400); price.setValueCount(400);
    vectors.add(order); vectors.add(user); vectors.add(price);
    if (links) {
      BigIntVector item = new BigIntVector("item_id", allocator); item.allocateNew(400);
      for (int i = 0; i < 400; i++) item.setSafe(i, i % 5 + 1);
      item.setValueCount(400); vectors.add(item);
    }
    if (temporal) {
      TimeStampMilliVector time = new TimeStampMilliVector("time", allocator); time.allocateNew(400);
      long first = LocalDate.of(2024, 1, 1).atStartOfDay().toInstant(ZoneOffset.UTC).toEpochMilli();
      for (int i = 0; i < 400; i++) time.setSafe(i, first + i * 86_400_000L);
      time.setValueCount(400); vectors.add(time);
    }
    return new Table(vectors);
  }

  static Table items(RootAllocator allocator) {
    BigIntVector id = new BigIntVector("item_id", allocator); id.allocateNew(5);
    Float8Vector price = new Float8Vector("price", allocator); price.allocateNew(5);
    for (int i = 0; i < 5; i++) { id.setSafe(i, i + 1); price.setSafe(i, i + 0.5); }
    id.setValueCount(5); price.setValueCount(5);
    return new Table(List.of(id, price));
  }

  static Schema fixture(boolean temporal, boolean links) {
    try (RootAllocator allocator = new RootAllocator(); Table users = users(allocator);
        Table orders = orders(allocator, temporal, links); Table items = items(allocator)) {
      Map<String, Table> tables = new LinkedHashMap<>();
      tables.put("users", users); tables.put("orders", orders);
      if (links) tables.put("items", items);
      Schema schema = Schema.fromData(tables).withPrimaryKey("users", "user_id")
          .withPrimaryKey("orders", "order_id").withStype("users", "value", "numerical");
      List<Relationship> relationships = new ArrayList<>();
      relationships.add(new Relationship("orders", "user_id", "users", "user_id"));
      if (links) {
        schema = schema.withPrimaryKey("items", "item_id");
        relationships.add(new Relationship("orders", "item_id", "items", "item_id"));
      }
      schema = schema.withRelationships(relationships);
      return temporal ? schema.withTimeColumn("orders", "time") : schema;
    }
  }

  static MaterializeOptions.Builder options(Object... ids) {
    return MaterializeOptions.builder().model(SdmClient.KUMO_RELATIONAL)
        .indices(ids).randomSeed(7).numContextSamples(18);
  }

  @SuppressWarnings("unchecked") static Map<String, Object> map(Object value) {
    return (Map<String, Object>) value;
  }
  static Map<String, Object> payload(Request request, int index) throws Exception {
    return map(Json.read(request.payloads().get(index).getBytes(StandardCharsets.UTF_8)));
  }
  static Map<String, Object> instance(Map<String, Object> payload, String half) {
    return map(map(payload.get(half)).get("instance_table"));
  }
  @SuppressWarnings("unchecked") static List<List<Object>> rows(Map<String, Object> table) {
    return (List<List<Object>>) table.get("rows");
  }
  static Set<Object> instanceIds(Map<String, Object> table) {
    int id = ((List<?>) table.get("columns")).indexOf("instance_id");
    Set<Object> result = new HashSet<>();
    for (List<Object> row : rows(table)) result.add(row.get(id));
    return result;
  }

  @Test void multipleTablesRequireExplicitRelationalAndSurviveCallerClosure() throws Exception {
    Schema schema = fixture(false, false);
    PredictiveQuery query = PredictiveQuery.parse(SCALAR);
    assertThrows(StructuredDataException.class,
        () -> query.materialize(schema, MaterializeOptions.builder().indices(1L, 2L).build()));
    Request request = query.materialize(schema, options(1L, 2L).build());
    assertEquals(2, request.predictRows()); assertEquals(16, request.contextRows());
    assertTrue(request.isRelational());
    assertEquals("regression", map(payload(request, 0).get("task")).get("kind"));
    assertTrue(map(map(payload(request, 0).get("schema")).get("related_tables")).containsKey("orders"));
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client()) {
      assertEquals(2, client.predict(request, SdmClient.KUMO_RELATIONAL).rows());
      assertEquals(2, client.predict(request, SdmClient.KUMO_RELATIONAL).rows());
    }
  }

  @Test void schemaSnapshotIgnoresCallerMutationBeforeFirstMaterialization() throws Exception {
    Schema schema;
    try (RootAllocator allocator = new RootAllocator(); Table users = users(allocator)) {
      schema = Schema.fromData(Map.of("users", users)).withPrimaryKey("users", "user_id")
          .withStype("users", "value", "numerical");
      // Transfer the caller table buffers to a mutable root, then change a context row.
      try (VectorSchemaRoot callerData = users.toVectorSchemaRoot()) {
        Float8Vector values = (Float8Vector) callerData.getVector("value");
        values.setSafe(2, 9999.0);
        assertEquals(9999.0, values.get(2));
      }
    }
    Request request = PredictiveQuery.parse(SCALAR).materialize(schema, options(1L, 2L).build());
    Map<String, Object> context = instance(payload(request, 0), "context");
    assertTrue(rows(context).stream().flatMap(List::stream)
        .noneMatch(value -> value instanceof Number n && n.doubleValue() == 9999.0));
  }

  @Test void temporalLinkRankingAndForecastUseTheSharedCore() throws Exception {
    Schema schema = fixture(true, true);
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client();
        RelationalMaterializer owner = new RelationalMaterializer(schema)) {
      Request temporal = PredictiveQuery.parse(TEMPORAL).materialize(owner, options(1L, 2L).anchorTime(LocalDate.of(2024, 6, 1)).build());
      assertTrue(temporal.targetMovesWithTime());
      assertEquals(2, client.predict(temporal, SdmClient.KUMO_RELATIONAL).rows());
      Request link = PredictiveQuery.parse(LINK).materialize(owner, options(1L, 2L).build());
      assertEquals("temporal_link_prediction", map(payload(link, 0).get("task")).get("kind"));
      PredictionResult linked = client.predict(link, SdmClient.KUMO_RELATIONAL);
      assertEquals(2, linked.rows()); assertTrue(linked.outputs().contains("rankings"));
      assertThrows(StructuredDataException.class,
          () -> PredictiveQuery.parse(RANK).materialize(owner, options(1L, 2L).build()));
      Request ranked = PredictiveQuery.parse(RANK).materialize(owner,
          options(1L, 2L).anchorTime(LocalDate.of(2024, 6, 1)).candidates(List.of(1L, 2L, 3L)).build());
      assertEquals(2, client.predict(ranked, SdmClient.KUMO_RELATIONAL).rows());
      Request forecast = PredictiveQuery.parse(FORECAST).materialize(owner, options(1L, 2L).build());
      PredictionResult forecasted = client.predict(forecast, SdmClient.KUMO_RELATIONAL);
      assertEquals(4, forecasted.rows());
      try (ArrowReader reader = forecasted.read(ArrowData.allocator())) {
        assertTrue(reader.loadNextBatch());
        assertNotNull(reader.getVectorSchemaRoot().getVector("forecast_step"));
      }
      assertThrows(StructuredDataException.class, () -> client.fit(ranked, SdmClient.KUMO_RELATIONAL));
      assertThrows(StructuredDataException.class, () -> client.fit(forecast, SdmClient.KUMO_RELATIONAL));
    }
  }

  @Test void independentOwnersAgreeAndSeededCallsResetNeighborSampling() throws Exception {
    Schema schema = fixture(false, false);
    PredictiveQuery query = PredictiveQuery.parse(SCALAR);
    MaterializeOptions seeded = options(1L, 2L).option("num_neighbors", List.of(3)).build();
    MaterializeOptions continuing = options(1L, 2L).option("random_seed", null)
        .option("num_neighbors", List.of(3)).build();
    try (RelationalMaterializer first = new RelationalMaterializer(schema);
        RelationalMaterializer second = new RelationalMaterializer(schema)) {
      Request initial = query.materialize(first, seeded);
      assertEquals(initial.payloads(), query.materialize(second, seeded).payloads());
      Object a = map(payload(query.materialize(first, continuing), 0).get("predict")).get("related_tables");
      Object b = map(payload(query.materialize(second, continuing), 0).get("predict")).get("related_tables");
      assertEquals(a, b);
      assertNotEquals(map(payload(initial, 0).get("predict")).get("related_tables"), a);
      assertEquals(initial.payloads(), query.materialize(first, seeded).payloads());
    }
  }

  @Test void concurrentCallsAreSerializedAndCloseIsIdempotent() throws Exception {
    RelationalMaterializer owner = new RelationalMaterializer(fixture(false, false));
    PredictiveQuery query = PredictiveQuery.parse(SCALAR);
    MaterializeOptions options = options(1L, 2L).build();
    try {
      List<CompletableFuture<List<String>>> calls = new ArrayList<>();
      for (int i = 0; i < 6; i++) calls.add(CompletableFuture.supplyAsync(
          () -> query.materialize(owner, options).payloads()));
      List<String> expected = calls.get(0).get(30, TimeUnit.SECONDS);
      for (CompletableFuture<List<String>> call : calls) assertEquals(expected, call.get(30, TimeUnit.SECONDS));
    } finally { owner.close(); owner.close(); }
    StructuredDataException closed = assertThrows(StructuredDataException.class,
        () -> query.materialize(owner, options));
    assertEquals("INVALID_CONFIGURATION", closed.code());
  }

  @Test void closeRacingAnAdmittedCallCannotReleaseItsGraphEarly() throws Exception {
    RelationalMaterializer owner = new RelationalMaterializer(fixture(false, false));
    PredictiveQuery query = PredictiveQuery.parse(SCALAR);
    query.materialize(owner, options(1L, 2L).build());
    CountDownLatch start = new CountDownLatch(1);
    CompletableFuture<Boolean> running = CompletableFuture.supplyAsync(() -> {
      try {
        start.await();
        return query.materialize(owner, options(1L, 2L).build()).predictRows() == 2;
      } catch (StructuredDataException closed) {
        assertEquals("INVALID_CONFIGURATION", closed.code()); return true;
      } catch (InterruptedException interrupted) { throw new AssertionError(interrupted); }
    });
    start.countDown(); owner.close();
    assertTrue(running.get(30, TimeUnit.SECONDS));
  }

  @Test void sessionsExcludeHoldoutAndSendOnlyTheScoringHalf() throws Exception {
    PredictiveQuery query = PredictiveQuery.parse(SCALAR);
    try (RelationalMaterializer owner = new RelationalMaterializer(fixture(false, false));
        Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client()) {
      Request opening = query.materialize(owner, options(1L, 2L).build());
      List<String> before = opening.payloads();
      try (FittedModel fitted = client.fit(opening, SdmClient.KUMO_RELATIONAL, List.of(3L, 4L))) {
        assertEquals(before, opening.payloads());
        Map<String, Object> uploaded = endpoint.created.get(0);
        assertFalse(uploaded.containsKey("predict"));
        assertEquals(opening.contextRows() - 2, rows(instance(uploaded, "context")).size());
        Set<Object> remaining = instanceIds(instance(uploaded, "context"));
        for (Object related : map(map(uploaded.get("context")).get("related_tables")).values())
          assertTrue(remaining.containsAll(instanceIds(map(related))));
        Request score = query.materialize(owner, options(3L, 4L).build());
        assertEquals(2, fitted.predict(score).rows()); assertEquals(2, fitted.predict(score).rows());
        assertTrue(endpoint.scored.stream().allMatch(body -> !body.containsKey("context") && !body.containsKey("task")));
      }
      assertEquals(1, endpoint.deletes.get());
    }
  }

  @Test void changedQuestionAndEmptyHeldContextAreRejectedBeforeNetwork() throws Exception {
    Schema schema = fixture(false, false);
    PredictiveQuery query = PredictiveQuery.parse(SCALAR);
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client()) {
      Request request = query.materialize(schema, options(1L, 2L).build());
      List<Long> all = new ArrayList<>(); for (long i = 1; i <= 20; i++) all.add(i);
      assertThrows(StructuredDataException.class, () -> client.fit(request, SdmClient.KUMO_RELATIONAL, all));
      assertTrue(endpoint.created.isEmpty());
      try (FittedModel fitted = client.fit(request, SdmClient.KUMO_RELATIONAL)) {
        Request changed = PredictiveQuery.parse("PREDICT users.value > 3 FOR EACH users.user_id")
            .materialize(schema, options(3L).build());
        assertThrows(StructuredDataException.class, () -> fitted.predict(changed));
        assertTrue(endpoint.scored.isEmpty());
      }
    }
  }

  @Test void equivalentQuoteStylesKeepTheSameSessionQuestion() throws Exception {
    Schema schema = fixture(false, false);
    Request opening = PredictiveQuery.parse("PREDICT users.group = 'a' FOR EACH users.user_id")
        .materialize(schema, options(1L, 2L).build());
    Request scoring = PredictiveQuery.parse("PREDICT users.group = \"a\" FOR EACH users.user_id")
        .materialize(schema, options(3L, 4L).build());
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client();
        FittedModel fitted = client.fit(opening, SdmClient.KUMO_RELATIONAL)) {
      assertEquals(2, fitted.predict(scoring).rows());
      assertEquals(1, endpoint.scored.size());
    }
  }

  @Test void ordinaryNumericCandidateTypesKeepExactIntegralValues() throws Exception {
    Schema schema = fixture(true, true);
    PredictiveQuery query = PredictiveQuery.parse(RANK);
    MaterializeOptions integral = options(1L, 2L).anchorTime(LocalDate.of(2024, 6, 1))
        .candidates(List.of(1L, 2L, 3L)).build();
    MaterializeOptions numeric = options(1L, 2L).anchorTime(LocalDate.of(2024, 6, 1))
        .candidates(List.of(1.0f, 2.0d, new BigDecimal("3"))).build();
    assertEquals(query.materialize(schema, integral).payloads(),
        query.materialize(schema, numeric).payloads());
  }

  @Test void laterSessionFailureRetainsAlreadyAnsweredRows() throws Exception {
    Schema schema = fixture(false, false);
    PredictiveQuery query = PredictiveQuery.parse(SCALAR);
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client();
        FittedModel fitted = client.fit(query.materialize(schema, options(1L).build()), SdmClient.KUMO_RELATIONAL)) {
      endpoint.failScoreOn = 2;
      Request score = query.materialize(schema, options(2L, 3L).rowsPerCall(1).build());
      StructuredDataException failed = assertThrows(StructuredDataException.class, () -> fitted.predict(score));
      assertEquals(1, ((Number) failed.details().get("calls_answered")).intValue());
      assertEquals(2, endpoint.scored.size());
      assertInstanceOf(ModelEndpointException.class, failed);
      assertEquals(1, ((Number) failed.details().get("rows_answered")).intValue());
      try (ArrowReader answered = failed.answered(ArrowData.allocator()).orElseThrow()) {
        assertTrue(answered.loadNextBatch());
        assertEquals(1, answered.getVectorSchemaRoot().getRowCount());
        assertFalse(answered.loadNextBatch());
      }
    }
  }

  @Test void failedDeleteCanRetryAndClientCloseAttemptsEverySession() throws Exception {
    Request request = PredictiveQuery.parse(SCALAR).materialize(fixture(false, false), options(1L).build());
    try (Endpoint endpoint = new Endpoint()) {
      SdmClient client = endpoint.client();
      FittedModel first = client.fit(request, SdmClient.KUMO_RELATIONAL);
      endpoint.deleteFailures.set(1);
      assertThrows(StructuredDataException.class, first::close);
      assertFalse(first.isClosed()); first.close(); first.close();
      assertTrue(first.isClosed()); assertEquals(2, endpoint.deletes.get());
      FittedModel second = client.fit(request, SdmClient.KUMO_RELATIONAL);
      FittedModel third = client.fit(request, SdmClient.KUMO_RELATIONAL);
      endpoint.deleteFailures.set(1);
      assertThrows(StructuredDataException.class, client::close);
      assertEquals(4, endpoint.deletes.get());
      assertTrue(second.isClosed() ^ third.isClosed());
      client.close(); assertEquals(4, endpoint.deletes.get());
    }
  }

  /** Real HTTP transport with protocol-valid IDs and deliberately reversed prediction rows. */
  static final class Endpoint implements AutoCloseable {
    final HttpServer server;
    final List<Map<String, Object>> created = Collections.synchronizedList(new ArrayList<>());
    final List<Map<String, Object>> scored = Collections.synchronizedList(new ArrayList<>());
    final AtomicInteger deletes = new AtomicInteger();
    final AtomicInteger deleteFailures = new AtomicInteger();
    final AtomicInteger scoreCalls = new AtomicInteger();
    volatile int failScoreOn;
    final Map<String, Map<String, Object>> sessions = new java.util.concurrent.ConcurrentHashMap<>();

    Endpoint() throws Exception {
      server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
      server.createContext("/", exchange -> {
        try {
          String path = exchange.getRequestURI().getPath();
          if (exchange.getRequestMethod().equals("DELETE")) {
            deletes.incrementAndGet();
            exchange.sendResponseHeaders(deleteFailures.getAndUpdate(n -> Math.max(0, n - 1)) > 0 ? 400 : 204, -1);
            return;
          }
          Map<String, Object> body = map(Json.read(exchange.getRequestBody().readAllBytes()));
          byte[] answer;
          if (path.equals("/v1/sessions")) {
            created.add(body); String id = "s" + created.size(); sessions.put(id, body);
            answer = Json.write(Map.of("session_id", id));
          } else {
            boolean session = path.contains("/sessions/");
            if (session) {
              scored.add(body);
              if (scoreCalls.incrementAndGet() == failScoreOn) {
                byte[] failure = Json.write(Map.of("detail", "second call failed", "code", "INVALID_REQUEST"));
                exchange.sendResponseHeaders(400, failure.length); exchange.getResponseBody().write(failure); return;
              }
            }
            Map<String, Object> task = session ? map(sessions.get(path.split("/")[3]).get("task")) : map(body.get("task"));
            Map<String, Object> table = instance(body, "predict");
            int id = ((List<?>) table.get("columns")).indexOf("instance_id");
            List<Map<String, Object>> predictions = new ArrayList<>();
            for (int at = rows(table).size() - 1; at >= 0; at--) {
              Map<String, Object> row = new LinkedHashMap<>();
              row.put("row_index", at); row.put("id", String.valueOf(rows(table).get(at).get(id)));
              if ("temporal_link_prediction".equals(task.get("kind"))) {
                row.put("rankings", List.of(Map.of("item", "3", "score", 0.9), Map.of("item", "1", "score", 0.4)));
              } else if ("regression".equals(task.get("kind"))) {
                row.put("prediction", 5.0 + at);
              } else {
                row.put("prediction", true); row.put("probabilities", Map.of("True", 0.9, "False", 0.1));
              }
              predictions.add(row);
            }
            answer = Json.write(Map.of("id", "relational-response", "model", SdmClient.KUMO_RELATIONAL,
                "metadata", Map.of(), "predictions", predictions));
          }
          exchange.sendResponseHeaders(200, answer.length); exchange.getResponseBody().write(answer);
        } catch (Exception error) {
          byte[] failure = error.toString().getBytes(StandardCharsets.UTF_8);
          exchange.sendResponseHeaders(500, failure.length); exchange.getResponseBody().write(failure);
        } finally { exchange.close(); }
      });
      server.start();
    }
    SdmClient client() {
      return SdmClient.connect("http://127.0.0.1:" + server.getAddress().getPort(),
          ClientOptions.builder().maxRetries(0).build());
    }
    @Override public void close() { server.stop(0); }
  }
}
