// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Json;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.dictionary.DictionaryProvider;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.table.Table;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;

/** Public Arrow Table journeys through JNI, Rust policies and actual HTTP. */
class TabularTest {
  static Table table(RootAllocator allocator, boolean target, long... ids) {
    List<FieldVector> vectors = new ArrayList<>();
    BigIntVector id = new BigIntVector("id", allocator);
    Float8Vector feature = new Float8Vector("x", allocator);
    id.allocateNew(ids.length); feature.allocateNew(ids.length);
    for (int i = 0; i < ids.length; i++) { id.setSafe(i, ids[i]); feature.setSafe(i, ids[i] * 2.5); }
    id.setValueCount(ids.length); feature.setValueCount(ids.length);
    vectors.add(id); vectors.add(feature);
    if (target) {
      Float8Vector y = new Float8Vector("y", allocator); y.allocateNew(ids.length);
      for (int i = 0; i < ids.length; i++) y.setSafe(i, ids[i] * 10.5);
      y.setValueCount(ids.length); vectors.add(y);
    }
    return new Table(vectors);
  }

  static final class Endpoint implements AutoCloseable {
    final HttpServer server;
    final List<Map<String, Object>> calls = new ArrayList<>();
    int creates;
    int deletes;
    boolean failDelete;
    int deleteFailuresRemaining;
    boolean wrongId;
    Endpoint() throws Exception {
      server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
      server.createContext("/", exchange -> {
        try {
          String path = exchange.getRequestURI().getPath();
          if (exchange.getRequestMethod().equals("DELETE")) {
            deletes++;
            if (failDelete || deleteFailuresRemaining-- > 0) { exchange.sendResponseHeaders(400, -1); return; }
            exchange.sendResponseHeaders(204, -1); return;
          }
          @SuppressWarnings("unchecked") Map<String,Object> body = (Map<String,Object>)Json.read(exchange.getRequestBody().readAllBytes());
          calls.add(body);
          byte[] answer;
          if (path.equals("/v1/sessions")) {
            creates++; answer = Json.write(Map.of("session_id", "s" + creates));
          } else {
            @SuppressWarnings("unchecked") Map<String,Object> predict = (Map<String,Object>)body.get("predict");
            @SuppressWarnings("unchecked") Map<String,Object> table = (Map<String,Object>)predict.get("instance_table");
            List<?> rows = (List<?>)table.get("rows");
            List<Object> predictions = new ArrayList<>();
            for (int at = rows.size() - 1; at >= 0; at--) {
              Map<String,Object> row = new java.util.LinkedHashMap<>();
              row.put("row_index", at); row.put("prediction", 42.5);
              if (wrongId) row.put("id", "[99999]");
              predictions.add(row);
            }
            answer = Json.write(Map.of("object", "prediction.response", "id", "r1",
                "model", "kumo-tabular", "metadata", Map.of(), "predictions", predictions));
          }
          exchange.sendResponseHeaders(200, answer.length);
          exchange.getResponseBody().write(answer);
        } catch (Exception e) {
          byte[] error = e.toString().getBytes(StandardCharsets.UTF_8);
          exchange.sendResponseHeaders(500, error.length); exchange.getResponseBody().write(error);
        } finally { exchange.close(); }
      });
      server.start();
    }
    String url() { return "http://127.0.0.1:" + server.getAddress().getPort(); }
    @Override public void close() { server.stop(0); }
  }

  @Test void directRequestSurvivesCallerClosureAndRestoresRowOrder() throws Exception {
    Request request;
    try (RootAllocator allocator = new RootAllocator();
        Table context = table(allocator, true, 1, 2);
        Table predict = table(allocator, false, 3, 4)) {
      request = new Request(context, predict, "y", "regression");
      assertEquals(2, context.getRowCount());
      assertEquals(2, predict.getRowCount());
    }
    try (Endpoint endpoint = new Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      assertEquals(2, request.contextRows()); assertEquals(2, request.predictRows());
      PredictionResult result = client.predict(request, SdmClient.KUMO_TABULAR);
      assertEquals(2, result.rows());
      try (ArrowReader reader = result.read(ArrowData.allocator())) {
        assertTrue(reader.loadNextBatch());
        assertEquals(0L, ((Number)reader.getVectorSchemaRoot().getVector("row_index").getObject(0)).longValue());
      }
      assertEquals(2, client.predict(request, SdmClient.KUMO_TABULAR).rows());
      endpoint.wrongId = true;
      assertThrows(StructuredDataException.class, () -> client.predict(request, SdmClient.KUMO_TABULAR));
    }
  }

  @Test void fitScoresWithoutResendingContextAndCleanupIsRetryable() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table context = table(allocator, true, 1, 2);
        Table predict = table(allocator, false, 3); Endpoint endpoint = new Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request = new Request(context, predict, "y", "regression");
      FittedModel fitted = client.fit(request, SdmClient.KUMO_TABULAR);
      assertEquals("s1", fitted.sessionId());
      assertEquals(1, fitted.predict(request).rows());
      assertFalse(endpoint.calls.get(1).containsKey("context"));
      endpoint.failDelete = true;
      assertThrows(StructuredDataException.class, fitted::close);
      assertFalse(fitted.isClosed());
      endpoint.failDelete = false; fitted.close(); fitted.close();
      assertTrue(fitted.isClosed()); assertEquals(2, endpoint.deletes);
      assertThrows(StructuredDataException.class, () -> fitted.predict(request));
    }
  }

  @Test void fittedHoldoutAndOverlapUseEntityKeys() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table context = table(allocator, true, 1, 2, 3);
        Table predict = table(allocator, false, 2); Endpoint endpoint = new Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request = new Request(context, predict, "y", "regression");
      try (FittedModel fitted = client.fit(request, SdmClient.KUMO_TABULAR, List.of(2L))) {
        assertEquals(1, fitted.predict(request).rows());
      }
      try (FittedModel fitted = client.fit(request, SdmClient.KUMO_TABULAR)) {
        assertEquals(1, fitted.predict(request).rows());
      }
    }
  }

  @Test void clientCloseAttemptsEverySessionWhenOneDeleteFails() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table context = table(allocator, true, 1, 2);
        Table scoring = table(allocator, false, 3); Endpoint endpoint = new Endpoint()) {
      SdmClient client = SdmClient.connect(endpoint.url());
      Request request = new Request(context, scoring, "y", "regression");
      FittedModel first = client.fit(request, SdmClient.KUMO_TABULAR);
      FittedModel second = client.fit(request, SdmClient.KUMO_TABULAR);
      endpoint.deleteFailuresRemaining = 1;
      assertThrows(StructuredDataException.class, client::close);
      assertEquals(2, endpoint.deletes);
      assertTrue(first.isClosed() ^ second.isClosed());
      assertThrows(StructuredDataException.class, () -> client.predict(request, SdmClient.KUMO_TABULAR));
      client.close();
      assertEquals(2, endpoint.deletes);
    }
  }

  @Test void pqlUsesOneTableAndDoesNotCallServerDuringMaterialize() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table all = table(allocator, true, 1, 2, 3, 4, 5, 6);
        Endpoint endpoint = new Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      Schema schema = Schema.fromData(Map.of("data", all));
      Request request = PredictiveQuery.parse("PREDICT data.y FOR EACH data.id").materialize(schema,
          MaterializeOptions.builder().indices(5L, 6L).numContextSamples(2).build());
      assertEquals(0, endpoint.calls.size());
      assertEquals(2, request.contextRows()); assertEquals(2, request.predictRows());
      assertEquals(2, client.predict(request, SdmClient.KUMO_TABULAR).rows());
      assertThrows(StructuredDataException.class, () -> PredictiveQuery.parse(
          "PREDICT data.y FOR EACH data.id").materialize(schema,
          MaterializeOptions.builder().indices(999L).build()));
    }
  }

  @Test void suppliedPqlScoringAndClientSessionCleanup() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table all = table(allocator, true, 1, 2, 3);
        Table scoring = table(allocator, false, 8, 9); Endpoint endpoint = new Endpoint()) {
      Schema schema = Schema.fromData(Map.of("data", all));
      Request request = PredictiveQuery.parse("PREDICT data.y FOR EACH data.id").materialize(schema, scoring);
      assertEquals(3, request.contextRows());
      assertEquals(2, request.predictRows());
      SdmClient client = SdmClient.connect(endpoint.url());
      FittedModel fitted = client.fit(request, SdmClient.KUMO_TABULAR);
      assertEquals(2, fitted.predict(request).rows());
      Request different = PredictiveQuery.parse("PREDICT data.y FOR EACH data.id ").materialize(schema, scoring);
      assertThrows(StructuredDataException.class, () -> fitted.predict(different));
      client.close();
      assertTrue(fitted.isClosed());
      assertEquals(1, endpoint.deletes);
      client.close();
      assertEquals(1, endpoint.deletes);
    }
  }

  @Test void pqlRetainsLogicalTimestampAndExactDecimalKeys() throws Exception {
    for (boolean large : new boolean[] {true, false}) {
    try (RootAllocator allocator = new RootAllocator()) {
      var id = new org.apache.arrow.vector.DecimalVector("id", allocator, 30, 2);
      var amount = new org.apache.arrow.vector.DecimalVector("amount", allocator, 30, 2);
      var time = new org.apache.arrow.vector.TimeStampMicroVector("when", allocator);
      var target = new Float8Vector("y", allocator);
      id.allocateNew(3); amount.allocateNew(3); time.allocateNew(3); target.allocateNew(3);
      for (int i = 0; i < 3; i++) {
        id.setSafe(i, new java.math.BigDecimal(large ? "9007199254740993.25" : "100.25").add(java.math.BigDecimal.valueOf(i)));
        amount.setSafe(i, new java.math.BigDecimal("3.25"));
        time.setSafe(i, 1_700_000_000_123456L + i); target.setSafe(i, i * 10.5);
      }
      id.setValueCount(3); amount.setValueCount(3); time.setValueCount(3); target.setValueCount(3);
      try (Table all = new Table(List.of(id, amount, time, target)); Endpoint endpoint = new Endpoint();
          SdmClient client = SdmClient.connect(endpoint.url())) {
        if (large) {
          StructuredDataException precision = assertThrows(StructuredDataException.class,
              () -> PredictiveQuery.parse("PREDICT data.y FOR EACH data.id")
                  .materialize(Schema.fromData(Map.of("data", all)), MaterializeOptions.builder()
                      .indices(new java.math.BigDecimal("9007199254740994.25")).build()));
          assertTrue(precision.getMessage().contains("without losing precision"));
          continue;
        }
        Request request = PredictiveQuery.parse("PREDICT data.y FOR EACH data.id")
            .materialize(Schema.fromData(Map.of("data", all)), MaterializeOptions.builder()
                .indices(new java.math.BigDecimal("101.25")).build());
        assertEquals(1, request.predictRows());
        assertEquals(1, client.predict(request, SdmClient.KUMO_TABULAR).rows());
        assertTrue(new String(Json.write(endpoint.calls.get(0)), StandardCharsets.UTF_8).contains("3.25"));
        try (ArrowReader reader = ArrowData.reader(request.predictData(), allocator)) {
          assertTrue(reader.loadNextBatch());
          assertTrue(reader.getVectorSchemaRoot().getSchema().findField("when").getType()
              instanceof ArrowType.Timestamp);
          assertEquals(1_700_000_000_123457L,
              ((org.apache.arrow.vector.TimeStampMicroVector) reader.getVectorSchemaRoot().getVector("when")).get(0));
        }
      }
    }
  }
    }

  @Test void pqlBigIntegerKeysDoNotRoundThroughDouble() {
    try (RootAllocator allocator = new RootAllocator();
        Table all = table(allocator, true, 9_007_199_254_740_993L, 9_007_199_254_740_994L)) {
      Request request = PredictiveQuery.parse("PREDICT data.y FOR EACH data.id")
          .materialize(Schema.fromData(Map.of("data", all)), MaterializeOptions.builder()
              .indices(new java.math.BigInteger("9007199254740993")).build());
      assertEquals(1, request.predictRows());
    }
  }

  @Test void snapshotRetainsNullAndValidNanSeparately() throws Exception {
    try (RootAllocator allocator = new RootAllocator()) {
      var id = new BigIntVector("id", allocator);
      var feature = new Float8Vector("x", allocator);
      var target = new Float8Vector("y", allocator);
      id.allocateNew(2); feature.allocateNew(2); target.allocateNew(2);
      id.setSafe(0, 1); id.setSafe(1, 2);
      feature.setNull(0); feature.setSafe(1, Double.NaN);
      target.setSafe(0, 1.5); target.setSafe(1, 3.5);
      id.setValueCount(2); feature.setValueCount(2); target.setValueCount(2);
      try (Table context = new Table(List.of(id, feature, target));
          Table scoring = table(allocator, false, 3); Endpoint endpoint = new Endpoint();
          SdmClient client = SdmClient.connect(endpoint.url())) {
        Request request = new Request(context, scoring, "y", "regression");
        try (ArrowReader reader = ArrowData.reader(request.contextData(), allocator)) {
          assertTrue(reader.loadNextBatch());
          var values = (Float8Vector) reader.getVectorSchemaRoot().getVector("x");
          assertTrue(values.isNull(0));
          assertFalse(values.isNull(1));
          assertTrue(Double.isNaN(values.get(1)));
        }
        assertEquals(1, client.predict(request, SdmClient.KUMO_TABULAR).rows());
      }
    }
  }

  @Test
  @org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable(named = "SDM_TABULAR_LIVE_URL", matches = ".+")
  void liveTabularDirectAndRepeatedSession() throws Exception {
    long[] ids = java.util.stream.LongStream.rangeClosed(1, 40).toArray();
    try (RootAllocator allocator = new RootAllocator(); Table context = table(allocator, true, ids);
        Table scoring = table(allocator, false, 41, 42);
        SdmClient client = SdmClient.connect(System.getenv("SDM_TABULAR_LIVE_URL"))) {
      Request request = new Request(context, scoring, "y", "regression");
      Map<String, Object> evidence = new java.util.LinkedHashMap<>();
      evidence.put("direct", liveRows(client.predict(request, SdmClient.KUMO_TABULAR), allocator));
      try (FittedModel fitted = client.fit(request, SdmClient.KUMO_TABULAR)) {
        evidence.put("first", liveRows(fitted.predict(request), allocator));
        evidence.put("second", liveRows(fitted.predict(request), allocator));
      }
      String destination = System.getenv("SDM_TABULAR_LIVE_EVIDENCE");
      if (destination != null) java.nio.file.Files.write(java.nio.file.Path.of(destination), Json.write(evidence));
    }
  }

  private static List<Object> liveRows(PredictionResult result, RootAllocator allocator) throws Exception {
    assertEquals(2, result.rows());
    List<Object> rows = new ArrayList<>();
    try (ArrowReader reader = result.read(allocator)) {
      while (reader.loadNextBatch()) {
        var root = reader.getVectorSchemaRoot();
        for (int row = 0; row < root.getRowCount(); row++) {
          Map<String, Object> values = new java.util.LinkedHashMap<>();
          for (FieldVector vector : root.getFieldVectors()) {
            Object value = vector.getObject(row);
            values.put(vector.getName(), value instanceof Number || value instanceof Boolean || value == null
                ? value : value.toString());
          }
          rows.add(values);
        }
      }
    }
    return rows;
  }

  @Test void snapshotPreparationMeasurement() {
    long[] ids = java.util.stream.LongStream.range(0, 10_000).toArray();
    try (RootAllocator allocator = new RootAllocator(); Table context = table(allocator, true, ids);
        Table scoring = table(allocator, false, 10_001)) {
      long before = System.nanoTime();
      Request request = new Request(context, scoring, "y", "regression");
      long nanos = System.nanoTime() - before;
      assertEquals(10_000, request.contextRows());
      assertEquals(10_000, context.getRowCount());
      System.out.println("Tabular 10k x 3 preparation: ms=" + nanos / 1_000_000.0
          + ", retained IPC bytes=" + (request.contextData().length + request.predictData().length)
          + ", caller Arrow allocated bytes=" + allocator.getAllocatedMemory());
    }
  }

  @Test void emptyScoringDoesNotContactModelAndInvalidTasksFailAtConstruction() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table context = table(allocator, true, 1, 2);
        Table empty = table(allocator, false); Endpoint endpoint = new Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request = new Request(context, empty, "y", "regression");
      assertEquals(0, client.predict(request, SdmClient.KUMO_TABULAR).rows());
      assertEquals(0, endpoint.calls.size());
      assertThrows(StructuredDataException.class, () -> new Request(context, empty, "y", "bogus"));
      assertThrows(StructuredDataException.class, () -> new Request(context, context, "y", "regression"));
    }
  }

  @Test void dictionaryInputPreservesValuesAndCallerOwnership() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); VarCharVector words = new VarCharVector("dictionary", allocator)) {
      words.allocateNew(); words.setSafe(0, "red".getBytes(StandardCharsets.UTF_8));
      words.setSafe(1, "blue".getBytes(StandardCharsets.UTF_8)); words.setValueCount(2);
      DictionaryEncoding encoding = new DictionaryEncoding(7L, false, new ArrowType.Int(32, true));
      Dictionary dictionary = new Dictionary(words, encoding);
      DictionaryProvider.MapDictionaryProvider provider = new DictionaryProvider.MapDictionaryProvider(dictionary);
      Field field = new Field("color", new FieldType(true, new ArrowType.Int(32, true), encoding), null);
      IntVector contextColor = (IntVector)field.createVector(allocator);
      contextColor.allocateNew(2); contextColor.setSafe(0, 0); contextColor.setSafe(1, 1); contextColor.setValueCount(2);
      Float8Vector y = new Float8Vector("y", allocator); y.allocateNew(2); y.setSafe(0, 2); y.setSafe(1, 3); y.setValueCount(2);
      IntVector predictColor = (IntVector)field.createVector(allocator);
      predictColor.allocateNew(1); predictColor.setSafe(0, 1); predictColor.setValueCount(1);
      try (Table context = new Table(List.of(contextColor, y), 2, provider);
          Table predict = new Table(List.of(predictColor), 1, provider);
          Endpoint endpoint = new Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
        try (var reader = com.nvidia.sdm.internal.ArrowData.reader(com.nvidia.sdm.internal.ArrowTabular.snapshot(context), allocator)) {
          assertTrue(reader.loadNextBatch());
        }
        Request request = new Request(context, predict, "y", "regression");
        assertEquals(2, context.getRowCount());
        assertEquals(1, client.predict(request, SdmClient.KUMO_TABULAR).rows());
        assertTrue(Json.write(endpoint.calls.get(0)).length > 0);
        assertTrue(new String(Json.write(endpoint.calls.get(0)), StandardCharsets.UTF_8).contains("blue"));
      }
    }
  }
}
