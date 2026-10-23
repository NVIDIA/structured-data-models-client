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

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Trace;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Java S1 end to end (T3.12): a query materialized on the core over each database, the request's
 * attributes, its prediction through the model transport over {@code java.net.http}, the result
 * read as Arrow and inspected, and the transport sending the core's call as the bytes the core
 * wrote (design choice 10, "Model calls"; the brief's exit 4).
 */
class T312PredictTest {
  private static final String BINARY =
      "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id";
  private static final String REGRESSION =
      "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";
  private static final String MULTICLASS = "PREDICT users.country FOR EACH users.user_id";

  private static MaterializeOptions twoUsers() {
    return MaterializeOptions.builder().indices(List.of(0L, 1L)).anchorTime(LocalDate.of(2024, 3, 1)).build();
  }

  /** The bodies of the model calls the core wrote, as the listener is told them. */
  static final class Calls implements Trace {
    final List<byte[]> bodies = Collections.synchronizedList(new ArrayList<>());
    private Integer asked;

    @Override
    public void stepped(Map<String, Object> step) {
      asked = "call_model".equals(step.get("step")) ? ((Number) step.get("body")).intValue() : null;
    }

    @Override
    public void taken(int id, Object value, boolean arrow) {
      if (asked != null && asked == id && !arrow) {
        bodies.add((byte[]) value);
      }
      asked = null;
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"duckdb", "sqlite", "postgres"})
  void eachKindOfQueryIsMaterializedAndPredicted(String database) throws Exception {
    try (Connection connection = T312Shop.open(database);
        T312Shop.Endpoint endpoint = new T312Shop.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Schema schema = Schema.fromSource(T312Shop.source(database, connection));
      for (String query : List.of(BINARY, REGRESSION, MULTICLASS)) {
        Request request = PredictiveQuery.parse(query).materialize(schema, twoUsers());
        assertEquals("users", request.entityTable());
        assertEquals(2, request.predictRows());
        assertTrue(request.contextRows() > 0);
        assertFalse(request.nothingToScore());
        assertTrue(request.isRelational());
        assertEquals(1, request.payloads().size());
        assertEquals(query.contains("COUNT") || query.contains("SUM"), request.targetMovesWithTime());

        PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL);
        assertEquals("kumo-relational", result.model());
        assertEquals(2, result.rows());
        try (ArrowReader reader = result.read(ArrowData.allocator())) {
          assertTrue(reader.loadNextBatch());
          VectorSchemaRoot rows = reader.getVectorSchemaRoot();
          assertEquals(2, rows.getRowCount());
          assertEquals(0L, ((Number) rows.getVector("row_index").getObject(0)).longValue());
          assertTrue(rows.getSchema().findField("prediction") != null);
        }
        assertEquals(List.of(), result.absent(List.of("prediction")));
        assertEquals(List.of("explanation"), result.absent(List.of("explanation")));
        assertNull(result.explanation());
      }
    }
  }

  @Test
  void anExplanationIsReadFromTheResult() throws Exception {
    try (Connection connection = T312Shop.open("duckdb");
        T312Shop.Endpoint endpoint = new T312Shop.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request =
          PredictiveQuery.parse(BINARY)
              .materialize(
                  Schema.fromSource(DuckDbSource.of(connection)),
                  MaterializeOptions.builder().indices(List.of(0L)).anchorTime(LocalDate.of(2024, 3, 1)).build());
      PredictionResult result =
          client.predict(request, SdmClient.KUMO_RELATIONAL, List.of("prediction", "explanation"));
      assertEquals(List.of(), result.absent(List.of("prediction", "explanation")));
      assertEquals("scripted", result.explanation().get("format"));
      assertEquals(List.of("prediction", "explanation"), result.outputs());
    }
  }

  @Test
  void theTransportSendsTheCoresBytesUnchanged() throws Exception {
    Calls calls = new Calls();
    try (Connection connection = T312Shop.open("duckdb");
        T312Shop.Endpoint endpoint = new T312Shop.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Schema schema = Schema.fromSource(DuckDbSource.of(connection));
      Request request =
          PredictiveQuery.parse(BINARY)
              .materialize(
                  schema,
                  MaterializeOptions.builder()
                      .indices(List.of(0L, 1L, 2L, 3L, 4L))
                      .anchorTime(LocalDate.of(2024, 3, 1))
                      .rowsPerCall(2)
                      .build());
      assertEquals(3, request.payloads().size());
      Trace.install(calls);
      try {
        assertEquals(5, client.predict(request, SdmClient.KUMO_RELATIONAL).rows());
      } finally {
        Trace.install(null);
      }
      assertEquals(3, calls.bodies.size());
      assertEquals(calls.bodies.size(), endpoint.received.size());
      for (int at = 0; at < calls.bodies.size(); at++) {
        assertArrayEquals(calls.bodies.get(at), endpoint.received.get(at), "call " + at);
      }
    }
  }

  @Test
  void anEndpointsRefusalIsRaisedWithItsStatusAndWords() throws Exception {
    byte[] refusal =
        "{\"code\": \"BAD_INPUT\", \"detail\": \"the endpoint's own words\", \"field\": \"x\"}"
            .getBytes(StandardCharsets.UTF_8);
    try (Connection connection = T312Shop.open("duckdb");
        T312Shop.Endpoint endpoint = new T312Shop.Endpoint(body -> Map.entry(400, refusal));
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request =
          PredictiveQuery.parse(BINARY).materialize(Schema.fromSource(DuckDbSource.of(connection)), twoUsers());
      StructuredDataException raised =
          assertThrows(StructuredDataException.class, () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      ModelEndpointException refused = assertInstanceOf(ModelEndpointException.class, raised);
      assertEquals(400, refused.status());
      assertEquals("BAD_INPUT", refused.code());
      assertEquals("the endpoint's own words", refused.getMessage());
      assertEquals("x", refused.details().get("field"));
    }
  }

  @Test
  void anAnswerThatIsNotJsonIsATransportError() throws Exception {
    try (Connection connection = T312Shop.open("duckdb");
        T312Shop.Endpoint endpoint =
            new T312Shop.Endpoint(body -> Map.entry(200, "not json".getBytes(StandardCharsets.UTF_8)));
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request =
          PredictiveQuery.parse(BINARY).materialize(Schema.fromSource(DuckDbSource.of(connection)), twoUsers());
      StructuredDataException raised =
          assertThrows(StructuredDataException.class, () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      assertEquals("TRANSPORT_ERROR", raised.code());
    }
  }

  @Test
  void anAnswerNestedPastTheReaderIsTheCoresToRefuse() throws Exception {
    byte[] deep = ("{\"a\": " + "[".repeat(5_000) + "]".repeat(5_000) + "}").getBytes(StandardCharsets.UTF_8);
    try (Connection connection = T312Shop.open("duckdb");
        T312Shop.Endpoint endpoint = new T312Shop.Endpoint(body -> Map.entry(200, deep));
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request =
          PredictiveQuery.parse(BINARY).materialize(Schema.fromSource(DuckDbSource.of(connection)), twoUsers());
      StructuredDataException raised =
          assertThrows(StructuredDataException.class, () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      assertEquals("INVALID_RESPONSE", raised.code());
    }
  }

  @Test
  void anUnreachableEndpointIsATransportError() throws Exception {
    String url;
    try (T312Shop.Endpoint gone = new T312Shop.Endpoint()) {
      url = gone.url();
    }
    try (Connection connection = T312Shop.open("duckdb");
        SdmClient client =
            SdmClient.connect(url, ClientOptions.builder().maxRetries(0).build())) {
      Request request =
          PredictiveQuery.parse(BINARY).materialize(Schema.fromSource(DuckDbSource.of(connection)), twoUsers());
      StructuredDataException raised =
          assertThrows(StructuredDataException.class, () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      assertEquals("TRANSPORT_ERROR", raised.code());
    }
  }

  @Test
  void theModelsCapabilitiesAreTheCoresTable() {
    try (SdmClient client = SdmClient.connect("http://localhost:1")) {
      ModelCapabilities relational = client.capabilities(SdmClient.KUMO_RELATIONAL);
      assertEquals("kumo-relational", relational.model());
      assertTrue(relational.tasks().contains("binary_classification"));
      assertTrue(relational.outputs().contains("prediction"));
      assertTrue(relational.multipleTables() && relational.relationships() && relational.singleTable());
      assertEquals("sdk", relational.source());
      assertFalse(client.capabilities("kumo-tabular").relationships());
      assertEquals("UNKNOWN_MODEL", assertThrows(StructuredDataException.class, () -> client.capabilities("nope")).code());
      assertEquals("http://localhost:1", client.url());
    }
  }

  @Test
  void aUrlThatIsNoBaseEndpointIsRefused() {
    for (String url : List.of("ftp://host", "http://", "http://host/?api_key=x")) {
      assertEquals(
          "INVALID_CONFIGURATION",
          assertThrows(StructuredDataException.class, () -> SdmClient.connect(url)).code(),
          url);
    }
    assertEquals(
        "INVALID_CONFIGURATION",
        assertThrows(
                StructuredDataException.class,
                () -> SdmClient.connect("http://example.com", ClientOptions.builder().apiKey("k").build()))
            .code());
  }

  @Test
  void materializeNeedsTheEntities() throws Exception {
    try (Connection connection = T312Shop.open("duckdb")) {
      Schema schema = Schema.fromSource(DuckDbSource.of(connection));
      assertEquals(
          "INVALID_REQUEST",
          assertThrows(
                  StructuredDataException.class,
                  () -> PredictiveQuery.parse(BINARY).materialize(schema, MaterializeOptions.builder().build()))
              .code());
      assertEquals(
          "INVALID_REQUEST",
          assertThrows(
                  StructuredDataException.class,
                  () ->
                      PredictiveQuery.parse(BINARY)
                          .materialize(schema, MaterializeOptions.builder().indices(0L).option("no_such_option", 1).build()))
              .code());
    }
  }
}
