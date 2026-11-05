// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static com.nvidia.sdm.RelationalTest.*;
import static org.junit.jupiter.api.Assertions.*;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Json;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Public regressions for task coverage across Arrow and JDBC inputs. */
class ModelConnectorCoverageTest {
  static Schema linkSchema(String provider, Connection connection) throws Exception {
    try (Statement sql = connection.createStatement()) {
      sql.execute("CREATE TABLE items (item_id BIGINT, price DOUBLE)");
      for (int i = 1; i <= 5; i++) sql.execute("INSERT INTO items VALUES (" + i + ", " + i + ".5)");
      sql.execute("ALTER TABLE orders ADD COLUMN item_id BIGINT");
      sql.execute("UPDATE orders SET item_id = order_id % 5 + 1");
    }
    return WarehouseSessionTest.schema(provider, connection).withPrimaryKey("items", "item_id")
        .withRelationships(List.of(new Relationship("orders", "user_id", "users", "user_id"),
            new Relationship("orders", "item_id", "items", "item_id")));
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void jdbcRankingRetainsTypedCandidatesAndPredictsAfterConnectionCloses(String provider) throws Exception {
    Request request;
    try (Connection connection = WarehouseSessionTest.open(provider)) {
      Schema schema = linkSchema(provider, connection);
      request = PredictiveQuery.parse(RANK).materialize(schema,
          WarehouseSessionTest.asked(1L, 2L)
              .candidates(List.of(1L, new java.math.BigDecimal("1.0"), 2, 3L)).build());
    }
    assertEquals(3, request.payloads().size());
    List<String> original = request.payloads();
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client()) {
      for (int attempt = 0; attempt < 2; attempt++) {
        PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL);
        assertEquals(2, result.rows());
        try (ArrowReader reader = result.read(ArrowData.allocator())) {
          assertTrue(reader.loadNextBatch());
          List<?> ranked = (List<?>) reader.getVectorSchemaRoot().getVector("rankings").getObject(0);
          assertEquals(List.of("1", "2"), ranked.stream().map(Object::toString).toList());
        }
      }
      assertEquals(original, request.payloads());
      assertThrows(StructuredDataException.class, () -> client.fit(request, SdmClient.KUMO_RELATIONAL));
    }
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void jdbcForecastRetainsStepsAndPredictsAfterConnectionCloses(String provider) throws Exception {
    Request request;
    try (Connection connection = WarehouseSessionTest.open(provider)) {
      request = PredictiveQuery.parse(FORECAST).materialize(
          WarehouseSessionTest.schema(provider, connection), WarehouseSessionTest.asked(1L, 2L).build());
    }
    assertEquals(2, request.payloads().size());
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client()) {
      PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL);
      assertEquals(4, result.rows());
      try (ArrowReader reader = result.read(ArrowData.allocator())) {
        assertTrue(reader.loadNextBatch());
        assertNotNull(reader.getVectorSchemaRoot().getVector("forecast_step"));
      }
      assertThrows(StructuredDataException.class, () -> client.fit(request, SdmClient.KUMO_RELATIONAL));
    }
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void failedLaterForecastKeepsCompletedRows(String provider) throws Exception {
    Request request;
    try (Connection connection = WarehouseSessionTest.open(provider)) {
      request = PredictiveQuery.parse(FORECAST).materialize(
          WarehouseSessionTest.schema(provider, connection), WarehouseSessionTest.asked(1L, 2L).build());
    }
    java.util.concurrent.atomic.AtomicInteger calls = new java.util.concurrent.atomic.AtomicInteger();
    try (Shop.Endpoint endpoint = new Shop.Endpoint(body -> calls.incrementAndGet() == 1
        ? Map.entry(200, Shop.scripted(body))
        : Map.entry(400, "{\"detail\":\"later step failed\"}".getBytes(StandardCharsets.UTF_8)));
        SdmClient client = SdmClient.connect(endpoint.url(), ClientOptions.builder().maxRetries(0).build())) {
      StructuredDataException error = assertThrows(StructuredDataException.class,
          () -> client.predict(request, SdmClient.KUMO_RELATIONAL));
      assertEquals(2, calls.get());
      try (ArrowReader reader = error.answered(ArrowData.allocator()).orElseThrow()) {
        assertTrue(reader.loadNextBatch()); assertEquals(2, reader.getVectorSchemaRoot().getRowCount());
        assertNotNull(reader.getVectorSchemaRoot().getVector("forecast_step"));
      }
    }
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void jdbcTemporalLinksPredictAndReuseSessionsAfterConnectionCloses(String provider) throws Exception {
    Request request;
    try (Connection connection = WarehouseSessionTest.open(provider)) {
      request = PredictiveQuery.parse(LINK).materialize(linkSchema(provider, connection),
          WarehouseSessionTest.asked(1L, 2L).build());
    }
    Map<String, Object> body = payload(request, 0);
    assertEquals("temporal_link_prediction", map(body.get("task")).get("kind"));
    assertEquals(List.of("users", "items"), map(body.get("task")).get("entity_table_names"));
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client()) {
      PredictionResult direct = client.predict(request, SdmClient.KUMO_RELATIONAL);
      assertEquals(2, direct.rows()); assertTrue(direct.outputs().contains("rankings"));
      try (FittedModel fitted = client.fit(request, SdmClient.KUMO_RELATIONAL)) {
        assertEquals(2, fitted.predict(request).rows());
        assertEquals(2, fitted.predict(request).rows());
        assertEquals(1, endpoint.created.size()); assertEquals(2, endpoint.scored.size());
        assertTrue(endpoint.scored.stream().noneMatch(value -> value.containsKey("context")));
      }
      assertEquals(1, endpoint.deletes.get());
    }
  }

  static Table classificationTable(RootAllocator allocator, boolean target, int classes) {
    int count = target ? 12 : 2;
    List<FieldVector> vectors = new ArrayList<>();
    BigIntVector id = new BigIntVector("id", allocator); id.allocateNew(count);
    Float8Vector x = new Float8Vector("x", allocator); x.allocateNew(count);
    for (int i = 0; i < count; i++) { id.setSafe(i, target ? i : i + 20); x.setSafe(i, i * 1.25); }
    id.setValueCount(count); x.setValueCount(count); vectors.add(id); vectors.add(x);
    if (target) {
      VarCharVector y = new VarCharVector("y", allocator); y.allocateNew();
      for (int i = 0; i < count; i++) y.setSafe(i, ("class" + i % classes).getBytes(StandardCharsets.UTF_8));
      y.setValueCount(count); vectors.add(y);
    }
    return new Table(vectors);
  }

  static Map.Entry<Integer, byte[]> tabularClassificationAnswer(byte[] body) {
    try {
      Map<String, Object> call = map(Json.read(body));
      List<List<Object>> scoring = rows(instance(call, "predict"));
      List<Map<String, Object>> predictions = new ArrayList<>();
      for (int at = scoring.size() - 1; at >= 0; at--)
        predictions.add(Map.of("row_index", at, "prediction", "class0"));
      return Map.entry(200, Json.write(Map.of("model", SdmClient.KUMO_TABULAR,
          "id", "classification-response", "metadata", Map.of(), "predictions", predictions)));
    } catch (Exception error) { throw new AssertionError(error); }
  }

  @ParameterizedTest @ValueSource(ints = {2, 3})
  void directTabularClassificationRunsThroughHttpAndRestoresLabels(int classes) throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Table context = classificationTable(allocator, true, classes);
        Table scoring = classificationTable(allocator, false, 0);
        Shop.Endpoint endpoint = new Shop.Endpoint(ModelConnectorCoverageTest::tabularClassificationAnswer);
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request = new Request(context, scoring, "y", "classification");
      PredictionResult result = client.predict(request, SdmClient.KUMO_TABULAR);
      assertEquals(2, result.rows()); assertEquals(SdmClient.KUMO_TABULAR, result.model());
      assertEquals("classification", map(map(Json.read(endpoint.received.get(0))).get("task")).get("kind"));
      try (ArrowReader reader = result.read(ArrowData.allocator())) {
        assertTrue(reader.loadNextBatch());
        assertEquals("class0", reader.getVectorSchemaRoot().getVector("prediction").getObject(0).toString());
      }
    }
  }

  @Test void aSingleRelatedTableDefaultsToRelationalButAnUnrelatedTableStaysTabular() throws Exception {
    try (RootAllocator allocator = new RootAllocator()) {
      BigIntVector id = new BigIntVector("id", allocator); id.allocateNew(12);
      BigIntVector parent = new BigIntVector("parent", allocator); parent.allocateNew(12);
      Float8Vector value = new Float8Vector("value", allocator); value.allocateNew(12);
      for (int i = 0; i < 12; i++) {
        id.setSafe(i, i + 1); parent.setSafe(i, (i + 1) % 12 + 1); value.setSafe(i, i * 1.25);
      }
      id.setValueCount(12); parent.setValueCount(12); value.setValueCount(12);
      try (Table input = new Table(List.of(id, parent, value))) {
        Schema plain = Schema.fromData(Map.of("users", input)).withPrimaryKey("users", "id")
            .withStype("users", "value", "numerical").withRelationships(List.of());
        PredictiveQuery query = PredictiveQuery.parse("PREDICT users.value FOR EACH users.id");
        MaterializeOptions options = MaterializeOptions.builder().indices(1L, 2L).build();
        Request tabular = query.materialize(plain, options);
        assertFalse(tabular.isRelational()); assertTrue(tabular.payloads().isEmpty());
        Schema related = plain.withRelationship(new Relationship("users", "parent", "users", "id"));
        Request relational = query.materialize(related, options);
        assertTrue(relational.isRelational()); assertFalse(relational.payloads().isEmpty());
        try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client()) {
          assertEquals(2, client.predict(relational, SdmClient.KUMO_RELATIONAL).rows());
        }
      }
    }
  }

  @Test void frameRelationalMulticlassUsesCategoricalTarget() throws Exception {
    Schema schema = fixture(false, false).withStype("users", "group", "categorical");
    Request request = PredictiveQuery.parse("PREDICT users.group FOR EACH users.user_id")
        .materialize(schema, options(1L, 2L).build());
    assertEquals("multiclass_classification", map(payload(request, 0).get("task")).get("kind"));
    try (Shop.Endpoint endpoint = new Shop.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      assertEquals(2, client.predict(request, SdmClient.KUMO_RELATIONAL).rows());
    }
  }
}
