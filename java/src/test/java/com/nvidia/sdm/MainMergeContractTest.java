// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.Trace;
import java.sql.Connection;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

/** Public main contracts must execute the shared engine, including refusal paths. */
class MainMergeContractTest {
  static final class Operations implements Trace, AutoCloseable {
    final List<String> calls = new ArrayList<>();
    int statements;
    final Trace previous = Trace.installed();
    Operations() { Trace.install(this); }
    @Override public void started(byte[] message, Map<Integer, Object> attachments) {
      try {
        Map<String, Object> operation = RelationalTest.map(
            RelationalTest.map(Json.read(message)).get("operation"));
        calls.add(operation.get("op") + (operation.containsKey("action") ? ":" + operation.get("action") : ""));
      } catch (Exception error) { throw new AssertionError(error); }
    }
    @Override public void statement(int handle, String sql) { statements++; }
    void require(String... required) {
      for (String name : required) assertTrue(calls.contains(name), () -> name + " absent from " + calls);
    }
    @Override public void close() { Trace.install(previous); }
  }

  @Test void emptyExplainOutputsNeverReachEitherModel() throws Exception {
    for (String model : List.of(SdmClient.KUMO_TABULAR, SdmClient.KUMO_RELATIONAL)) {
      Schema schema;
      try (RootAllocator allocator = new RootAllocator(); Table users = RelationalTest.users(allocator)) {
        schema = model.equals(SdmClient.KUMO_TABULAR)
            ? Schema.fromData(Map.of("users", users)).withPrimaryKey("users", "user_id")
                .withStype("users", "value", "numerical")
            : RelationalTest.fixture(true, false);
      }
      MaterializeOptions.Builder options = MaterializeOptions.builder().model(model).indices(1L, 2L).numContextSamples(12);
      if (model.equals(SdmClient.KUMO_RELATIONAL)) options.randomSeed(7);
      Request request = PredictiveQuery.parse("EXPLAIN " + RelationalTest.SCALAR).materialize(schema, options.build());
      try (TabularTest.Endpoint endpoint = new TabularTest.Endpoint();
          SdmClient client = SdmClient.connect(endpoint.url()); Operations operations = new Operations()) {
        StructuredDataException predictionError = assertThrows(StructuredDataException.class,
            () -> client.predict(request, model, List.of()));
        StructuredDataException fitError = assertThrows(StructuredDataException.class,
            () -> client.fit(request, model, List.of(), List.of()));
        assertTrue(predictionError.getMessage().contains("outputs"), predictionError::getMessage);
        assertTrue(fitError.getMessage().contains("outputs"), fitError::getMessage);
        assertTrue(endpoint.calls.isEmpty(), "invalid outputs must not make HTTP calls");
        if (model.equals(SdmClient.KUMO_TABULAR)) operations.require("tabular_arrow:predict", "tabular_arrow:fit");
        else operations.require("predict", "frame_session:fit");
      }
    }
  }

  @Test void tabularSessionAndMergeUseNativeOperations() throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Table context = TabularTest.table(allocator, true, 1, 2, 3);
        Table predict = TabularTest.table(allocator, false, 4, 5);
        TabularTest.Endpoint endpoint = new TabularTest.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url()); Operations operations = new Operations()) {
      endpoint.gradients = true;
      Request request = new Request(context, predict, "y", "regression");
      PredictionResult direct = client.predict(request, SdmClient.KUMO_TABULAR, List.of("prediction", "explanation"));
      try (FittedModel fitted = client.fit(request, SdmClient.KUMO_TABULAR, List.of(), List.of("prediction", "explanation"))) {
        PredictionResult scored = fitted.predict(request);
        assertEquals(4, PredictionResult.merge(List.of(direct, scored)).explanations().gradients().entries().size());
      }
      operations.require("tabular_arrow:predict", "tabular_arrow:fit", "tabular_arrow:score", "tabular_arrow:close", "merge_results");
      assertEquals(1, endpoint.deletes);
      assertTrue(endpoint.calls.stream().noneMatch(body -> body.containsKey("client_generated_keys")));
    }
  }

  @Test void explainRankingIsRefusedBeforeWarehouseReads() throws Exception {
    try (Connection connection = WarehouseSessionTest.open("sqlite")) {
      try (var sql = connection.createStatement()) {
        sql.execute("ALTER TABLE orders ADD COLUMN item_id BIGINT");
        sql.execute("UPDATE orders SET item_id = order_id % 5 + 1");
        sql.execute("CREATE TABLE items (item_id BIGINT PRIMARY KEY)");
        sql.execute("INSERT INTO items VALUES (1), (2), (3), (4), (5)");
      }
      Schema schema = WarehouseSessionTest.schema("sqlite", connection)
          .withPrimaryKey("items", "item_id")
          .withRelationships(List.of(new Relationship("orders", "user_id", "users", "user_id"),
              new Relationship("orders", "item_id", "items", "item_id")));
      try (Operations operations = new Operations()) {
        StructuredDataException error = assertThrows(StructuredDataException.class,
            () -> PredictiveQuery.parse("EXPLAIN " + RelationalTest.RANK).materialize(schema,
                WarehouseSessionTest.asked(1L, 2L).candidates(List.of(1L, 2L)).build()));
        assertTrue(error.getMessage().contains("explanation"), error::getMessage);
        assertEquals(0, operations.statements, "unsupported query must fail before source sampling");
        operations.require("materialize");
      }
    }
  }

  @Test void operationAssertionRejectsUnrelatedNativeWork() {
    try (Operations operations = new Operations()) {
      operations.calls.add("describe");
      assertThrows(AssertionError.class, () -> operations.require("predict"));
    }
  }
  static Map<String, Object> gradient(Map<String, Object> payload, int index) {
    Map<String, Object> predict = RelationalTest.map(payload.get("predict"));
    Map<String, Object> instance = RelationalTest.map(predict.get("instance_table"));
    int ownerColumn = ((List<?>) instance.get("columns")).indexOf("instance_id");
    Object owner = RelationalTest.rows(instance).get(index).get(ownerColumn);
    Map<String, Object> blocks = new java.util.LinkedHashMap<>();
    for (var entry : RelationalTest.map(predict.get("related_tables")).entrySet()) {
      Map<String, Object> table = RelationalTest.map(entry.getValue());
      int at = ((List<?>) table.get("columns")).indexOf("instance_id");
      List<Integer> positions = new ArrayList<>();
      List<List<Object>> gradients = new ArrayList<>();
      List<List<Object>> rows = RelationalTest.rows(table);
      for (int row = 0; row < rows.size(); row++) {
        if (owner.equals(rows.get(row).get(at))) { positions.add(row); gradients.add(List.of()); }
      }
      blocks.put(entry.getKey(), Map.of("row_indices", positions, "columns", List.of(), "gradients", List.of(gradients)));
    }
    return Map.of("format", "sdm_gradient_v1", "details", Map.of(
        "input_space", "preprocessed_numerical",
        "outputs", List.of(Map.of("kind", "prediction", "statistic", "mean")),
        "instance_table", Map.of("row_indices", List.of(index), "columns", List.of(), "gradients", List.of(List.of(List.of()))),
        "related_tables", blocks));
  }

  @Test void generatedKeysStayInternalAcrossPersistenceAndRelationalSessions() throws Exception {
    Schema schema = RelationalTest.fixture(false, false).withoutPrimaryKey("orders");
    Request request = PredictiveQuery.parse("EXPLAIN " + RelationalTest.SCALAR)
        .materialize(schema, RelationalTest.options(1L, 2L).build());
    var saved = new java.io.ByteArrayOutputStream();
    request.writeTo(saved);
    Request restored = Request.readFrom(new java.io.ByteArrayInputStream(saved.toByteArray()));
    assertTrue(RelationalTest.payload(restored, 0).containsKey("client_generated_keys"));
    try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint();
        SdmClient client = endpoint.client(); Operations operations = new Operations()) {
      endpoint.gradients = true;
      PredictionResult direct = client.predict(restored, SdmClient.KUMO_RELATIONAL);
      try (FittedModel fitted = client.fit(restored, SdmClient.KUMO_RELATIONAL)) {
        PredictionResult scored = fitted.predict(restored);
        for (PredictionResult result : List.of(direct, scored)) {
          assertEquals(2, result.explanations().gradients().entries().size());
          for (Map<String, Object> entry : result.explanations().gradients().entries().values()) {
            Map<String, Object> orders = RelationalTest.map(RelationalTest.map(entry.get("related")).get("orders"));
            assertNotNull(orders);
            assertFalse(orders.containsKey("keys"));
            for (Object row : (List<?>) orders.get("rows")) assertFalse(RelationalTest.map(row).containsKey("__node_id"));
          }
        }
      }
      operations.require("predict", "frame_session:fit", "frame_session:predict", "frame_session:close");
      assertTrue(endpoint.created.stream().noneMatch(body -> body.containsKey("client_generated_keys")));
      assertTrue(endpoint.scored.stream().noneMatch(body -> body.containsKey("client_generated_keys")));
      assertEquals(1, endpoint.deletes.get());
    }
  }

  @Test void genuineNodeIdNamedSourceKeyIsNotHidden() throws Exception {
    Schema schema;
    try (RootAllocator allocator = new RootAllocator(); Table users = RelationalTest.users(allocator);
        var ids = new org.apache.arrow.vector.BigIntVector("__node_id", allocator);
        var owners = new org.apache.arrow.vector.BigIntVector("user_id", allocator);
        var prices = new org.apache.arrow.vector.Float8Vector("price", allocator)) {
      ids.allocateNew(60); owners.allocateNew(60); prices.allocateNew(60);
      for (int index = 0; index < 60; index++) {
        ids.setSafe(index, index + 100); owners.setSafe(index, index % 20 + 1); prices.setSafe(index, index * 1.5);
      }
      ids.setValueCount(60); owners.setValueCount(60); prices.setValueCount(60);
      try (Table orders = new Table(List.of(ids, owners, prices))) {
        schema = Schema.fromData(Map.of("users", users, "orders", orders))
            .withPrimaryKey("users", "user_id").withPrimaryKey("orders", "__node_id")
            .withStype("users", "value", "numerical")
            .withRelationships(List.of(new Relationship("orders", "user_id", "users", "user_id")));
      }
    }
    Request request = PredictiveQuery.parse("EXPLAIN " + RelationalTest.SCALAR)
        .materialize(schema, RelationalTest.options(1L, 2L).build());
    try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint(); SdmClient client = endpoint.client()) {
      endpoint.gradients = true;
      PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL);
      for (Map<String, Object> entry : result.explanations().gradients().entries().values()) {
        Map<String, Object> orders = RelationalTest.map(RelationalTest.map(entry.get("related")).get("orders"));
        assertFalse(((List<?>) orders.get("keys")).isEmpty());
        for (Object row : (List<?>) orders.get("rows")) assertTrue(RelationalTest.map(row).containsKey("__node_id"));
      }
    }
  }

  @Test void ordinaryResultsStillMergeRepeatedPartsWithoutGradients() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table context = TabularTest.table(allocator, true, 1, 2, 3);
        Table predict = TabularTest.table(allocator, false, 4, 5);
        TabularTest.Endpoint endpoint = new TabularTest.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      PredictionResult result = client.predict(new Request(context, predict, "y", "regression"), SdmClient.KUMO_TABULAR);
      PredictionResult merged = PredictionResult.merge(List.of(PredictionResult.merge(List.of(result, result)), result));
      assertEquals(6, merged.rows());
      assertTrue(merged.explanations().gradients().entries().isEmpty());
    }
  }

}
