// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

/** Public option journeys through Arrow, Rust validation and HTTP payloads. */
class DirectOptionsTest {
  @Test void directSchemaAndOptionsAreOwnedAndInferenceTravelsOnPredictAndScore() throws Exception {
    Request request;
    Map<String, Object> nested = new LinkedHashMap<>();
    nested.put("temperature", 0.5);
    nested.put("labels", new ArrayList<>(List.of("a", "b")));
    Map<String, Object> inference = new LinkedHashMap<>();
    inference.put("config", nested);
    RequestOptions options;
    try (RootAllocator allocator = new RootAllocator();
        Table context = TabularTest.table(allocator, true, 1, 2);
        Table scoring = TabularTest.table(allocator, false, 3)) {
      Schema schema = Schema.fromData(Map.of("data", context));
      options = RequestOptions.builder().entityTable("data").schema(schema)
          .inference(inference).taskOptions(Map.of("unused", Map.of("value", 7))).build();
      nested.put("temperature", 9.0);
      request = new Request(context, scoring, "y", "regression", options);
    }
    assertEquals("data", request.entityTable());
    assertEquals(options.taskOptions(), request.taskOptions());
    assertThrows(UnsupportedOperationException.class, () -> request.inference().put("new", 1));
    try (TabularTest.Endpoint endpoint = new TabularTest.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      assertEquals(1, client.predict(request, SdmClient.KUMO_TABULAR).rows());
      assertEquals(options.inference(), endpoint.calls.get(0).get("inference"));
      assertFalse(((Map<?, ?>) endpoint.calls.get(0).get("task")).containsKey("unused"));
      try (FittedModel fitted = client.fit(request, SdmClient.KUMO_TABULAR)) {
        assertEquals(1, fitted.predict(request).rows());
        assertEquals(options.inference(), endpoint.calls.get(endpoint.calls.size() - 1).get("inference"));
      }
    }
  }

  @Test void temporalTaskOptionIsRefusedBeforeNetworkAndFalseIsNotTemporal() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table context = TabularTest.table(allocator, true, 1, 2);
        Table scoring = TabularTest.table(allocator, false, 3);
        TabularTest.Endpoint endpoint = new TabularTest.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      Request anchored = new Request(context, scoring, "y", "regression", RequestOptions.builder()
          .taskOptions(Map.of("anchor_time_column", "date")).build());
      assertThrows(StructuredDataException.class, () -> client.predict(anchored, SdmClient.KUMO_TABULAR));
      assertTrue(endpoint.calls.isEmpty());
      Request plain = new Request(context, scoring, "y", "regression", RequestOptions.builder()
          .taskOptions(Map.of("anchor_time_column", false)).build());
      assertEquals(1, client.predict(plain, SdmClient.KUMO_TABULAR).rows());
    }
  }

  @Test void nativeInferenceChecksRejectLossyIntegersBeforeSending() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table context = TabularTest.table(allocator, true, 1, 2);
        Table scoring = TabularTest.table(allocator, false, 3);
        TabularTest.Endpoint endpoint = new TabularTest.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request = new Request(context, scoring, "y", "regression", RequestOptions.builder()
          .inference(Map.of("seed", new java.math.BigInteger("9007199254740993"))).build());
      assertThrows(StructuredDataException.class, () -> client.predict(request, SdmClient.KUMO_TABULAR));
      assertTrue(endpoint.calls.isEmpty());
    }
  }

  @Test void explicitTaskOverridesPqlInferenceForBothScoringPaths() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table all = TabularTest.table(allocator, true, 1, 2, 3, 4);
        Table scoring = TabularTest.table(allocator, false, 5)) {
      Schema schema = Schema.fromData(Map.of("data", all));
      PredictiveQuery query = PredictiveQuery.parse("PREDICT data.y FOR EACH data.id");
      Request selected = query.materialize(schema, MaterializeOptions.builder().indices(4L)
          .task("classification").numContextSamples(2).build());
      assertEquals("classification", selected.problemType().orElseThrow());
      Request supplied = query.materialize(schema, scoring,
          MaterializeOptions.builder().task("classification").numContextSamples(2).build());
      assertEquals("classification", supplied.problemType().orElseThrow());
      assertEquals(2, supplied.contextRows());
      assertThrows(StructuredDataException.class, () -> query.materialize(schema, scoring,
          MaterializeOptions.builder().task("invalid").build()));
    }
  }

  @Test void frameKeyEditsCheckActualRowsAndPreserveOriginalSchema() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table all = TabularTest.table(allocator, true, 1, 1, 2)) {
      Schema schema = Schema.fromData(Map.of("data", all));
      assertThrows(StructuredDataException.class, () -> schema.withPrimaryKey("data", "id"));
      assertEquals(schema.tables(), schema.snapshot().tables());
    }
  }
}
