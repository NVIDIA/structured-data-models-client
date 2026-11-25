// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.sql.Connection;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

class ResultItemsTest {
  @Test
  void directItemsAreSemanticImmutableAndRawReaderStaysCompatible() throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Table context = TabularTest.table(allocator, true, 1, 2);
        Table scoring = TabularTest.table(allocator, false, 3, 4);
        TabularTest.Endpoint endpoint = new TabularTest.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      PredictionResult result =
          client.predict(new Request(context, scoring, "y", "regression"), SdmClient.KUMO_TABULAR);
      List<PredictionItem> items = result.items();
      assertEquals(2, items.size());
      assertTrue(items.get(0).prediction() instanceof Number);
      assertFalse(items.get(0).fields().containsKey("answer_index"));
      assertThrows(UnsupportedOperationException.class, () -> items.add(items.get(0)));
      assertThrows(
          UnsupportedOperationException.class, () -> items.get(0).fields().put("prediction", 0));
      try (var raw = result.read(allocator)) {
        assertTrue(raw.loadNextBatch());
        assertNotNull(raw.getVectorSchemaRoot().getVector("answer_index"));
      }
      assertEquals(items.get(0).prediction(), result.items().get(0).prediction());
    }
  }

  @Test
  void rankingOwnersSurviveDifferentCandidateListsAndRepeatedMixedTaskMerges() throws Exception {
    Request first, second, scalar;
    try (Connection connection = WarehouseSessionTest.open("duckdb")) {
      Schema schema = ModelConnectorCoverageTest.linkSchema("duckdb", connection);
      first =
          PredictiveQuery.parse(RelationalTest.RANK)
              .materialize(
                  schema,
                  WarehouseSessionTest.asked(1L, 2L).candidates(List.of(1L, 2L, 999L)).build());
      second =
          PredictiveQuery.parse(RelationalTest.RANK)
              .materialize(
                  schema,
                  WarehouseSessionTest.asked(1L, 2L).candidates(List.of(2L, 3L, 999L)).build());
      scalar =
          PredictiveQuery.parse(RelationalTest.SCALAR)
              .materialize(schema, WarehouseSessionTest.asked(1L, 2L).build());
    }
    try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint();
        SdmClient client = endpoint.client()) {
      PredictionResult a = client.predict(first, SdmClient.KUMO_RELATIONAL);
      PredictionResult b = client.predict(second, SdmClient.KUMO_RELATIONAL);
      assertEquals(List.of(1L, 2L), a.items().get(0).prediction());
      assertEquals(Map.of("candidate", 1L, "score", .9), a.items().get(0).rankings().get(0));
      assertThrows(
          UnsupportedOperationException.class,
          () -> a.items().get(0).rankings().get(0).put("score", 0));
      assertEquals(List.of(999L), a.unscoredCandidates());
      PredictionResult joined = PredictionResult.merge(List.of(a, b));
      assertEquals(List.of(1L, 2L), joined.items().get(0).prediction());
      assertEquals(List.of(2L, 3L), joined.items().get(2).prediction());
      PredictionResult twice = PredictionResult.merge(List.of(b, joined));
      assertEquals(List.of(2L, 3L), twice.items().get(0).prediction());
      assertEquals(List.of(1L, 2L), twice.items().get(2).prediction());
      assertEquals(List.of(2L, 3L), twice.items().get(4).prediction());

      PredictionResult rankedOnly =
          client.predict(first, SdmClient.KUMO_RELATIONAL, List.of("prediction"));
      PredictionResult ordinary =
          client.predict(scalar, SdmClient.KUMO_RELATIONAL, List.of("prediction"));
      assertNull(rankedOnly.items().get(0).rankings());
      PredictionResult mixed = PredictionResult.merge(List.of(ordinary, rankedOnly, ordinary));
      assertTrue(mixed.items().get(0).prediction() instanceof Number);
      assertEquals(List.of(1L, 2L), mixed.items().get(2).prediction());
      assertTrue(mixed.items().get(4).prediction() instanceof Number);
    }
  }

  @Test
  void temporalLinkDictionariesAndForecastStepsNeedNoJsonParsingByCaller() throws Exception {
    Schema schema = RelationalTest.fixture(true, true);
    try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint();
        SdmClient client = endpoint.client()) {
      StructuredDataException refused =
          assertThrows(
              StructuredDataException.class,
              () ->
                  PredictiveQuery.parse(RelationalTest.LINK)
                      .materialize(schema, RelationalTest.options(1L, 2L).build()));
      assertEquals("UNSUPPORTED_REQUEST", refused.code());
      Request forecast =
          PredictiveQuery.parse(RelationalTest.FORECAST)
              .materialize(schema, RelationalTest.options(1L, 2L).build());
      List<PredictionItem> predicted = client.predict(forecast, SdmClient.KUMO_RELATIONAL).items();
      assertEquals(4, predicted.size());
      assertEquals(
          List.of(1L, 1L, 2L, 2L), predicted.stream().map(PredictionItem::forecastStep).toList());
    }
  }
}
