// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;

class ParitySupplementTest {
  @Test
  void tabularHoldoutUsesActualContextAndRecordsTransportResponse() throws Exception {
    try (var allocator = new RootAllocator();
        var context = TabularTest.table(allocator, true, 1, 2, 3);
        var scoring = TabularTest.table(allocator, false, 4);
        var endpoint = new TabularTest.Endpoint();
        var client = SdmClient.connect(endpoint.url())) {
      exercise(
          new Request(context, scoring, "y", "regression"),
          client,
          SdmClient.KUMO_TABULAR,
          allocator);
      assertEquals(List.of(0), endpoint.deleteBodySizes);
    }
  }

  @Test
  void relationalHoldoutChecksAllRelatedRowsAndSemanticItems() throws Exception {
    var request =
        PredictiveQuery.parse(RelationalTest.SCALAR)
            .materialize(
                RelationalTest.fixture(true, true), WarehouseSessionTest.asked(1L, 2L).build());
    try (var allocator = new RootAllocator();
        var endpoint = new RelationalTest.Endpoint();
        var client = endpoint.client()) {
      exercise(request, client, SdmClient.KUMO_RELATIONAL, allocator);
    }
  }

  @Test
  void checkpointsNeverClaimSuccessBeforeRouteFinishes() throws Exception {
    var output = java.nio.file.Files.createTempFile("parity-checkpoint-test-", ".jsonl");
    String previous = System.getProperty("sdm.parity.output");
    System.setProperty("sdm.parity.output", output.toString());
    try {
      var record = new LinkedHashMap<String, Object>();
      record.put("status", "passed");
      record.put("route", Map.of("scope", "whole-journey"));
      ParityRunner.checkpoint(List.of(record), false);
      assertTrue(java.nio.file.Files.readString(output).contains("route-observation-pending"));
      assertEquals("passed", record.get("status"));
      ParityRunner.checkpoint(List.of(record), true);
      assertFalse(java.nio.file.Files.readString(output).contains("route-observation-pending"));
    } finally {
      if (previous == null) System.clearProperty("sdm.parity.output");
      else System.setProperty("sdm.parity.output", previous);
      java.nio.file.Files.deleteIfExists(output);
    }
  }

  static void exercise(Request request, SdmClient client, String model, RootAllocator allocator)
      throws Exception {
    var calls = new ArrayList<Map<String, Object>>();
    var record = new LinkedHashMap<String, Object>();
    record.put("stage", "fit");
    record.put("status", "incomplete");
    String before = ParityRunner.digest(ParityRunner.requestEvidence(request, allocator));
    try (var route = new ParityRoute()) {
      route.begin(record);
      route.capture(calls);
      var held = List.of(ParityHoldout.key(request, allocator));
      try (var fitted = client.fit(request, model, held, List.of("prediction"))) {
        for (int at = 0; at < 2; at++) {
          var result = fitted.predict(request);
          var frame = ParityResults.frame(result);
          assertEquals(List.of("row_index", "id", "prediction"), frame.get("columns"));
          var details = ParityResults.details(result);
          assertEquals(11, ParityRunner.map(ParityRunner.list(details.get("items")).get(0)).size());
          assertEquals(result.requestId(), calls.get(calls.size() - 1).get("response_id"));
        }
      }
      ParityHoldout.check(request, held, calls);
      assertNull(calls.get(3).get("payload"), "DELETE sends no native sentinel body");
      Object sessionId = calls.get(0).get("session_id");
      assertInstanceOf(String.class, sessionId);
      assertFalse(sessionId.toString().isEmpty());
      assertTrue(calls.stream().allMatch(call -> sessionId.equals(call.get("session_id"))));
      assertEquals(before, ParityRunner.digest(ParityRunner.requestEvidence(request, allocator)));
      route.end(record);
    }
  }
}
