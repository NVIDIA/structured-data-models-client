// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;

class FitOutputsTest {
  @Test void tabularFitForwardsRequestedOutputs() throws Exception {
    try (var allocator = new RootAllocator();
         var context = TabularTest.table(allocator, true, 1, 2);
         var scoring = TabularTest.table(allocator, false, 3);
         var endpoint = new TabularTest.Endpoint();
         var client = SdmClient.connect(endpoint.url())) {
      var request = new Request(context, scoring, "y", "regression");
      try (var fitted = client.fit(request, SdmClient.KUMO_TABULAR, List.of(), List.of("prediction"))) {
        var result = fitted.predict(request);
        assertEquals(List.of("prediction"), ParityRunner.map(endpoint.calls.get(1).get("output")).get("fields"));
        assertEquals(List.of("prediction"), result.outputs());
        assertEquals(1, result.rows());
      }
    }
  }
  @Test void relationalFitForwardsRequestedOutputsAndTraceObservesActualResponses() throws Exception {
    var request = PredictiveQuery.parse(RelationalTest.SCALAR).materialize(
        RelationalTest.fixture(true, true), WarehouseSessionTest.asked(1L, 2L).build());
    var responses = new java.util.ArrayList<byte[]>();
    com.nvidia.sdm.internal.Trace.install(new com.nvidia.sdm.internal.Trace() {
      @Override public void modelAnswered(byte[] response) { responses.add(response); }
    });
    try (var endpoint = new RelationalTest.Endpoint(); var client = endpoint.client();
         var fitted = client.fit(request, SdmClient.KUMO_RELATIONAL, List.of(), List.of("prediction"))) {
      var result = fitted.predict(request);
      assertEquals(List.of("prediction"), ParityRunner.map(endpoint.scored.get(0).get("output")).get("fields"));
      assertEquals(List.of("prediction"), result.outputs());
      assertEquals(2, result.rows());
      var raw = ParityRunner.map(com.nvidia.sdm.internal.Json.read(responses.get(1)));
      assertEquals(result.requestId(), raw.get("id"));
    } finally { com.nvidia.sdm.internal.Trace.install(null); }
  }
}
