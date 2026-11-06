// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import java.util.stream.LongStream;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

/** The size limits a client holds requests to: their defaults and a raised one taking effect. */
class LimitsTest {
  @Test void theDefaultsAreTheServingLayers() {
    assertEquals(
        Map.of(
            "max_relational_context_rows", 10_000L,
            "max_relational_predict_rows", 10_000L,
            "max_temporal_link_predict_rows", 200L,
            "max_tabular_context_rows", 10_000L,
            "max_request_bytes", 31_457_280L),
        Limits.defaults().resolved());
  }

  @Test void aSettingGivenComesBeforeTheDefaultAndMustBeAtLeastOne() {
    Limits limits = Limits.builder().maxTabularContextRows(200_000).build();
    assertEquals(Map.of("max_tabular_context_rows", 200_000L), limits.values());
    assertEquals(200_000L, limits.resolved().get("max_tabular_context_rows"));
    assertEquals(10_000L, limits.resolved().get("max_relational_context_rows"));
    StructuredDataException refused = assertThrows(
        StructuredDataException.class, () -> Limits.builder().maxRequestBytes(0));
    assertEquals("INVALID_CONFIGURATION", refused.code());
  }

  @Test void aTabularContextPastTheDefaultIsSentWhenTheLimitIsRaised() throws Exception {
    long[] ids = LongStream.range(0, 10_001).toArray();
    try (RootAllocator allocator = new RootAllocator();
        Table context = TabularTest.table(allocator, true, ids);
        Table scoring = TabularTest.table(allocator, false, 20_000);
        TabularTest.Endpoint endpoint = new TabularTest.Endpoint()) {
      Request request = new Request(context, scoring, "y", "regression");
      try (SdmClient client = SdmClient.connect(endpoint.url())) {
        StructuredDataException refused = assertThrows(StructuredDataException.class,
            () -> client.predict(request, SdmClient.KUMO_TABULAR));
        assertTrue(refused.getMessage().contains("SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS"),
            refused.getMessage());
        assertTrue(endpoint.calls.isEmpty());
      }
      Limits raised = Limits.builder().maxTabularContextRows(20_000).build();
      try (SdmClient client = SdmClient.connect(endpoint.url(),
          ClientOptions.builder().limits(raised).build())) {
        assertEquals(raised, client.limits());
        assertEquals(1, client.predict(request, SdmClient.KUMO_TABULAR).rows());
        @SuppressWarnings("unchecked") Map<String, Object> sent =
            (Map<String, Object>) endpoint.calls.get(0).get("context");
        @SuppressWarnings("unchecked") Map<String, Object> table =
            (Map<String, Object>) sent.get("instance_table");
        assertEquals(10_001, ((List<?>) table.get("rows")).size());
      }
    }
  }
}
