// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.time.LocalDate;
import java.util.List;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

class RequestSplitTest {
  private static long count(ArrowReader reader) throws Exception {
    try (reader) {
      long rows = 0;
      while (reader.loadNextBatch()) rows += reader.getVectorSchemaRoot().getRowCount();
      return rows;
    }
  }

  @Test
  void directViewOwnsReadersAfterOriginalInputCloses() throws Exception {
    Request request;
    try (RootAllocator allocator = new RootAllocator();
        Table context = TabularTest.table(allocator, true, 1, 2);
        Table predict = TabularTest.table(allocator, false, 3)) {
      request = new Request(context, predict, "y", "regression");
    }
    try (RootAllocator allocator = new RootAllocator()) {
      assertEquals(2, count(request.context().read(allocator)));
      assertEquals(1, count(request.predict().read(allocator)));
      assertEquals(List.of(), request.context().relatedTables());
      assertThrows(
          StructuredDataException.class, () -> request.context().readRelated("missing", allocator));
    }
    try (TabularTest.Endpoint endpoint = new TabularTest.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      assertEquals(1, client.predict(request, SdmClient.KUMO_TABULAR).rows());
    }
  }

  @Test
  void gatheredViewsPartsAndRestorationLeavePredictionPayloadsUntouched() throws Exception {
    Schema schema = RelationalTest.fixture(true, true);
    for (String query :
        List.of(RelationalTest.SCALAR, RelationalTest.FORECAST, RelationalTest.RANK)) {
      var options = RelationalTest.options(1L, 3L);
      if (!query.equals(RelationalTest.SCALAR)) options.anchorTime(LocalDate.of(2024, 5, 1));
      if (query.equals(RelationalTest.RANK)) options.candidates(List.of(1L, 2L, 3L));
      Request request = PredictiveQuery.parse(query).materialize(schema, options.build());
      List<String> payloads = request.payloads();
      try (RootAllocator allocator = new RootAllocator()) {
        assertEquals(request.contextRows(), count(request.context().read(allocator)));
        assertEquals(request.predictRows(), count(request.predict().read(allocator)));
        for (String name : request.predict().relatedTables()) {
          // Independent readers survive their finished core run and close in either order.
          try (ArrowReader first = request.predict().readRelated(name, allocator);
              ArrowReader second = request.predict().readRelated(name, allocator)) {
            assertEquals(
                first.getVectorSchemaRoot().getSchema(), second.getVectorSchemaRoot().getSchema());
            assertEquals(first.loadNextBatch(), second.loadNextBatch());
          }
        }
        if (request.parts().isPresent())
          for (Request part : request.parts().get()) {
            assertEquals(part.predictRows(), count(part.predict().read(allocator)));
          }
        assertThrows(
            StructuredDataException.class,
            () -> request.predict().readRelated("missing", allocator));
      }
      assertEquals(payloads, request.payloads());
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      request.writeTo(bytes);
      Request restored = Request.readFrom(new ByteArrayInputStream(bytes.toByteArray()));
      try (RootAllocator allocator = new RootAllocator()) {
        assertEquals(restored.predictRows(), count(restored.predict().read(allocator)));
        assertEquals(request.context().relatedTables(), restored.context().relatedTables());
      }
      try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint();
          SdmClient client = endpoint.client()) {
        assertEquals(
            client.predict(request, SdmClient.KUMO_RELATIONAL).rows(),
            client.predict(restored, SdmClient.KUMO_RELATIONAL).rows());
      }
    }
  }
}
