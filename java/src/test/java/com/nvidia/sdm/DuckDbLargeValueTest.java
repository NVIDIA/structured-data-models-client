// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.sql.DriverManager;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Explicit resource-heavy public journey: a single provider value exceeds Java array limits. */
class DuckDbLargeValueTest {
  @Test
  @EnabledIfSystemProperty(named = "sdm.test.giant", matches = "true")
  void describeOmitAndPredictWithBlobLargerThanTwoGiB() throws Exception {
    try (var connection = DriverManager.getConnection("jdbc:duckdb:");
        var statement = connection.createStatement()) {
      statement.execute("SET threads=2");
      statement.execute("SET memory_limit='10GB'");
      statement.execute("SET arrow_large_buffer_size=true");
      statement.execute("CREATE TABLE users AS SELECT 1::BIGINT id, 1.0::DOUBLE y, "
          + "repeat('a', 2147483664)::BLOB payload");
      statement.execute("INSERT INTO users VALUES (2, 2.0, NULL), (3, 3.0, NULL)");
      Schema schema = Schema.fromSource(DuckDbSource.of(connection))
          .withPrimaryKey("users", "id");
      Request request = PredictiveQuery.parse("PREDICT users.y FOR EACH users.id")
          .materialize(schema, MaterializeOptions.builder().indices(1L).build());
      assertTrue(schema.table("users").columns().stream().noneMatch(c -> c.name().equals("payload")));
      String serving = System.getProperty("sdm.test.giant.endpoint");
      if (serving != null) {
        assertFalse(serving.isBlank(), "sdm.test.giant.endpoint must name a serving endpoint");
        try (SdmClient client = SdmClient.forDatabricksServing(
            DatabricksServing.of(new com.databricks.sdk.WorkspaceClient(), serving))) {
          assertPrediction(client, request);
        }
      } else {
        try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint();
            SdmClient client = endpoint.client()) {
          assertPrediction(client, request);
        }
      }
      assertFalse(connection.isClosed());
    }
  }

  private static void assertPrediction(SdmClient client, Request request) throws Exception {
    PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL);
    assertEquals(1, result.rows());
    try (var allocator = new org.apache.arrow.memory.RootAllocator();
        var reader = result.read(allocator)) {
      long rows = 0;
      while (reader.loadNextBatch()) {
        var root = reader.getVectorSchemaRoot();
        var prediction = root.getVector("prediction");
        assertNotNull(prediction);
        for (int at = 0; at < root.getRowCount(); at++) {
          assertInstanceOf(Number.class, prediction.getObject(at));
          assertTrue(Double.isFinite(((Number) prediction.getObject(at)).doubleValue()));
        }
        rows += root.getRowCount();
      }
      assertEquals(1, rows);
    }
  }
}
