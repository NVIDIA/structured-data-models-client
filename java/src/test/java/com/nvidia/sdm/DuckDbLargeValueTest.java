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
      // Same shop, query and indices as contract/support/duckdb_value_scenarios.py.
      // The previous three-row numeric toy target inferred classification and the live
      // model rejected that task independently of the provider's large-value support.
      statement.execute("CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20+i)::INTEGER AS age, "
          + "['US','DE','IN'][1+i%3] AS country FROM range(8) t(i)");
      statement.execute("CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i%8)::BIGINT AS user_id, "
          + "(10+i*1.5)::DOUBLE AS price, TIMESTAMP '2023-01-01'+to_days((i*7)::INTEGER) AS ts FROM range(80) t(i)");
      statement.execute("ALTER TABLE users ADD COLUMN f BLOB");
      statement.execute("UPDATE users SET f='a'");
      statement.execute("UPDATE users SET f=repeat('a',2147483664)::BLOB WHERE user_id=3");
      Schema schema = Schema.fromSource(DuckDbSource.of(connection));
      Request request = PredictiveQuery.parse("PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id")
          .materialize(schema, MaterializeOptions.builder().indices(0L,1L,2L,3L,4L,5L,6L,7L)
              .anchorTime(java.time.LocalDate.of(2024,3,1)).build());
      assertTrue(schema.table("users").columns().stream().noneMatch(c -> c.name().equals("f")));
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
    assertEquals(8, result.rows());
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
      assertEquals(8, rows);
    }
  }
}
