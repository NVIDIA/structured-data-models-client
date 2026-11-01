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
      try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint();
          SdmClient client = endpoint.client()) {
        assertEquals(1, client.predict(request, SdmClient.KUMO_RELATIONAL).rows());
      }
      assertFalse(connection.isClosed());
    }
  }
}
