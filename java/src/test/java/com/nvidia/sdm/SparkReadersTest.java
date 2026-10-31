// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import com.nvidia.sdm.internal.Providers;
import com.nvidia.sdm.internal.ReadTables;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.table.Table;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;

class SparkReadersTest {
  @Test void callerSessionTypedRowsAndBoundNativeSqlStayIntact() throws Exception {
    try (SparkSession session = SparkSession.builder().master("local[1]").appName("sdm-reader-test")
        .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.host", "127.0.0.1").config("spark.sql.session.timeZone", "America/Los_Angeles")
        .getOrCreate(); RootAllocator allocator = new RootAllocator()) {
      session.sparkContext().setLogLevel("ERROR");
      session.sql("SELECT CAST(9007199254740993 AS BIGINT) AS id, CAST(12.123456789 AS DECIMAL(20,9)) AS amount, "
          + "timestamp_ntz'2024-01-01 01:02:03.123456' AS local_time, array(1, NULL, 3) AS items, "
          + "named_struct('x', 7, 'text', 'value') AS detail, map('key', 9) AS mapping")
          .createOrReplaceTempView("reader_private");
      Source source = DatabricksSparkSource.of(session);
      assertSame(session, ((DatabricksSparkSource) source).session());
      try (Table rows = Readers.readTable(source, "reader_private", allocator)) {
        assertEquals(9007199254740993L, ReadersTest.value(rows, 0, 0));
        assertEquals(new java.math.BigDecimal("12.123456789"), ReadersTest.value(rows, 1, 0));
        assertEquals(java.time.LocalDateTime.parse("2024-01-01T01:02:03.123456"), ReadersTest.value(rows, 2, 0));
        assertEquals(3, ((List<?>) ReadersTest.value(rows, 3, 0)).size());
        assertEquals(7, ((Map<?, ?>) ReadersTest.value(rows, 4, 0)).get("x"));
      }
      var chosen = Providers.of(source);
      assertEquals("databricks", chosen.choice().get("dialect"));
      var plan = Map.<String, Object>of("plan", Map.of("plan", "sql", "statements", List.of(
          Map.of("dialect", "databricks", "text", "SELECT id FROM reader_private WHERE id = ?", "params", List.of(
              Map.of("param", "value", "value", Map.of("t", "i64", "v", "9007199254740993")))))));
      try (Table rows = ReadTables.collect(chosen.provider().answer(plan, Map.of(), allocator), allocator)) {
        assertEquals(1, rows.getRowCount());
        assertEquals(9007199254740993L, ReadersTest.value(rows, 0, 0));
      }
      session.range(100_003).createOrReplaceTempView("reader_batches");
      try (Table rows = Readers.readTable(source, "reader_batches", allocator)) {
        assertEquals(100_003, rows.getRowCount());
        assertEquals(100_002L, ReadersTest.value(rows, 0, 100_002));
      }
      assertFalse(session.sparkContext().isStopped());
    }
  }
}
