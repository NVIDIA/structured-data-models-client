// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.DriverManager;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;

/**
 * The SQL kit from Java: a built-in source's typed request answered by the statements the kit
 * writes, run on the caller's own DuckDB connection through the JDBC provider that runs the core's
 * statements today.
 */
class NativeSqlKitTest {

  private static byte[] utf8(String text) {
    return text.getBytes(StandardCharsets.UTF_8);
  }

  @Test
  @SuppressWarnings("unchecked")
  void aDuckDbScanIsAnsweredByTheKitOnTheCallersConnection() throws Exception {
    try (BufferAllocator allocator = new RootAllocator();
        Connection connection = DriverManager.getConnection("jdbc:duckdb:");
        NativeSqlKit kit = new NativeSqlKit(utf8("{\"provider\":\"sql\",\"dialect\":\"duckdb\"}"))) {
      try (var statement = connection.createStatement()) {
        statement.execute(
            "CREATE TABLE users AS SELECT * FROM (VALUES (1, 25), (2, 40), (3, 35)) t(user_id, age)");
      }
      Map<String, Object> capabilities = (Map<String, Object>) Json.read(kit.capabilities());
      assertTrue(((List<Object>) capabilities.get("optional")).contains("lookup"));
      JdbcProvider driver = new JdbcProvider(connection, JdbcProvider.Driver.DUCKDB);
      List<Object> plans = new ArrayList<>();
      byte[] request =
          utf8(
              "{\"request\":\"scan\",\"table\":{\"source\":[\"users\"]},"
                  + "\"columns\":[\"user_id\",\"age\"],\"filters\":[],\"limit\":null}");
      try (NativeArrow answer =
              kit.answer(
                  request,
                  Map.of(),
                  allocator,
                  (plan, asked, keys) -> {
                    Map<String, Object> step = new LinkedHashMap<>();
                    step.put("plan", Json.read(plan));
                    step.put("request", Json.read(asked));
                    plans.add(step.get("plan"));
                    Map<Integer, ArrowReader> readers = new LinkedHashMap<>();
                    for (var key : keys.entrySet()) {
                      readers.put(key.getKey(), key.getValue().reader(allocator));
                    }
                    return driver.answer(step, readers, allocator);
                  });
          ArrowReader reader = answer.reader(allocator)) {
        long sum = 0;
        int rows = 0;
        while (reader.loadNextBatch()) {
          VectorSchemaRoot root = reader.getVectorSchemaRoot();
          var ages = root.getVector("age");
          for (int at = 0; at < root.getRowCount(); at++) {
            sum += ((Number) ages.getObject(at)).longValue();
            rows++;
          }
        }
        assertEquals(1, plans.size());
        assertEquals(3, rows);
        assertEquals(100, sum);
      }
    }
  }

  @Test
  void aDriverExceptionIsThrownAsItIs() throws Exception {
    try (BufferAllocator allocator = new RootAllocator();
        NativeSqlKit kit = new NativeSqlKit(utf8("{\"provider\":\"sql\",\"dialect\":\"duckdb\"}"))) {
      byte[] request =
          utf8(
              "{\"request\":\"scan\",\"table\":{\"source\":[\"users\"]},"
                  + "\"columns\":[\"user_id\"],\"filters\":[],\"limit\":null}");
      var thrown =
          assertThrows(
              IllegalStateException.class,
              () ->
                  kit.answer(
                      request,
                      Map.of(),
                      allocator,
                      (plan, asked, keys) -> {
                        throw new IllegalStateException("the driver failed");
                      }));
      assertEquals("the driver failed", thrown.getMessage());
    }
  }

  @Test
  void aSourceTheKitWritesNoSqlForIsRefused() {
    assertThrows(
        NativeError.class,
        () -> new NativeSqlKit(utf8("{\"provider\":\"sql\",\"dialect\":\"oracle\"}")).close());
  }
}
