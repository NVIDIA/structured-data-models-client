// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.jdbc.ConversionTable;
import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.junit.jupiter.api.Test;

/**
 * T3.28: the core writes a Postgres key set as the JSON text its statement reads and hands it
 * over as a one-cell text attachment the statement's {@code text} parameter names; the JDBC
 * provider binds that cell as the statement's one parameter, as it was handed, and writes no JSON
 * of its own. A plan naming the core's Arrow key set on Postgres, or a text on a driver that
 * binds key sets as relations, is not run.
 */
class T328KeyTextTest {
  /** A key set's text as the core hands it over: one text cell, as an Arrow stream. */
  private static ArrowReader text(String json) throws Exception {
    BufferAllocator allocator = ArrowData.allocator();
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (VarCharVector cell = new VarCharVector("keys", allocator)) {
      cell.allocateNew(1);
      cell.setSafe(0, json.getBytes(StandardCharsets.UTF_8));
      cell.setValueCount(1);
      try (VectorSchemaRoot root = VectorSchemaRoot.of(cell);
          ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bytes)) {
        root.setRowCount(1);
        writer.start();
        writer.writeBatch();
        writer.end();
      }
    }
    return ArrowData.reader(bytes.toByteArray(), allocator);
  }

  private static Map<String, Object> step(String dialect, String text, Map<String, Object> param) {
    Map<String, Object> statement =
        Map.of("dialect", dialect, "text", text, "params", List.of(param));
    Map<String, Object> plan = new java.util.LinkedHashMap<>();
    plan.put("plan", "sql");
    plan.put("statements", List.of(statement));
    if (dialect.equals("postgres")) {
      plan.put("result_conversions", List.of(Map.of("conversion", "zoned_text_utc")));
    }
    return Map.of("plan", plan);
  }

  private static final Map<String, Object> TEXT = Map.of("param", "text", "attachment", 1);

  private static final Map<String, Object> KEY_SET =
      Map.of(
          "param", "key_set", "relation", "TMP", "keys", Map.of("attachment", 1, "columns", List.of()));

  /** The first column's values of an answer, as text. */
  private static List<String> values(ArrowReader read) throws Exception {
    List<String> out = new ArrayList<>();
    try (read) {
      while (read.loadNextBatch()) {
        var column = read.getVectorSchemaRoot().getVector(0);
        for (int row = 0; row < column.getValueCount(); row++) {
          Object value = column.getObject(row);
          out.add(value == null ? null : value.toString());
        }
      }
    }
    return out;
  }

  @Test
  void aPostgresStatementBindsTheKeySetsTextAsItWasHanded() throws Exception {
    try (Connection connection = T312Shop.open("postgres")) {
      JdbcProvider provider =
          new JdbcProvider(connection, JdbcProvider.Driver.POSTGRES, ConversionTable.REVIEWED);
      String sql =
          "SELECT r.value->>0 AS k, (r.value->>1)::bigint AS b"
              + " FROM jsonb_array_elements($1::jsonb) AS r(value) ORDER BY b";
      List<String> read =
          values(
              provider.answer(
                  step("postgres", sql, TEXT),
                  Map.of(1, text("[[\"7\",\"0\"],[\"é\\\"\",\"1\"],[null,\"2\"]]")),
                  ArrowData.allocator()));
      assertEquals(java.util.Arrays.asList("7", "é\"", null), read);
    }
  }

  @Test
  void aPostgresPlanNamingTheCoresArrowKeySetIsNotRun() {
    JdbcProvider provider =
        new JdbcProvider(null, JdbcProvider.Driver.POSTGRES, ConversionTable.REVIEWED);
    assertThrows(
        JdbcProvider.PlanNotRun.class,
        () ->
            provider.answer(
                step("postgres", "SELECT $1::jsonb", KEY_SET),
                Map.of(1, text("[]")),
                ArrowData.allocator()));
  }

  @Test
  void aTextOnADriverBindingKeySetsAsRelationsIsNotRun() {
    for (JdbcProvider.Driver driver :
        List.of(JdbcProvider.Driver.DUCKDB, JdbcProvider.Driver.SQLITE)) {
      JdbcProvider provider = new JdbcProvider(null, driver, ConversionTable.REVIEWED);
      assertThrows(
          JdbcProvider.PlanNotRun.class,
          () ->
              provider.answer(
                  step(driver.dialect(), "SELECT ?", TEXT),
                  Map.of(1, text("[]")),
                  ArrowData.allocator()),
          driver.toString());
    }
  }

  @Test
  void aTextNotHandedOverIsNotRun() {
    JdbcProvider provider =
        new JdbcProvider(null, JdbcProvider.Driver.POSTGRES, ConversionTable.REVIEWED);
    assertThrows(
        JdbcProvider.PlanNotRun.class,
        () ->
            provider.answer(
                step("postgres", "SELECT $1::jsonb", TEXT), Map.of(), ArrowData.allocator()));
  }
}
