// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.DatabricksSource;
import com.nvidia.sdm.FromSourceOptions;
import com.nvidia.sdm.Schema;
import com.nvidia.sdm.StructuredDataException;
import com.nvidia.sdm.internal.ArrowData;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.sql.SQLException;
import java.sql.SQLTransientException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.junit.jupiter.api.Test;

/**
 * Java's Databricks provider (T3.13): the core's statements run in order on the caller's
 * connection, through the Databricks JDBC driver (a stand-in here, fed answers as the driver
 * gives them), each statement's values bound to its {@code ?} markers in order (a text scalar,
 * and a key set's JSON text the core wrote, as it was handed), as Python's Databricks provider
 * binds them; the last statement's rows read by the Databricks column of the conversion table;
 * and a failure told by the facts it carries (its SQLSTATE and Databricks error class), never its
 * text, as the Python provider tells them.
 */
class T313DatabricksProviderTest {
  private static FromSourceOptions users() {
    return FromSourceOptions.builder().table("users").primaryKey("users", "user_id")
        .noTimeColumn("users").relationships(List.of()).checkKeys(false).build();
  }

  private static FakeJdbc.Database nestedDatabase(String kind, Object... values) {
    return new FakeJdbc.Database(sql -> {
      if (sql.contains("information_schema.columns")) {
        return List.of(
            FakeJdbc.Column.of("column_name", "STRING", "user_id", "score", "f"),
            FakeJdbc.Column.of("full_data_type", "STRING", "BIGINT", "DOUBLE", kind),
            FakeJdbc.Column.of("is_nullable", "STRING", "NO", "YES", "YES"),
            FakeJdbc.Column.of("table_catalog", "STRING", "catalog", "catalog", "catalog"),
            FakeJdbc.Column.of("table_schema", "STRING", "schema", "schema", "schema"),
            FakeJdbc.Column.of("table_name", "STRING", "users", "users", "users"),
            FakeJdbc.Column.of("table_type", "STRING", "MANAGED", "MANAGED", "MANAGED"));
      }
      if (sql.startsWith("DESCRIBE QUERY ")) {
        return List.of(FakeJdbc.Column.of("col_name", "STRING", "user_id", "score", "f"),
            FakeJdbc.Column.of("data_type", "STRING", "BIGINT", "DOUBLE", kind));
      }
      if (sql.contains("information_schema.referential_constraints")) {
        return List.of(FakeJdbc.Column.of("constraint_name", "STRING"),
            FakeJdbc.Column.of("source_column", "STRING"),
            FakeJdbc.Column.of("table_catalog", "STRING"),
            FakeJdbc.Column.of("table_schema", "STRING"),
            FakeJdbc.Column.of("table_name", "STRING"),
            FakeJdbc.Column.of("referenced_column", "STRING"));
      }
      if (sql.contains("information_schema.table_constraints")) {
        return List.of(FakeJdbc.Column.of("constraint_type", "STRING"),
            FakeJdbc.Column.of("constraint_name", "STRING"),
            FakeJdbc.Column.of("column_name", "STRING"));
      }
      if (sql.contains("FROM `catalog`.`schema`.`users`")) {
        return List.of(FakeJdbc.Column.of("user_id", "BIGINT", 0L, 1L, 2L, 3L),
            FakeJdbc.Column.of("score", "DOUBLE", 0.0, 1.0, 2.0, 3.0),
            FakeJdbc.Column.of("f", kind, values));
      }
      throw new AssertionError("unexpected SQL: " + sql);
    });
  }

  @Test
  void publicCatalogTypingKeepsNestedRefusalsAndListOmission() {
    for (String kind : List.of("STRUCT<age: INT, country: STRING>", "MAP<STRING, INT>",
        "ARRAY<TIMESTAMP>")) {
      FakeJdbc.Database db = nestedDatabase(kind, "value", "value", null, "value");
      StructuredDataException error = assertThrows(StructuredDataException.class,
          () -> Schema.fromSource(DatabricksSource.of(db.connection(), "catalog", "schema"), users()));
      assertEquals("UNSUPPORTED_REQUEST", error.code(), kind);
      assertEquals("column_type", error.details().get("rule"), kind);
      assertEquals(0, db.statementsOpen, kind);
    }
    for (String kind : List.of("ARRAY<DECIMAL(8,2)>", "ARRAY<INT>", "STRING")) {
      Object[] values = kind.equals("ARRAY<INT>")
          ? new Object[] {List.of(1, 2), List.of(), null, List.of(3)}
          : new Object[] {"[1.25,2.50]", "[]", null, "[3.75]"};
      FakeJdbc.Database db = nestedDatabase(kind, values);
      Schema schema = Schema.fromSource(DatabricksSource.of(db.connection(), "catalog", "schema"), users());
      assertEquals(kind.equals("STRING") ? List.of("user_id", "score", "f")
          : List.of("user_id", "score"),
          schema.table("users").columns().stream().map(c -> c.name()).toList(), kind);
      assertEquals(0, db.statementsOpen, kind);
    }
  }

  @Test
  void publicOversizedDriverFailuresKeepFactsAndTheNextCallUsable() {
    for (String message : List.of("secret_short", "secret_" + "x".repeat(100_000), "secret_next")) {
      FakeJdbc.Database db = new FakeJdbc.Database(
          sql -> new SQLTransientException("[UNRESOLVED_COLUMN.WITH_SUGGESTION] " + message, "42703", 1003));
      StructuredDataException error = assertThrows(StructuredDataException.class,
          () -> Schema.fromSource(DatabricksSource.of(db.connection(), "catalog", "schema"), users()));
      assertEquals("PROVIDER_ERROR", error.code());
      assertTrue(error.retryable());
      assertEquals("42703", error.details().get("sqlstate"));
      assertEquals("1003", error.details().get("vendor_code"));
      assertEquals(SQLTransientException.class.getName(), error.details().get("driver_class"));
      assertFalse(error.getMessage().contains("secret_"));
      assertFalse(error.details().toString().contains("secret_"));
      assertEquals(0, db.statementsOpen);
    }
  }

  /** A key set's text as the core hands it over: one text cell, as an Arrow stream. */
  static ArrowReader text(String json) throws Exception {
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

  static Map<String, Object> statement(String text, Object... params) {
    return Map.of("dialect", "databricks", "text", text, "params", List.of(params));
  }

  static Map<String, Object> plan(Object... statements) {
    return Map.of("plan", Map.of("plan", "sql", "statements", List.of(statements)));
  }

  static final Map<String, Object> TEXT_7 = Map.of("param", "text", "attachment", 7);

  static Map<String, Object> scalar(String tag, Object value) {
    return Map.of("param", "value", "value", Map.of("t", tag, "v", value));
  }

  /** Every column of an answer, read whole, by name. */
  static Map<String, List<Object>> read(ArrowReader answer) throws Exception {
    Map<String, List<Object>> out = new LinkedHashMap<>();
    try (answer) {
      VectorSchemaRoot root = answer.getVectorSchemaRoot();
      root.getSchema().getFields().forEach(f -> out.put(f.getName(), new ArrayList<>()));
      while (answer.loadNextBatch()) {
        for (var vector : root.getFieldVectors()) {
          for (int row = 0; row < root.getRowCount(); row++) {
            out.get(vector.getName()).add(vector.getObject(row));
          }
        }
      }
    }
    return out;
  }

  @Test
  void theStatementsRunInOrderEachValueBoundToItsMarkersInOrder() throws Exception {
    FakeJdbc.Database database =
        new FakeJdbc.Database(
            text ->
                List.of(FakeJdbc.Column.of("user_id", "BIGINT", 9007199254740993L, null, 3L)));
    JdbcProvider provider =
        new JdbcProvider(database.connection(), JdbcProvider.Driver.DATABRICKS);
    Map<Integer, ArrowReader> keys = new HashMap<>();
    keys.put(7, text("[3,\"é\",null]"));
    ArrowReader answer =
        provider.answer(
            plan(
                statement("SET TIME ZONE 'UTC'"),
                statement(
                    "SELECT user_id FROM users WHERE s = ? AND k IN (SELECT col FROM"
                        + " explode(from_json(?, 'array<bigint>'))) AND t = ?",
                    scalar("utf8", "sales"),
                    TEXT_7,
                    scalar("utf8", "x"))),
            keys,
            ArrowData.allocator());
    Map<String, List<Object>> got = read(answer);
    List<Object> ids = new ArrayList<>();
    ids.add(9007199254740993L);
    ids.add(null);
    ids.add(3L);
    assertEquals(Map.of("user_id", ids), got);
    assertEquals(2, database.ran.size());
    assertEquals(new FakeJdbc.Ran("SET TIME ZONE 'UTC'", List.of()), database.ran.get(0));
    assertEquals(List.of("sales", "[3,\"é\",null]", "x"), database.ran.get(1).bound());
    assertEquals(0, database.statementsOpen, "every statement is closed");
  }

  @Test
  void aKeySetReadTwiceIsReadOnceAndAnEmptyAnswerKeepsItsSchema() throws Exception {
    FakeJdbc.Database database =
        new FakeJdbc.Database(
            text -> List.of(FakeJdbc.Column.of("n", "INT"), FakeJdbc.Column.of("s", "STRING")));
    JdbcProvider provider =
        new JdbcProvider(database.connection(), JdbcProvider.Driver.DATABRICKS);
    Map<Integer, ArrowReader> keys = new HashMap<>();
    keys.put(7, text("[1]"));
    ArrowReader answer =
        provider.answer(
            plan(statement("SELECT 1 WHERE ? IS NOT NULL", TEXT_7), statement("SELECT ?", TEXT_7)),
            keys,
            ArrowData.allocator());
    try (answer) {
      VectorSchemaRoot root = answer.getVectorSchemaRoot();
      assertEquals("Int(32, true)", root.getSchema().findField("n").getType().toString());
      assertEquals("Utf8", root.getSchema().findField("s").getType().toString());
      int rows = 0;
      while (answer.loadNextBatch()) {
        rows += root.getRowCount();
      }
      assertEquals(0, rows);
    }
    assertEquals(List.of("[1]"), database.ran.get(0).bound());
    assertEquals(List.of("[1]"), database.ran.get(1).bound());
  }

  @Test
  void aPlanTheProviderCannotRunIsRefusedBeforeAnythingRuns() throws Exception {
    FakeJdbc.Database database = new FakeJdbc.Database(text -> List.of());
    JdbcProvider provider =
        new JdbcProvider(database.connection(), JdbcProvider.Driver.DATABRICKS);
    Map<String, Object> keySet =
        Map.of(
            "param",
            "key_set",
            "relation",
            "TMP",
            "keys",
            Map.of("attachment", 7, "columns", List.of()));
    List<Map<String, Object>> refused =
        List.of(
            // Another dialect's statement.
            Map.of(
                "plan",
                Map.of(
                    "plan",
                    "sql",
                    "statements",
                    List.of(Map.of("dialect", "postgres", "text", "SELECT 1", "params", List.of())))),
            // The core's Arrow key set, which Databricks binds only as its text.
            plan(statement("SELECT 1", keySet)),
            // A text not handed over.
            plan(statement("SELECT ?", Map.of("param", "text", "attachment", 9))),
            // A scalar no driver value holds here.
            plan(statement("SELECT ?", scalar("duration", Map.of("unit", "us", "v", 1)))));
    for (Map<String, Object> request : refused) {
      Map<Integer, ArrowReader> keys = new HashMap<>();
      keys.put(7, text("[1]"));
      assertThrows(
          JdbcProvider.PlanNotRun.class,
          () -> provider.answer(request, keys, ArrowData.allocator()),
          request.toString());
    }
    assertEquals(List.of(), database.ran);
  }

  @Test
  void aDatabricksFailureCarriesItsDiagnosticOnlyAsAnInternalFact() {
    JdbcProvider databricks = new JdbcProvider(null, JdbcProvider.Driver.DATABRICKS);
    // The class a Databricks message opens with, and the state the driver gives.
    SQLException opened =
        new SQLException(
            "[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column `secret_value` cannot be resolved."
                + " SQLSTATE: 42703\n== SQL (line 1, position 8) ==\nSELECT [NOT_IT] FROM t",
            "42703",
            0);
    assertEquals(
        Map.of("sqlstate", "42703", "error_class", "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            "driver_message", opened.getMessage()),
        databricks.failure(opened));
    // The driver's own wrapping, its state left out: the state and class the message carries,
    // the driver's mixed-case brackets not a class.
    SQLException wrapped =
        new SQLException(
            "[Databricks][JDBCDriver](500051) ERROR processing query/statement. Error Code: 0,"
                + " Error message: [TABLE_OR_VIEW_NOT_FOUND] The table `secret` cannot be found."
                + " SQLSTATE: 42P01",
            null,
            500051);
    assertEquals(
        Map.of(
            "sqlstate", "42P01", "vendor_code", "500051", "error_class", "TABLE_OR_VIEW_NOT_FOUND",
            "driver_message", wrapped.getMessage()),
        databricks.failure(wrapped));
    // A message that says neither.
    assertEquals(Map.of("driver_message", "connection reset"),
        databricks.failure(new SQLException("connection reset")));
    assertEquals(Map.of(), databricks.failure(new SQLException((String) null)));
    // A transient failure keeps its state.
    assertEquals(
        Map.of("sqlstate", "08001", "driver_message", "no route"),
        databricks.failure(new SQLTransientException("no route", "08001")));
    // Anything else the provider raised carries no fact.
    assertEquals(Map.of(), databricks.failure(new IllegalStateException("[NOT_A_CLASS] x")));
    // Another driver's failure is read as before: its state and code, no class.
    JdbcProvider postgres = new JdbcProvider(null, JdbcProvider.Driver.POSTGRES);
    assertEquals(
        Map.of("sqlstate", "42P01", "driver_message", "[TABLE_OR_VIEW_NOT_FOUND] x"),
        postgres.failure(new SQLException("[TABLE_OR_VIEW_NOT_FOUND] x", "42P01", 0)));
    for (Map<String, Object> facts :
        List.of(databricks.failure(opened), databricks.failure(wrapped))) {
      assertTrue(facts.entrySet().stream().filter(e -> !e.getKey().equals("driver_message"))
          .noneMatch(e -> String.valueOf(e.getValue()).contains("secret")));
    }
  }
}
