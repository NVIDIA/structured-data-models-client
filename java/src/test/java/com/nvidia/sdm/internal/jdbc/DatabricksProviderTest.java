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
 * Java's Databricks provider: the core's statements run in order on the caller's
 * connection, through the Databricks JDBC driver (a stand-in here, fed answers as the driver
 * gives them), each statement's values bound to its {@code ?} markers in order (a text scalar,
 * and a key set's JSON text the core wrote, as it was handed), as Python's Databricks provider
 * binds them; the last statement's rows read by the Databricks column of the conversion table;
 * and a failure told by the facts it carries (its SQLSTATE and Databricks error class), never its
 * text, as the Python provider tells them.
 */
class DatabricksProviderTest {
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

  /** A metric view joining a filtered fact table to a dimension, as Python's filter test reads. */
  private static final String METRIC_VIEW =
      String.join(
          "\n",
          "version: 0.1",
          "source: cat.sch.order_lines",
          "filter: source.status = 'VALID'",
          "joins:",
          "  - name: customers",
          "    source: cat.sch.customers",
          "    on: source.customer_id = customers.customer_id",
          "dimensions:",
          "  - name: line_id",
          "    expr: source.line_id",
          "  - name: ordered_at",
          "    expr: source.ordered_at",
          "  - name: quantity",
          "    expr: source.quantity",
          "  - name: segment",
          "    expr: customers.segment",
          "measures:",
          "  - name: total_quantity",
          "    expr: SUM(source.quantity)");

  private static final Map<String, List<List<String>>> VIEW_TABLES =
      Map.of(
          "order_lines",
          List.of(
              List.of("line_id", "INT"),
              List.of("customer_id", "INT"),
              List.of("ordered_at", "TIMESTAMP"),
              List.of("quantity", "INT"),
              List.of("status", "STRING")),
          "customers",
          List.of(List.of("customer_id", "INT"), List.of("segment", "STRING")));

  /**
   * A Databricks metric view is a source of the same class in Java, as in Python's {@code
   * test_semantic_model_source.py}: its source table and each join become tables, each dimension
   * a column of the table it reads, and its measures are left out.
   */
  @Test
  void aMetricViewIsDescribedAsTheTablesItJoins() {
    FakeJdbc.Database[] held = new FakeJdbc.Database[1];
    FakeJdbc.Database db =
        new FakeJdbc.Database(
            sql -> {
              List<Object> bound = held[0].ran.get(held[0].ran.size() - 1).bound();
              if (sql.contains("information_schema.columns")) {
                String table =
                    bound.stream().map(String::valueOf).filter(VIEW_TABLES::containsKey).findFirst()
                        .orElseThrow(() -> new AssertionError("no table bound: " + bound));
                List<List<String>> columns = VIEW_TABLES.get(table);
                int n = columns.size();
                return List.of(
                    FakeJdbc.Column.of("column_name", "STRING", columns.stream().map(c -> c.get(0)).toArray()),
                    FakeJdbc.Column.of("full_data_type", "STRING", columns.stream().map(c -> c.get(1)).toArray()),
                    FakeJdbc.Column.of("is_nullable", "STRING", java.util.Collections.nCopies(n, "YES").toArray()),
                    FakeJdbc.Column.of("table_catalog", "STRING", java.util.Collections.nCopies(n, "cat").toArray()),
                    FakeJdbc.Column.of("table_schema", "STRING", java.util.Collections.nCopies(n, "sch").toArray()),
                    FakeJdbc.Column.of("table_name", "STRING", java.util.Collections.nCopies(n, table).toArray()),
                    FakeJdbc.Column.of("table_type", "STRING", java.util.Collections.nCopies(n, "MANAGED").toArray()));
              }
              // No declared constraints: the view's joins are its links.
              if (sql.contains("information_schema.table_constraints")) {
                return List.of(FakeJdbc.Column.of("constraint_type", "STRING"),
                    FakeJdbc.Column.of("constraint_name", "STRING"),
                    FakeJdbc.Column.of("column_name", "STRING"));
              }
              if (sql.contains("information_schema.referential_constraints")) {
                return List.of(FakeJdbc.Column.of("constraint_name", "STRING"),
                    FakeJdbc.Column.of("source_column", "STRING"),
                    FakeJdbc.Column.of("table_catalog", "STRING"),
                    FakeJdbc.Column.of("table_schema", "STRING"),
                    FakeJdbc.Column.of("table_name", "STRING"),
                    FakeJdbc.Column.of("referenced_column", "STRING"));
              }
              if (sql.contains(" AS k, COUNT(*) AS n, COUNT(DISTINCT `")) {
                return keyCounts(sql);
              }
              if (sql.startsWith("DESCRIBE QUERY SELECT ") || sql.startsWith("SELECT `")) {
                return viewRows(sql);
              }
              if (sql.startsWith("DESCRIBE TABLE EXTENDED")) {
                return List.of(
                    FakeJdbc.Column.of("col_name", "STRING", "line_id", "ordered_at", "quantity",
                        "segment", "total_quantity", "", "# Detailed Table Information", "Catalog",
                        "Database", "Type", "View Text"),
                    FakeJdbc.Column.of("data_type", "STRING", "int", "timestamp", "int", "string",
                        "bigint measure", "", "", "cat", "sch", "METRIC_VIEW", METRIC_VIEW),
                    FakeJdbc.Column.of("comment", "STRING", null, null, null, null, null, null,
                        null, null, null, null, null));
              }
              throw new AssertionError("unexpected SQL: " + sql + " " + bound);
            });
    held[0] = db;
    Schema schema =
        Schema.fromSource(
            DatabricksSource.of(db.connection(), "cat", "sch").semanticModel("sales_metrics"));
    assertEquals(
        Map.of(
            // The dimensions of each table, then the join's key the view reads them by.
            "order_lines", List.of("line_id", "ordered_at", "quantity", "customer_id"),
            "customers", List.of("segment", "customer_id")),
        schema.tables().stream()
            .collect(
                java.util.stream.Collectors.toMap(
                    t -> t.name(), t -> t.columns().stream().map(c -> c.name()).toList())));
    assertEquals(
        List.of(new com.nvidia.sdm.Relationship("order_lines", "customer_id", "customers", "customer_id")),
        schema.relationships());
    // The view's filter holds for every row read from its source table.
    for (FakeJdbc.Ran ran : db.ran) {
      if (ran.text().contains("`order_lines`") && !ran.text().startsWith("DESCRIBE TABLE")) {
        assertTrue(ran.text().contains("WHERE (source.status = 'VALID')"), ran.text());
      }
    }
    assertEquals(0, db.statementsOpen);
  }

  /**
   * Each key counted, as the core counts a join's key: its label, the rows, the distinct values,
   * and the rows missing it, over the rows the view keeps.
   */
  private static List<FakeJdbc.Column> keyCounts(String sql) {
    List<Object> labels = new ArrayList<>(), rows = new ArrayList<>();
    List<Object> distinct = new ArrayList<>(), missing = new ArrayList<>();
    for (String part : sql.split("UNION ALL")) {
      java.util.regex.Matcher counted =
          java.util.regex.Pattern.compile("SELECT (\\d+) AS k, .*COUNT\\(DISTINCT `(\\w+)`\\)")
              .matcher(part.trim());
      if (!counted.find()) throw new AssertionError("unexpected key count: " + part);
      String key = counted.group(2);
      List<FakeJdbc.Column> read =
          viewRows("SELECT `" + key + "`" + part.substring(part.indexOf(" FROM ") - 1));
      List<Object> values = read.get(0).values();
      labels.add(Long.parseLong(counted.group(1)));
      rows.add((long) values.size());
      distinct.add(values.stream().filter(v -> v != null).distinct().count());
      missing.add(values.stream().filter(v -> v == null).count());
    }
    return List.of(
        new FakeJdbc.Column("k", "BIGINT", 0, 0, labels),
        new FakeJdbc.Column("n", "BIGINT", 0, 0, rows),
        new FakeJdbc.Column("d", "BIGINT", 0, 0, distinct),
        new FakeJdbc.Column("z", "BIGINT", 0, 0, missing));
  }

  /**
   * The metric view's tables as the stand-in warehouse answers a read of them: the described
   * columns of a query ({@code DESCRIBE QUERY}), or its rows, the fact table's rows those the view's
   * filter keeps.
   */
  private static List<FakeJdbc.Column> viewRows(String sql) {
    boolean describe = sql.startsWith("DESCRIBE QUERY ");
    String select = describe ? sql.substring("DESCRIBE QUERY ".length()) : sql;
    String table = select.contains("`order_lines`") ? "order_lines" : "customers";
    Map<String, String> types = new LinkedHashMap<>();
    for (List<String> column : VIEW_TABLES.get(table)) types.put(column.get(0), column.get(1));
    // Each item of the SELECT list, named by its last quoted name; a timestamp read as its
    // microseconds (UNIX_MICROS), as the core asks Databricks for one.
    List<String> names = new ArrayList<>();
    List<Boolean> micros = new ArrayList<>();
    for (String item : select.substring("SELECT ".length(), select.indexOf(" FROM ")).split(", ")) {
      java.util.regex.Matcher quoted = java.util.regex.Pattern.compile("`(\\w+)`").matcher(item);
      String name = null;
      while (quoted.find()) name = quoted.group(1);
      names.add(name);
      micros.add(item.startsWith("UNIX_MICROS("));
    }
    List<FakeJdbc.Column> columns = new ArrayList<>();
    if (describe) {
      columns.add(FakeJdbc.Column.of("col_name", "STRING", names.toArray()));
      columns.add(
          FakeJdbc.Column.of(
              "data_type", "STRING", names.stream().map(n -> types.get(n).toLowerCase()).toArray()));
      return columns;
    }
    List<Map<String, Object>> rows = new ArrayList<>();
    if (table.equals("order_lines")) {
      for (int i = 0; i < 120; i++) {
        if (i % 3 == 0) continue; // void: the view's filter leaves it out
        rows.add(
            Map.of(
                "line_id", i + 1,
                "customer_id", 1 + i % 6,
                "ordered_at", java.time.LocalDateTime.of(2024, 1, 1, 0, 0).plusDays(i),
                "quantity", 1 + i % 4,
                "status", "VALID"));
      }
    } else {
      for (int i = 1; i <= 6; i++) rows.add(Map.of("customer_id", i, "segment", i % 2 == 1 ? "a" : "b"));
    }
    for (int at = 0; at < names.size(); at++) {
      String name = names.get(at);
      boolean asMicros = micros.get(at);
      columns.add(
          new FakeJdbc.Column(
              name,
              asMicros ? "BIGINT" : types.get(name),
              0,
              0,
              rows.stream()
                  .map(r -> r.get(name))
                  .map(v -> asMicros
                      ? (Object) (((java.time.LocalDateTime) v).toEpochSecond(java.time.ZoneOffset.UTC) * 1_000_000L)
                      : v)
                  .toList()));
    }
    return columns;
  }
}
