// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import static com.nvidia.sdm.CustomExamplesTest.*;

import com.nvidia.sdm.internal.Json;
import java.sql.*;
import java.time.LocalDate;
import java.util.*;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

/** Public Java/JNI PQL paths must share the model's temporal-link batch limit. */
class TemporalLinkBatchingTest {
  static final String QUERY = "PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) FOR EACH users.user_id";

  static Map<String, Object> data() {
    List<List<Object>> users = new ArrayList<>(), items = new ArrayList<>(), orders = new ArrayList<>();
    for (int i = 1; i <= 201; i++) users.add(List.of(i));
    for (int i = 1; i <= 5; i++) items.add(List.of(i, i - 1));
    for (int i = 0; i < 1005; i++) orders.add(List.of(i, i % 201 + 1, i % 5 + 1,
        LocalDate.of(2024, 1, 1).plusDays(i % 100).toString()));
    return Map.of("users", Map.of("columns", List.of("user_id"), "rows", users),
        "items", Map.of("columns", List.of("item_id", "price"), "rows", items),
        "orders", Map.of("columns", List.of("order_id", "user_id", "item_id", "time"), "rows", orders));
  }

  static Schema schema(String kind, Connection connection) throws Exception {
    Schema result;
    if (kind.equals("memory")) {
      try (RootAllocator allocator = new RootAllocator()) {
        Map<String, Table> tables = new LinkedHashMap<>();
        try {
          for (var entry : data().entrySet()) tables.put(entry.getKey(), arrow(allocator, object(entry.getValue()), ""));
          result = Schema.fromData(tables);
        } finally { tables.values().forEach(Table::close); }
      }
    } else {
      try (Statement statement = connection.createStatement()) {
        statement.execute("CREATE TABLE users(user_id BIGINT PRIMARY KEY)");
        statement.execute("CREATE TABLE items(item_id BIGINT PRIMARY KEY, price DOUBLE)");
        statement.execute("CREATE TABLE orders(order_id BIGINT PRIMARY KEY, user_id BIGINT REFERENCES users(user_id), item_id BIGINT REFERENCES items(item_id), time TIMESTAMP)");
      }
      for (var entry : data().entrySet()) {
        var table = object(entry.getValue());
        int columns = ((List<?>) table.get("columns")).size();
        try (PreparedStatement insert = connection.prepareStatement("INSERT INTO " + entry.getKey()
            + " VALUES (" + String.join(",", Collections.nCopies(columns, "?")) + ")")) {
          for (var row : rows(table)) {
            for (int c = 0; c < columns; c++) insert.setObject(c + 1, row.get(c));
            insert.addBatch();
          }
          insert.executeBatch();
        }
      }
      result = Schema.fromSource(kind.equals("duckdb") ? DuckDbSource.of(connection) : SqliteSource.of(connection));
    }
    return result.withPrimaryKey("users", "user_id").withPrimaryKey("items", "item_id")
        .withPrimaryKey("orders", "order_id").withTimeColumn("orders", "time");
  }

  static Request materialize(Schema schema, String query, int count, Integer size) {
    List<Object> ids = new ArrayList<>();
    for (int i = 1; i <= count; i++) ids.add(i);
    var options = MaterializeOptions.builder().model(SdmClient.KUMO_RELATIONAL).indices(ids)
        .anchorTime(LocalDate.of(2024, 3, 1)).numContextSamples(10).option("random_seed", 7);
    if (size != null) options.option("rows_per_call", size);
    return PredictiveQuery.parse(query).materialize(schema, options.build());
  }

  static List<Integer> sizes(Request request) throws Exception {
    List<Integer> result = new ArrayList<>();
    for (String text : request.payloads()) result.add(rows(object(object(object(Json.read(text.getBytes(java.nio.charset.StandardCharsets.UTF_8))).get("predict")).get("instance_table"))).size());
    return result;
  }

  @Test void allPublicInputRoutesUseOneLimitAndKeepEveryEntity() throws Exception {
    for (String kind : List.of("memory", "duckdb", "sqlite")) {
      try (Connection connection = kind.equals("memory") ? null : DriverManager.getConnection(kind.equals("duckdb") ? "jdbc:duckdb:" : "jdbc:sqlite::memory:")) {
        Schema schema = schema(kind, connection);
        for (int count : List.of(199, 200, 201)) {
          Request request = materialize(schema, QUERY, count, null);
          assertEquals(count <= 200 ? List.of(count) : List.of(200, 1), sizes(request), kind);
          List<Integer> ids = new ArrayList<>();
          for (String text : request.payloads()) {
            var payload = object(Json.read(text.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
            assertEquals("temporal_link_prediction", object(payload.get("task")).get("kind"));
            var table = object(object(payload.get("predict")).get("instance_table"));
            int column = ((List<?>) table.get("columns")).indexOf("__kumo_entity_ref_0");
            for (var row : rows(table)) ids.add(Integer.parseInt(row.get(column).toString()));
          }
          assertEquals(java.util.stream.IntStream.rangeClosed(1, count).boxed().toList(), ids);
        }
        assertEquals(List.of(100, 100, 1), sizes(materialize(schema, QUERY, 201, 100)));
        assertEquals(List.of(200, 1), sizes(materialize(schema, QUERY, 201, 1000)));
        assertEquals(List.of(201), sizes(materialize(schema,
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id", 201, null)));
      }
    }
  }

  @Test void suppliedExamplesKeepTheSameCap() throws Exception {
    var fixture = fixture();
    var test = cases(fixture).stream().filter(c -> c.get("id").equals("temporal_link_prediction")).findFirst().orElseThrow();
    var predicted = new LinkedHashMap<>(object(test.get("predict")));
    predicted.put("rows", Collections.nCopies(201, rows(predicted).get(0)));
    try (RootAllocator allocator = new RootAllocator();
        Table context = arrow(allocator, object(test.get("context")), "temporal_link_prediction");
        Table predict = arrow(allocator, predicted, "temporal_link_prediction")) {
      Request request = memory(fixture, false).materializeExamples(task(test), context, predict,
          ExamplesOptions.builder().rowsPerCall(1000).build());
      assertEquals(List.of(200, 1), sizes(request));
    }
  }
}
