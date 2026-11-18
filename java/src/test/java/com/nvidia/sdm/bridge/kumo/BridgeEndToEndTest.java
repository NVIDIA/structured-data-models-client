// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.bridge.kumo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.kumo.data.DataAccess;
import ai.kumo.data.Query;
import ai.kumo.data.Rows;
import ai.kumo.graph.Graph;
import ai.kumo.sql.JdbcExecutor;
import ai.kumo.sql.SQLiteCompiler;
import ai.kumo.sql.SQLiteSchema;
import ai.kumo.sql.SqlDataAccess;
import ai.kumo.sql.SqlObserver;
import com.nvidia.sdm.MaterializeOptions;
import com.nvidia.sdm.PredictiveQuery;
import com.nvidia.sdm.Request;
import com.nvidia.sdm.Schema;
import com.nvidia.sdm.SqliteSource;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * One prediction's request built twice from the same SQLite data: through the bridge over the
 * prototype's own {@link SqlDataAccess}, and through the client's built-in SQLite source. Both
 * must send the model the same task, schema and prediction rows. The context sample is drawn by
 * each provider's own sampler (SQLite's hashing for the built-in source, the core's in-memory
 * draw for the bridge), so only its size is compared.
 */
class BridgeEndToEndTest {

  private static Connection database() throws Exception {
    Connection con = DriverManager.getConnection("jdbc:sqlite::memory:");
    try (Statement s = con.createStatement()) {
      s.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER)");
      StringBuilder users = new StringBuilder("INSERT INTO users VALUES");
      for (int id = 1; id <= 12; id++) {
        users.append(id == 1 ? "" : ",").append("(").append(id).append(",'u").append(id).append("',").append(20 + id).append(")");
      }
      s.execute(users.toString());
      s.execute(
          "CREATE TABLE orders(id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id),"
              + " amount REAL, ts TIMESTAMP)");
      StringBuilder orders = new StringBuilder("INSERT INTO orders VALUES");
      int id = 1;
      for (int user = 1; user <= 12; user++) {
        for (int k = 0; k < user % 5 + 1; k++) {
          int day = 1 + (user * 7 + k * 11) % 120;
          LocalDate date = LocalDate.of(2024, 12, 1).plusDays(day);
          orders.append(id == 1 ? "" : ",")
              .append("(").append(id).append(",").append(user).append(",").append(5 + (id * 3) % 17)
              .append(",'").append(date).append("T00:00:00Z')");
          id++;
        }
      }
      s.execute(orders.toString());
    }
    return con;
  }

  /** Records every query the bridge sends, to check it stays within the narrow algebra. */
  private static final class Recording implements DataAccess {
    final DataAccess inner;
    final List<Query> queries = new ArrayList<>();

    Recording(DataAccess inner) {
      this.inner = inner;
    }

    @Override
    public Rows execute(Query request) {
      queries.add(request);
      return inner.execute(request);
    }
  }

  /** The rows with the {@code __node_id} column dropped: it numbers rows across context and predict. */
  @SuppressWarnings("unchecked")
  private static Object withoutNodeIds(Object part) {
    java.util.Map<String, Object> copy = new java.util.LinkedHashMap<>((java.util.Map<String, Object>) part);
    java.util.Map<String, Object> related = new java.util.LinkedHashMap<>((java.util.Map<String, Object>) copy.get("related_tables"));
    for (var entry : related.entrySet()) {
      java.util.Map<String, Object> table = (java.util.Map<String, Object>) entry.getValue();
      List<Object> columns = (List<Object>) table.get("columns");
      int at = columns.indexOf("__node_id");
      if (at < 0) {
        continue;
      }
      List<Object> rows = new ArrayList<>();
      for (Object row : (List<Object>) table.get("rows")) {
        List<Object> kept = new ArrayList<>((List<Object>) row);
        kept.remove(at);
        rows.add(kept);
      }
      java.util.Map<String, Object> trimmed = new java.util.LinkedHashMap<>(table);
      trimmed.put("rows", rows);
      entry.setValue(trimmed);
    }
    copy.put("related_tables", related);
    return copy;
  }

  @Test
  void bridgedAndBuiltInSqliteSendTheSameTaskSchemaAndPredictRows() throws Exception {
    try (Connection con = database()) {
      Graph graph = SQLiteSchema.discover(con).withTimeColumn("orders", "ts");
      Recording access =
          new Recording(new SqlDataAccess(new SQLiteCompiler(), new JdbcExecutor(con, false), SqlObserver.NONE));

      Schema bridged =
          Schema.fromSource(
              KumoBridge.source(graph, access, BridgeOptions.defaults().withMaxKeysPerQuery(3)),
              KumoBridge.options(graph));
      Schema direct = Schema.fromSource(SqliteSource.of(con), KumoBridge.options(graph));

      PredictiveQuery query =
          PredictiveQuery.parse("PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.id");
      MaterializeOptions options =
          MaterializeOptions.builder()
              .indices(1L, 2L, 3L, 4L, 5L)
              .anchorTime(LocalDate.of(2025, 3, 1))
              .numContextSamples(6)
              .randomSeed(7)
              .build();
      Request viaBridge = query.materialize(bridged, options);
      Request viaSqlite = query.materialize(direct, options);

      assertEquals(viaSqlite.contextRows(), viaBridge.contextRows());
      assertEquals(viaSqlite.predictRows(), viaBridge.predictRows());
      assertEquals(1, viaBridge.payloads().size());
      java.util.Map<?, ?> bridgePayload = (java.util.Map<?, ?>) ai.kumo.client.Json.read(viaBridge.payloads().get(0));
      java.util.Map<?, ?> sqlitePayload = (java.util.Map<?, ?>) ai.kumo.client.Json.read(viaSqlite.payloads().get(0));
      for (String part : List.of("model", "task", "schema", "inference", "metadata")) {
        assertEquals(sqlitePayload.get(part), bridgePayload.get(part), part);
      }
      assertEquals(withoutNodeIds(sqlitePayload.get("predict")), withoutNodeIds(bridgePayload.get("predict")));

      assertTrue(!access.queries.isEmpty());
      for (Query sent : access.queries) {
        assertTrue(sent.joins().isEmpty(), "no joins");
        assertTrue(sent.order().isEmpty(), "no order");
        assertTrue(!sent.toString().contains("Scalar["), "no scalar subqueries");
      }
    }
  }
}
