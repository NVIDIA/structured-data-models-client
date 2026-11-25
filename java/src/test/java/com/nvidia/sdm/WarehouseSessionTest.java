// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static com.nvidia.sdm.RelationalTest.*;
import static org.junit.jupiter.api.Assertions.*;

import com.nvidia.sdm.internal.ArrowData;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Public warehouse materialization followed by reusable sessions, without further SQL. */
class WarehouseSessionTest {
  static Connection open(String provider) throws Exception {
    Connection connection = DriverManager.getConnection(
        provider.equals("duckdb") ? "jdbc:duckdb:" : "jdbc:sqlite::memory:");
    try {
      try (Statement sql = connection.createStatement()) {
        sql.execute("CREATE TABLE users (user_id BIGINT, value DOUBLE)");
        sql.execute("CREATE TABLE orders (order_id BIGINT, user_id BIGINT, price DOUBLE, time "
            + (provider.equals("duckdb") ? "TIMESTAMP" : "TEXT") + ")");
      }
      try (PreparedStatement user = connection.prepareStatement("INSERT INTO users VALUES (?, ?)")) {
        for (int i = 1; i <= 20; i++) {
          user.setLong(1, i); user.setDouble(2, i * 1.25); user.executeUpdate();
        }
      }
      try (PreparedStatement order = connection.prepareStatement("INSERT INTO orders VALUES (?, ?, ?, ?)")) {
        for (int i = 0; i < 400; i++) {
          order.setLong(1, i); order.setLong(2, i % 20 + 1); order.setDouble(3, i * 1.25);
          order.setString(4, LocalDate.of(2024, 1, 1).plusDays(i) + " 00:00:00");
          order.executeUpdate();
        }
      }
      return connection;
    } catch (Exception error) { connection.close(); throw error; }
  }

  static Schema schema(String provider, Connection connection) {
    Source source = provider.equals("duckdb") ? DuckDbSource.of(connection) : SqliteSource.of(connection);
    return Schema.fromSource(source).withPrimaryKey("users", "user_id")
        .withPrimaryKey("orders", "order_id").withTimeColumn("orders", "time")
        .withStype("users", "value", "numerical")
        .withRelationships(List.of(new Relationship("orders", "user_id", "users", "user_id")));
  }

  static MaterializeOptions.Builder asked(Object... ids) {
    return MaterializeOptions.builder().indices(ids).randomSeed(7).numContextSamples(20)
        .anchorTime(LocalDate.of(2024, 6, 1));
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void scalarSessionsRemoveHeldEntitiesAndScoreAfterConnectionCloses(String provider) throws Exception {
    Request opening;
    Request scoring;
    try (Connection connection = open(provider)) {
      Schema schema = schema(provider, connection);
      opening = PredictiveQuery.parse(SCALAR).materialize(schema, asked(1L, 2L).build());
      scoring = PredictiveQuery.parse(SCALAR).materialize(schema, asked(3L, 4L).build());
    }
    List<String> original = opening.payloads();
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client();
        FittedModel fitted = client.fit(opening, SdmClient.KUMO_RELATIONAL, List.of(3L, 4L))) {
      Map<String, Object> uploaded = endpoint.created.get(0);
      assertEquals(opening.contextRows() - 2, rows(instance(uploaded, "context")).size());
      Set<Object> kept = instanceIds(instance(uploaded, "context"));
      Map<String, Object> related = map(map(uploaded.get("context")).get("related_tables"));
      assertFalse(related.isEmpty());
      for (Object table : related.values()) assertTrue(kept.containsAll(instanceIds(map(table))));
      assertEquals(original, opening.payloads());
      assertEquals(2, fitted.predict(scoring).rows());
      assertEquals(2, fitted.predict(scoring).rows());
      assertEquals(2, endpoint.scored.size());
      assertTrue(endpoint.scored.stream().allMatch(body ->
          !body.containsKey("context") && !body.containsKey("schema") && !body.containsKey("task")));
    }
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void temporalSessionsReuseTheMaterializedWindow(String provider) throws Exception {
    Request request;
    try (Connection connection = open(provider)) {
      request = PredictiveQuery.parse(TEMPORAL).materialize(schema(provider, connection), asked(1L, 2L).build());
    }
    assertTrue(request.targetMovesWithTime());
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client();
        FittedModel fitted = client.fit(request, SdmClient.KUMO_RELATIONAL)) {
      assertEquals(2, fitted.predict(request).rows());
      assertEquals(2, fitted.predict(request).rows());
      assertEquals(1, endpoint.created.size());
      assertEquals(2, endpoint.scored.size());
    }
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void changedQuestionAndEmptyPinnedContextAreRefusedBeforeScoring(String provider) throws Exception {
    Request opening;
    Request changed;
    try (Connection connection = open(provider)) {
      Schema schema = schema(provider, connection);
      opening = PredictiveQuery.parse(SCALAR).materialize(schema, asked(1L, 2L).build());
      changed = PredictiveQuery.parse("PREDICT users.value > 8 FOR EACH users.user_id")
          .materialize(schema, asked(3L, 4L).build());
    }
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client()) {
      List<Long> all = java.util.stream.LongStream.rangeClosed(1, 20).boxed().toList();
      assertThrows(StructuredDataException.class, () -> client.fit(opening, SdmClient.KUMO_RELATIONAL, all));
      assertTrue(endpoint.created.isEmpty());
      try (FittedModel fitted = client.fit(opening, SdmClient.KUMO_RELATIONAL)) {
        assertThrows(StructuredDataException.class, () -> fitted.predict(changed));
        assertTrue(endpoint.scored.isEmpty());
      }
      assertEquals(1, endpoint.deletes.get());
    }
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void laterBatchFailurePreservesAlreadyPaidForRows(String provider) throws Exception {
    Request opening;
    Request scoring;
    try (Connection connection = open(provider)) {
      Schema schema = schema(provider, connection);
      opening = PredictiveQuery.parse(SCALAR).materialize(schema, asked(1L).build());
      scoring = PredictiveQuery.parse(SCALAR).materialize(schema, asked(2L, 3L).rowsPerCall(1).build());
    }
    try (Endpoint endpoint = new Endpoint(); SdmClient client = endpoint.client();
        FittedModel fitted = client.fit(opening, SdmClient.KUMO_RELATIONAL)) {
      endpoint.failScoreOn = 2;
      StructuredDataException error = assertThrows(StructuredDataException.class, () -> fitted.predict(scoring));
      assertInstanceOf(ModelEndpointException.class, error);
      assertEquals(1, ((Number) error.details().get("calls_answered")).intValue());
      assertEquals(1, ((Number) error.details().get("rows_answered")).intValue());
      try (ArrowReader answered = error.answered(ArrowData.allocator()).orElseThrow()) {
        assertTrue(answered.loadNextBatch()); assertEquals(1, answered.getVectorSchemaRoot().getRowCount());
        assertFalse(answered.loadNextBatch());
      }
    }
  }

  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void failedDeletionRetriesAndClientCloseStillAttemptsOtherSessions(String provider) throws Exception {
    Request request;
    try (Connection connection = open(provider)) {
      request = PredictiveQuery.parse(SCALAR).materialize(schema(provider, connection), asked(1L).build());
    }
    try (Endpoint endpoint = new Endpoint()) {
      SdmClient client = endpoint.client();
      FittedModel first = client.fit(request, SdmClient.KUMO_RELATIONAL);
      endpoint.deleteFailures.set(1);
      assertThrows(StructuredDataException.class, first::close);
      assertFalse(first.isClosed()); first.close(); assertTrue(first.isClosed());
      FittedModel second = client.fit(request, SdmClient.KUMO_RELATIONAL);
      FittedModel third = client.fit(request, SdmClient.KUMO_RELATIONAL);
      endpoint.deleteFailures.set(1);
      assertThrows(StructuredDataException.class, client::close);
      assertEquals(4, endpoint.deletes.get()); assertTrue(second.isClosed() ^ third.isClosed());
      client.close(); assertEquals(4, endpoint.deletes.get());
    }
  }
}
