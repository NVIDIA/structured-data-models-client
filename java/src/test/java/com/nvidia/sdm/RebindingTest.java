// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.sql.DriverManager;
import java.util.List;
import org.apache.arrow.vector.FieldVector;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class RebindingTest {
  @SuppressWarnings("try") // Deliberately expire the old connection before rebinding.
  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void replacementPreservesDeclarationsAndOwnership(String backend) throws Exception {
    String url = "jdbc:" + backend + ":" + (backend.equals("sqlite") ? ":memory:" : "");
    try (var first = DriverManager.getConnection(url); var second = DriverManager.getConnection(url)) {
      var connections = List.of(first, second);
      for (int offset = 0; offset < 2; offset++) {
        try (var statement = connections.get(offset).createStatement()) {
          statement.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, value DOUBLE)");
          statement.execute("CREATE TABLE events(event_id INTEGER PRIMARY KEY, item_id INTEGER, opened TIMESTAMP, closed TIMESTAMP)");
          statement.execute("INSERT INTO events VALUES (1, 1, '2024-01-01', '2024-02-01')");
          for (int i = 0; i < 30; i++) statement.execute("INSERT INTO items VALUES (" + i + ", " + (i + offset * 100) + ")");
        }
      }
      Source oldSource = backend.equals("duckdb") ? DuckDbSource.of(first) : SqliteSource.of(first);
      Source newSource = backend.equals("duckdb") ? DuckDbSource.of(second) : SqliteSource.of(second);
      Schema original = Schema.fromSource(oldSource).withPrimaryKey("items", "id").withPrimaryKey("events", "event_id").withTimeColumn("events", "opened").withEndTimeColumn("events", "closed").withRelationships(List.of(new Relationship("events", "item_id", "items", "id"))).withSqlColumn("items", "doubled", "value * 2");
      var query = PredictiveQuery.parse("PREDICT items.doubled FOR EACH items.id");
      Request prepared = query.materialize(original, MaterializeOptions.builder().indices(List.of(0, 1)).build());
      byte[] preparedPayload = prepared.payloadBytes().get(0).clone();
      Schema replacement = original.withSource(newSource);
      assertArrayEquals(preparedPayload, prepared.payloadBytes().get(0));
      assertEquals("opened", replacement.table("events").timeColumn().orElseThrow());
      assertEquals("closed", replacement.table("events").endTimeColumn().orElseThrow());
      assertEquals(original.relationships(), replacement.relationships());
      assertSame(oldSource, original.source());
      assertSame(newSource, replacement.source());
      assertEquals(original.table("items").primaryKey(), replacement.table("items").primaryKey());
      assertEquals(original.table("items").column("doubled").dtype(), replacement.table("items").column("doubled").dtype());
      for (int offset = 0; offset < 2; offset++) {
        Schema schema = offset == 0 ? original : replacement;
        try (var labels = query.labels(schema, LabelsOptions.builder(100).build());
             FieldVector ids = labels.getVectorCopy(0); FieldVector values = labels.getVectorCopy(2)) {
          assertEquals(30, labels.getRowCount());
          for (int i = 0; i < labels.getRowCount(); i++) assertEquals((((Number) ids.getObject(i)).longValue() + offset * 100) * 2.0, ((Number) values.getObject(i)).doubleValue());
        }
      }
      assertEquals(2, prepared.predictRows());
      assertThrows(StructuredDataException.class, () -> EndTimeTest.memory().withSource(newSource));
      try (var incompatible = DriverManager.getConnection(backend.equals("duckdb") ? "jdbc:sqlite::memory:" : "jdbc:duckdb:")) {
        Source wrong = backend.equals("duckdb") ? SqliteSource.of(incompatible) : DuckDbSource.of(incompatible);
        assertThrows(StructuredDataException.class, () -> original.withSource(wrong));
      }
      first.close();
      assertNotNull(original.withSource(newSource));
      try (var statement = second.createStatement()) { statement.execute("ALTER TABLE items DROP COLUMN value"); }
      assertThrows(StructuredDataException.class, () -> original.withSource(newSource));
      assertTrue(first.isClosed());
      assertFalse(second.isClosed());

    }
  }

  @org.junit.jupiter.api.Test
  @org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable(named = "SDM_SQL_COLUMN_CONFIG", matches = ".+")
  void postgresReconnectAndLivePrediction() throws Exception {
    var config = ParityRunner.read(java.nio.file.Path.of(System.getenv("SDM_SQL_COLUMN_CONFIG")));
    var fixture = ParityRunner.read(java.nio.file.Path.of("../tests/compatibility/fixtures/shop.json"));
    try (var allocator = new org.apache.arrow.memory.RootAllocator();
         var oldInputs = new ParityRunner.Inputs(); var newInputs = new ParityRunner.Inputs()) {
      oldInputs.open("postgres", config, fixture, allocator);
      newInputs.open("postgres", config, fixture, allocator);
      Schema original = Schema.fromSource(oldInputs.source, FromSourceOptions.builder()
          .table("users", oldInputs.names.get("users")).primaryKey("users", "id").relationships(List.of()).build())
          .withSqlColumn("users", "doubled", "value * 2");
      Schema rebound = original.withSource(newInputs.source);
      assertSame(oldInputs.source, original.source());
      assertSame(newInputs.source, rebound.source());
      var query = PredictiveQuery.parse("PREDICT users.doubled FOR EACH users.id");
      var expected = new java.util.HashMap<Long, Double>();
      var users = ParityRunner.map(ParityRunner.map(fixture.get("tables")).get("users"));
      for (Object value : ParityRunner.list(users.get("rows"))) {
        var row = ParityRunner.list(value);
        expected.put(((Number) row.get(0)).longValue(), ((Number) row.get(1)).doubleValue() * 2);
      }
      try (var labels = query.labels(rebound, LabelsOptions.builder(100).build());
           FieldVector ids = labels.getVectorCopy(0); FieldVector values = labels.getVectorCopy(2)) {
        assertEquals(expected.size(), labels.getRowCount());
        for (int i = 0; i < labels.getRowCount(); i++) assertEquals(expected.get(((Number)ids.getObject(i)).longValue()), ((Number)values.getObject(i)).doubleValue());
      }
      Request request = query.materialize(rebound, MaterializeOptions.builder().indices(List.of(0, 1)).numContextSamples(12).randomSeed(42).build());
      var endpoint = ParityRunner.map(ParityRunner.map(config.get("endpoints")).get("relational"));
      try (var client = ParityRunner.client(endpoint)) {
        assertEquals(2, client.predict(request, SdmClient.KUMO_RELATIONAL).rows());
      }
    }
  }

  @org.junit.jupiter.api.Test
  void changedDeclaredTypeIsRejected() throws Exception {
    try (var old = DriverManager.getConnection("jdbc:duckdb:"); var replacement = DriverManager.getConnection("jdbc:duckdb:")) {
      try (var statement = old.createStatement()) { statement.execute("CREATE TABLE items AS SELECT 1 AS id, 2.0::DOUBLE AS value"); }
      try (var statement = replacement.createStatement()) { statement.execute("CREATE TABLE items AS SELECT 1 AS id, 'other' AS value"); }
      Schema schema = Schema.fromSource(DuckDbSource.of(old));
      assertThrows(StructuredDataException.class, () -> schema.withSource(DuckDbSource.of(replacement)));
      try (var statement = old.createStatement(); var result = statement.executeQuery("SELECT value FROM items")) {
        assertTrue(result.next());
        assertEquals(2.0, result.getDouble(1));
      }
    }
  }
}
