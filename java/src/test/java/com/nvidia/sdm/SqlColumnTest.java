// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.sql.DriverManager;
import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.FieldVector;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class SqlColumnTest {
  @ParameterizedTest @ValueSource(strings = {"duckdb", "sqlite"})
  void projectedValuesAreOwnedAndSurviveRefresh(String backend) throws Exception {
    try (var connection = DriverManager.getConnection("jdbc:" + backend + ":" + (backend.equals("sqlite") ? ":memory:" : ""));
         var statement = connection.createStatement()) {
      statement.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, quantity INTEGER, price DOUBLE)");
      for (int i = 0; i < 30; i++) statement.execute("INSERT INTO orders VALUES (" + i + ", " + (i + 1) + ", " + (i == 2 ? "NULL" : "2.5") + ")");
      Source source = backend.equals("duckdb") ? DuckDbSource.of(connection) : SqliteSource.of(connection);
      Schema original = Schema.fromSource(source).withPrimaryKey("orders", "id").withStype("orders", "quantity", "categorical");
      Schema changed = original.withSqlColumn("orders", "line_total", "price * quantity").withStype("orders", "line_total", "numerical").withRelationships(List.of()).inferMetadata();
      assertThrows(StructuredDataException.class, () -> original.table("orders").column("line_total"));
      assertEquals("categorical", changed.table("orders").column("quantity").stype());
      assertEquals(original.table("orders").primaryKey(), changed.table("orders").primaryKey());
      for (int run = 0; run < 2; run++) {
        try (var output = PredictiveQuery.parse("PREDICT orders.line_total FOR EACH orders.id").labels(changed, LabelsOptions.builder(100).build());
             FieldVector ids = output.getVectorCopy(0); FieldVector labels = output.getVectorCopy(2)) {
          assertEquals(29, output.getRowCount());
          for (int i = 0; i < output.getRowCount(); i++) {
            long id = ((Number) ids.getObject(i)).longValue();
            assertNotEquals(2, id);
            assertEquals((id + 1) * 2.5, ((Number) labels.getObject(i)).doubleValue());
          }
        }
      }
      Request request = PredictiveQuery.parse("PREDICT orders.line_total FOR EACH orders.id").materialize(changed,
          MaterializeOptions.builder().indices(List.of(0,1)).randomSeed(7).build());
      assertEquals(2, request.predictRows());
      assertThrows(StructuredDataException.class, () -> request.writeTo(new java.io.ByteArrayOutputStream()));
      assertThrows(StructuredDataException.class, () -> original.withSqlColumn("orders", "price", "1"));
      assertThrows(StructuredDataException.class, () -> original.withSqlColumn("orders", "__kumo_bad", "1"));
      assertThrows(RuntimeException.class, () -> changed.withSqlColumn("orders", "chained", "line_total * 2"));
      assertNotNull(changed.withSqlColumn("orders", "quoted name", "price + 1").table("orders").column("quoted name"));
    }
  }
}
