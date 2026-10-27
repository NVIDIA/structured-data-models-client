// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.sql.Connection;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Java's schema on the core (T3.12): {@code Schema.fromSource} on DuckDB, SQLite and Postgres
 * through their JDBC drivers, the schema's attributes, its edits made by the core as plain data
 * (the schema.edit and schema.inspect paths of {@code scope-java.yaml}), and the schema as the
 * contract's {@code HeldSchema} ({@code toHeld}, the structure design's fix A).
 */
class T312SchemaTest {
  @ParameterizedTest
  @ValueSource(strings = {"duckdb", "sqlite", "postgres"})
  void theShopIsDescribedAsTheReferenceDescribesIt(String database) throws Exception {
    try (Connection connection = T312Shop.open(database)) {
      Schema schema = Schema.fromSource(T312Shop.source(database, connection));
      assertEquals(List.of("orders", "users"), schema.tables().stream().map(Table::name).sorted().toList());
      Table users = schema.table("users");
      assertEquals(List.of("user_id"), users.primaryKey());
      assertEquals("ID", users.column("user_id").stype());
      assertEquals("int64", users.column("user_id").dtype());
      assertEquals("categorical", users.column("country").stype());
      Table orders = schema.table("orders");
      assertEquals("ts", orders.timeColumn().orElseThrow());
      assertEquals(List.of(new Relationship("orders", "user_id", "users", "user_id")), schema.relationships());
      assertEquals("users", users.sourceName());
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"duckdb", "sqlite", "postgres"})
  void whatTheCallerDeclaresIsTakenByName(String database) throws Exception {
    try (Connection connection = T312Shop.open(database)) {
      Schema schema =
          Schema.fromSource(
              T312Shop.source(database, connection),
              FromSourceOptions.builder()
                  .table("buyers", "users")
                  .table("orders")
                  .primaryKey("orders", "order_id")
                  .noTimeColumn("orders")
                  .checkKeys(false)
                  .build());
      assertEquals(List.of("buyers", "orders"), schema.tables().stream().map(Table::name).toList());
      assertEquals("users", schema.table("buyers").sourceName());
      assertEquals(List.of("order_id"), schema.table("orders").primaryKey());
      assertTrue(schema.table("orders").timeColumn().isEmpty());
      StructuredDataException refused =
          assertThrows(
              StructuredDataException.class,
              () ->
                  Schema.fromSource(
                      T312Shop.source(database, connection),
                      FromSourceOptions.builder().primaryKey("nope", "user_id").build()));
      assertEquals("INVALID_REQUEST", refused.code());
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"duckdb", "sqlite"})
  void anEditGivesANewSchemaAndLeavesTheOldOneAsItWas(String database) throws Exception {
    try (Connection connection = T312Shop.open(database)) {
      Schema schema = Schema.fromSource(T312Shop.source(database, connection));
      Schema meaning = schema.withStype("users", "age", "categorical");
      assertEquals("categorical", meaning.table("users").column("age").stype());
      assertEquals("numerical", schema.table("users").column("age").stype());
      assertNotEquals(schema, meaning);

      Schema leftOut = schema.withoutColumn("users", "country");
      assertFalse(leftOut.table("users").columns().stream().anyMatch(c -> c.name().equals("country")));

      Schema untimed = schema.withoutTimeColumn("orders");
      assertTrue(untimed.table("orders").timeColumn().isEmpty());
      assertEquals("ts", untimed.withTimeColumn("orders", "ts").table("orders").timeColumn().orElseThrow());

      Schema unlinked = schema.withRelationships(List.of());
      assertEquals(List.of(), unlinked.relationships());
      Schema relinked =
          unlinked.withRelationship(new Relationship("orders", "user_id", "users", "user_id"));
      assertEquals(schema.relationships(), relinked.relationships());

      Schema keyed = schema.withPrimaryKey("orders", "order_id");
      assertEquals(List.of("order_id"), keyed.table("orders").primaryKey());

      // The schema is plain data: a snapshot is the schema itself, and it holds together.
      assertSame(schema, schema.snapshot());
      schema.checkConsistent();
      relinked.checkConsistent();
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"duckdb", "sqlite"})
  void anEditTheReferenceRefusesIsRefusedByTheCore(String database) throws Exception {
    try (Connection connection = T312Shop.open(database)) {
      Schema schema = Schema.fromSource(T312Shop.source(database, connection));
      for (Runnable edit :
          List.<Runnable>of(
              () -> schema.withPrimaryKey("users", "nope"),
              () -> schema.withTimeColumn("users", "nope"),
              () -> schema.withStype("users", "age", "no-such-meaning"),
              () -> schema.withoutColumn("users", "user_id"),
              () -> schema.withRelationships(List.of(new Relationship("orders", "nope", "users", "user_id"))))) {
        StructuredDataException refused = assertThrows(StructuredDataException.class, edit::run);
        assertEquals("INVALID_REQUEST", refused.code(), refused.getMessage());
        assertFalse(refused.getMessage().startsWith("the core said"), refused.getMessage());
      }
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"duckdb", "sqlite"})
  @SuppressWarnings("unchecked")
  void theHeldSchemaCarriesWhatTheSourceSaidOfEachColumn(String database) throws Exception {
    try (Connection connection = T312Shop.open(database)) {
      Schema schema = Schema.fromSource(T312Shop.source(database, connection));
      for (Schema held : List.of(schema, schema.withStype("users", "age", "categorical"))) {
        Map<String, Object> written = held.toHeld();
        for (Object table : (List<Object>) written.get("tables")) {
          Map<String, Object> each = (Map<String, Object>) table;
          assertEquals(Map.of("source", List.of(each.get("name"))), each.get("source"));
          for (Object column : (List<Object>) each.get("columns")) {
            Map<String, Object> facts = (Map<String, Object>) ((Map<String, Object>) column).get("source");
            assertEquals(each.get("source"), facts.get("relation"), String.valueOf(column));
            assertTrue(facts.containsKey("received") && facts.containsKey("nullable"));
          }
        }
        assertEquals(
            List.of(List.of(Map.of("table", "orders", "column", "user_id"), Map.of("table", "users", "column", "user_id"))),
            written.get("links"));
      }
    }
  }
}
