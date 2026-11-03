// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Path;
import java.util.*;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;

class SchemaRefinementTest {
  static final Relationship LINK = new Relationship("orders", "customer_id", "customers", "customer_id");

  static Schema schema(boolean links) throws Exception {
    try (var allocator = new RootAllocator();
        var customers = Readers.read(Path.of("../tests/fixtures/schema-refinement/customers.csv"), allocator);
        var orders = Readers.read(Path.of("../tests/fixtures/schema-refinement/orders.csv"), allocator)) {
      Map<String, org.apache.arrow.vector.table.Table> tables = new LinkedHashMap<>();
      tables.put("customers", customers); tables.put("orders", orders);
      Schema schema = Schema.fromData(tables).withPrimaryKey("customers", "customer_id").withPrimaryKey("orders", "order_id");
      return links ? schema : schema.withRelationships(List.of());
    }
  }

  @Test void inferencePreservesOwnershipAndProvenance() throws Exception {
    Schema original = schema(false);
    Schema inferred = original.inferLinks();
    assertEquals(List.of(), original.relationships());
    assertEquals(List.of(LINK), inferred.relationships());
    assertEquals("inferred", inferred.linkOrigins().get(0).origin());
    assertEquals(inferred.linkOrigins(), inferred.inferLinks().linkOrigins());
    var saved = new java.io.ByteArrayOutputStream(); inferred.writeTo(saved);
    assertEquals(inferred.linkOrigins(), Schema.readFrom(new java.io.ByteArrayInputStream(saved.toByteArray())).linkOrigins());
    assertThrows(UnsupportedOperationException.class, () -> inferred.linkOrigins().clear());
  }

  @Test void atomicEditsPreserveOriginalAndRejectMissingNames() throws Exception {
    Schema original = schema(false).withRelationship(LINK);
    assertEquals("declared", original.linkOrigins().get(0).origin());
    assertEquals(original.linkOrigins(), original.withRelationship(LINK).linkOrigins());
    Schema removed = original.withoutTable("orders");
    assertEquals(List.of("customers"), removed.tables().stream().map(Table::name).toList());
    assertTrue(removed.relationships().isEmpty());
    assertTrue(removed.linkOrigins().isEmpty());
    assertEquals(List.of(LINK), original.relationships());
    assertTrue(original.withoutRelationship(LINK).relationships().isEmpty());
    assertEquals("INVALID_REQUEST", assertThrows(StructuredDataException.class, () -> original.withoutTable("absent")).code());
    assertEquals("INVALID_REQUEST", assertThrows(StructuredDataException.class, () -> original.withoutRelationship(new Relationship("x","a","y","b"))).code());
  }

  @Test void explicitNoneAndTypesSurviveMetadataInference() throws Exception {
    Schema original = schema(false).withoutPrimaryKey("customers").withoutTimeColumn("customers").withStype("customers", "value", "categorical");
    Schema inferred = original.inferMetadata();
    assertTrue(inferred.table("customers").primaryKey().isEmpty());
    assertTrue(inferred.table("customers").timeColumn().isEmpty());
    assertEquals("categorical", inferred.table("customers").column("value").stype());
    assertTrue(inferred.relationships().isEmpty());
  }
  @Test void providerInferenceAndPureEditsUseExistingBoundary() throws Exception {
    Schema inferred;
    try (var connection = java.sql.DriverManager.getConnection("jdbc:duckdb:");
        var statement = connection.createStatement()) {
      statement.execute("CREATE TABLE customers(customer_id BIGINT PRIMARY KEY, value DOUBLE)");
      statement.execute("INSERT INTO customers SELECT i, i * 0.5 FROM range(30) t(i)");
      statement.execute("CREATE TABLE orders(order_id BIGINT PRIMARY KEY, customer_id BIGINT, amount DOUBLE)");
      statement.execute("INSERT INTO orders SELECT i, i % 30, i * 0.25 FROM range(60) t(i)");
      Schema original = Schema.fromSource(DuckDbSource.of(connection)).withRelationships(List.of());
      inferred = original.inferLinks();
      assertTrue(original.relationships().isEmpty());
      assertEquals(List.of(LINK), inferred.relationships());
      Request request = PredictiveQuery.parse("PREDICT customers.value FOR EACH customers.customer_id")
          .materialize(inferred, MaterializeOptions.builder().indices(List.of(0, 1, 2)).randomSeed(7).build());
      assertNotNull(request);
      assertEquals(1, inferred.withoutTable("orders").tables().size());
    }
    assertTrue(inferred.withoutRelationship(LINK).relationships().isEmpty());
  }

  @Test void conflictingRelationshipFailsAtomically() throws Exception {
    Schema original = schema(false).withRelationship(LINK);
    assertEquals("INVALID_REQUEST", assertThrows(StructuredDataException.class,
        () -> original.withRelationship(null)).code());
    assertEquals("INVALID_REQUEST", assertThrows(StructuredDataException.class,
        () -> original.withoutRelationship(null)).code());
    Relationship conflict = new Relationship("orders", "customer_id", "orders", "order_id");
    assertEquals("INVALID_REQUEST", assertThrows(StructuredDataException.class,
        () -> original.withRelationship(conflict)).code());
    assertEquals(List.of(LINK), original.relationships());
  }
}
