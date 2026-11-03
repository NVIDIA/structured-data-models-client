// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import com.nvidia.sdm.internal.Json;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class LabelsTest {
  @SuppressWarnings("unchecked")
  static Map<String,Object> fixture() throws Exception {
    return (Map<String,Object>) Json.read(Files.readAllBytes(Path.of("../tests/fixtures/labels/ground-truth.json")));
  }

  static Schema configure(Schema schema) {
    return schema.withPrimaryKey("users", "user_id").withPrimaryKey("items", "item_id")
        .withPrimaryKey("orders", "order_id").withTimeColumn("orders", "time")
        .withStype("users", "value", "numerical").withStype("users", "segment", "categorical")
        .withRelationships(List.of(new Relationship("orders", "user_id", "users", "user_id"),
            new Relationship("orders", "item_id", "items", "item_id")));
  }

  @SuppressWarnings("unchecked")
  static Schema memory(Map<String,Object> fixture) {
    try (RootAllocator allocator = new RootAllocator()) {
      Map<String,Table> tables = new LinkedHashMap<>();
      try {
        for (String name : List.of("users", "items", "orders")) {
          Map<String,Object> columns = (Map<String,Object>) fixture.get(name);
          List<FieldVector> vectors = new ArrayList<>();
          try {
            for (var entry : columns.entrySet()) {
              List<Object> values = (List<Object>) entry.getValue();
              String column = entry.getKey();
              FieldVector vector = column.equals("segment") ? new VarCharVector(column, allocator)
                  : column.equals("time") ? new TimeStampMilliVector(column, allocator)
                  : column.equals("price") ? new Float8Vector(column, allocator) : new BigIntVector(column, allocator);
              vectors.add(vector); vector.allocateNew();
              for (int i = 0; i < values.size(); i++) {
                Object value = values.get(i);
                if (vector instanceof VarCharVector text) text.setSafe(i, value.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                else if (vector instanceof TimeStampMilliVector time) time.setSafe(i, LocalDate.parse(value.toString()).toEpochDay() * 86_400_000L);
                else if (vector instanceof Float8Vector floating) floating.setSafe(i, ((Number)value).doubleValue());
                else ((BigIntVector)vector).setSafe(i, ((Number)value).longValue());
              }
              vector.setValueCount(values.size());
            }
            tables.put(name, new Table(vectors));
          } finally { vectors.forEach(FieldVector::close); }
        }
        return configure(Schema.fromData(tables));
      } finally { tables.values().forEach(Table::close); }
    }
  }

  static Object normalized(Object value) {
    if (value instanceof Number number) return number.doubleValue();
    if (value instanceof List<?> list) return list.stream().map(LabelsTest::normalized).toList();
    return value == null || value instanceof Boolean ? value : value.toString();
  }

  @SuppressWarnings("unchecked")
  static void verify(Schema schema, Map<String,Object> data) {
    for (Object entry : (List<Object>) data.get("cases")) {
      Map<String,Object> test = (Map<String,Object>) entry;
      try (Table output = PredictiveQuery.parse("PREDICT " + test.get("target") + " FOR EACH users.user_id")
          .labels(schema, LabelsOptions.builder(100).anchorTime(LocalDate.parse(data.get("anchor").toString())).build())) {
        assertEquals(List.of("ENTITY", "ANCHOR_TIMESTAMP", "TARGET"), output.getSchema().getFields().stream().map(f -> f.getName()).toList());
        Map<String,Object> actual = new LinkedHashMap<>();
        try (FieldVector ids = output.getVectorCopy(0); FieldVector targets = output.getVectorCopy(2)) {
          for (int i = 0; i < output.getRowCount(); i++) actual.put(ids.getObject(i).toString(), normalized(targets.getObject(i)));
        }
        Map<String,Object> expected = new LinkedHashMap<>();
        ((Map<String,Object>)test.get("expected")).forEach((key,value) -> expected.put(key, normalized(value)));
        assertEquals(expected, actual, test.get("name").toString());
      }
    }
    Map<String,Object> forecast = (Map<String,Object>) data.get("forecast");
    try (Table output = PredictiveQuery.parse(forecast.get("query").toString()).labels(schema,
        LabelsOptions.builder(3).anchorTime(LocalDate.parse(data.get("anchor").toString())).build());
        FieldVector ids = output.getVectorCopy(0); FieldVector targets = output.getVectorCopy(2)) {
      assertEquals(3, output.getRowCount());
      for (int i = 0; i < 3; i++) {
        assertEquals(normalized(((List<?>)forecast.get("entity")).get(i)), normalized(ids.getObject(i)));
        assertEquals(normalized(((List<?>)forecast.get("target")).get(i)), normalized(targets.getObject(i)));
      }
    }
  }

  @Test void sharedMemoryGroundTruthAndOwnedOutput() throws Exception {
    Map<String,Object> data = fixture();
    Schema schema = memory(data);
    verify(schema, data);
    try (Table output = PredictiveQuery.parse("PREDICT users.value FOR EACH users.user_id")
        .labels(schema, LabelsOptions.builder(0).build())) {
      assertEquals(0, output.getRowCount());
      assertEquals(3, output.getVectorCount());
    }
    assertThrows(StructuredDataException.class, () -> LabelsOptions.builder(-1).build());
    assertThrows(StructuredDataException.class, () -> LabelsOptions.builder(1).maxIterations(0).build());
  }

  @ParameterizedTest @ValueSource(strings = {"sqlite", "duckdb"})
  @SuppressWarnings("unchecked")
  void sharedProviderGroundTruth(String provider) throws Exception {
    Map<String,Object> data = fixture();
    try (Connection connection = DriverManager.getConnection(provider.equals("duckdb") ? "jdbc:duckdb:" : "jdbc:sqlite::memory:")) {
      Map<String,String> declarations = Map.of("users", "user_id BIGINT, value BIGINT, segment VARCHAR", "items", "item_id BIGINT",
          "orders", "order_id BIGINT, user_id BIGINT, item_id BIGINT, price DOUBLE, time " + (provider.equals("duckdb") ? "TIMESTAMP" : "TEXT"));
      for (String name : List.of("users", "items", "orders")) {
        Map<String,Object> columns = (Map<String,Object>)data.get(name);
        try (var sql = connection.createStatement()) { sql.execute("CREATE TABLE " + name + " (" + declarations.get(name) + ")"); }
        List<List<Object>> values = columns.values().stream().map(v -> (List<Object>)v).toList();
        try (var statement = connection.prepareStatement("INSERT INTO " + name + " VALUES (" + String.join(",", java.util.Collections.nCopies(values.size(), "?")) + ")")) {
          for (int row = 0; row < values.get(0).size(); row++) {
            for (int col = 0; col < values.size(); col++) statement.setObject(col + 1, values.get(col).get(row));
            statement.executeUpdate();
          }
        }
      }
      verify(configure(Schema.fromSource(provider.equals("duckdb") ? DuckDbSource.of(connection) : SqliteSource.of(connection))), data);
    }
  }
}
