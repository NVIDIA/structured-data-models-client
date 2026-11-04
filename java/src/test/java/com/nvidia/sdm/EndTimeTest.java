// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import com.nvidia.sdm.internal.Json;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.DriverManager;
import java.time.LocalDate;
import java.util.*;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class EndTimeTest {
  @SuppressWarnings("unchecked")
  static List<List<Object>> rows() throws Exception {
    var fixture = (Map<String,Object>) Json.read(Files.readAllBytes(Path.of("../tests/fixtures/end-time/accounts.json")));
    return (List<List<Object>>) fixture.get("rows");
  }
  static Schema configure(Schema schema) {
    return schema.withPrimaryKey("accounts", "id").withTimeColumn("accounts", "opened")
        .withStype("accounts", "value", "numerical").withEndTimeColumn("accounts", "closed");
  }
  static Schema memory() throws Exception {
    try (var allocator = new RootAllocator(); var id = new BigIntVector("id", allocator);
        var value = new Float8Vector("value", allocator); var opened = new TimeStampMilliVector("opened", allocator);
        var closed = new TimeStampMilliVector("closed", allocator)) {
      List<FieldVector> columns = List.of(id, value, opened, closed);
      columns.forEach(FieldVector::allocateNew);
      var rows = rows();
      for (int i = 0; i < rows.size(); i++) {
        var row = rows.get(i);
        id.setSafe(i, ((Number) row.get(0)).longValue()); value.setSafe(i, ((Number) row.get(1)).doubleValue());
        opened.setSafe(i, LocalDate.parse(row.get(2).toString()).toEpochDay() * 86400000L);
        if (row.get(3) == null) closed.setNull(i);
        else closed.setSafe(i, LocalDate.parse(row.get(3).toString()).toEpochDay() * 86400000L);
      }
      columns.forEach(c -> c.setValueCount(rows.size()));
      try (var table = new org.apache.arrow.vector.table.Table(columns)) {
        return configure(Schema.fromData(Map.of("accounts", table)));
      }
    }
  }
  static Set<Long> labels(Schema schema) {
    var query = PredictiveQuery.parse("PREDICT accounts.value FOR EACH accounts.id");
    try (var table = query.labels(schema, LabelsOptions.builder(100).anchorTime(LocalDate.of(2024,1,4)).build());
        var ids = table.getVectorCopy(0)) {
      Set<Long> result = new HashSet<>();
      for (int i=0; i<table.getRowCount(); i++) result.add(((Number) ids.getObject(i)).longValue());
      return result;
    }
  }
  @SuppressWarnings("unchecked")
  static void verify(Schema schema) throws Exception {
    assertEquals(Set.of(3L,4L,5L), labels(schema));
    assertEquals("closed", schema.table("accounts").endTimeColumn().orElseThrow());
    assertEquals("closed", schema.table("accounts").metadataReport().endTimeColumn());
    for (String bad : List.of("id","opened","value","missing"))
      assertThrows(StructuredDataException.class, () -> schema.withEndTimeColumn("accounts", bad));
    assertEquals(Set.of(0L,1L,2L,3L,4L,5L), labels(schema.withoutEndTimeColumn("accounts")));
    assertEquals(Set.of(3L,4L,5L), labels(schema.withoutTimeColumn("accounts")));
    assertEquals(Set.of(3L,4L,5L), labels(schema));
    var request = PredictiveQuery.parse("PREDICT accounts.value FOR EACH accounts.id").materialize(schema,
        MaterializeOptions.builder().model(SdmClient.KUMO_RELATIONAL).indices(List.of(0,3)).anchorTime(LocalDate.of(2024,1,4)).numContextSamples(3).build());
    var payload = (Map<String,Object>)Json.read(request.payloadBytes().get(0));
    for (String side : List.of("context","predict")) {
      var tables = (Map<String,Object>) ((Map<String,Object>)payload.get(side)).get("related_tables");
      var accounts = (Map<String,Object>)tables.get("accounts");
      var cols = (List<String>)accounts.get("columns");
      for (var row : (List<List<Object>>)accounts.get("rows")) {
        long id = ((Number)row.get(cols.indexOf("id"))).longValue();
        Object closed = row.get(cols.indexOf("closed"));
        if (id==0) assertTrue(closed.toString().startsWith("2024-01-02"));
        else assertNull(closed);
      }
    }
  }
  @Test void memoryRolesEligibilityMaskingAndPersistence() throws Exception {
    Schema schema = memory(); verify(schema);
    var saved = new java.io.ByteArrayOutputStream(); schema.writeTo(saved);
    var restored = Schema.readFrom(new java.io.ByteArrayInputStream(saved.toByteArray()));
    assertEquals("closed", restored.table("accounts").endTimeColumn().orElseThrow());
    assertEquals(labels(schema), labels(restored));
  }
  @ParameterizedTest @ValueSource(strings={"sqlite","duckdb"})
  void sqlRolesEligibilityAndMasking(String provider) throws Exception {
    try (var connection=DriverManager.getConnection(provider.equals("sqlite")?"jdbc:sqlite::memory:":"jdbc:duckdb:")) {
      String dtype=provider.equals("sqlite")?"TEXT":"TIMESTAMP";
      try(var statement=connection.createStatement()) { statement.execute("CREATE TABLE accounts(id BIGINT PRIMARY KEY,value DOUBLE,opened "+dtype+",closed "+dtype+")"); }
      try(var statement=connection.prepareStatement("INSERT INTO accounts VALUES(?,?,?,?)")) {
        for(var row:rows()) { for(int i=0;i<4;i++) statement.setObject(i+1,row.get(i)); statement.executeUpdate(); }
      }
      verify(configure(Schema.fromSource(provider.equals("sqlite")?SqliteSource.of(connection):DuckDbSource.of(connection))));
    }
  }
}
