// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;

class ParityHarnessTest {
  @Test
  void typedAllNullFixtureKeepsItsDeclaredTypeThroughThePublicReader() throws Exception {
    Map<String, Object> fixture = Map.of("tables", Map.of("users", Map.of(
        "columns", List.of("id", "all_null"),
        "types", Map.of("all_null", "string"),
        "rows", List.of(java.util.Arrays.asList(1L, null), java.util.Arrays.asList(2L, null)))));
    try (RootAllocator allocator = new RootAllocator(); var input = new ParityRunner.Inputs()) {
      input.open("duckdb", Map.of(), fixture, allocator);
      var table = input.tables.get("users");
      assertEquals(2, table.getRowCount());
      assertEquals(org.apache.arrow.vector.types.pojo.ArrowType.Utf8.INSTANCE,
          table.getSchema().findField("all_null").getType());
      var reader = table.getReader(1);
      for (int row = 0; row < table.getRowCount(); row++) {
        reader.setPosition(row);
        assertNull(reader.readObject());
      }
    }
  }

  @Test
  void decimalEvidenceKeepsExactFractionalAndWideIntegerValues() {
    assertEquals(
        Map.of("decimal", "1.123456"), ParityRunner.scalar(new java.math.BigDecimal("1.123456")));
    assertEquals(1.25, ParityRunner.scalar(new java.math.BigDecimal("1.25")));
    assertEquals(
        new java.math.BigInteger("9007199254740993"),
        ParityRunner.scalar(new java.math.BigDecimal("9007199254740993")));
  }

  @Test
  void fixtureVerificationRejectsChangedValuesAndAllowsDriverRowOrder() throws Exception {
    Map<String, Object> declared =
        Map.of(
            "columns", List.of("id", "value"), "rows", List.of(List.of(1L, 2.0), List.of(2L, 3.0)));
    Map<String, Object> fixture =
        Map.of("tables", Map.of("users", declared), "primary_keys", Map.of("users", "id"));
    try (var allocator = new RootAllocator();
        var reordered =
            ParityRunner.table(
                Map.of(
                    "columns",
                    List.of("id", "value"),
                    "rows",
                    List.of(List.of(2L, 3.0), List.of(1L, 2.0))),
                allocator);
        var changed =
            ParityRunner.table(
                Map.of(
                    "columns",
                    List.of("id", "value"),
                    "rows",
                    List.of(List.of(1L, 2.0), List.of(2L, 4.0))),
                allocator)) {
      assertDoesNotThrow(() -> ParityRunner.verify(Map.of("users", reordered), fixture, false));
      assertThrows(
          ParityRunner.Incomplete.class,
          () -> ParityRunner.verify(Map.of("users", changed), fixture, false));
    }
  }

  @Test
  void canonicalFixtureDigestMatchesPythonSortedCompactJson() throws Exception {
    assertEquals(
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777",
        ParityRunner.digest(Map.of("b", 2L, "a", 1L)));
  }

  @Test
  void unicodeFixtureDigestMatchesPythonAsciiEscapesAndCodePointOrdering() throws Exception {
    assertEquals(
        "aded8de5b0dca721dbd433fa6eeccf2478e932d4a0f41f00d8022e32c0ce9853",
        ParityRunner.digest(
            Map.of("é", "こんにちは🐍", "text", List.of("e\u0301", "\u007f", "\\u00e9", "line\nnext"))));
    assertEquals(
        "672b1a96a5e20d692bc0e7f96715ab1ce3647485dd47a747d38e718b967cf760",
        ParityRunner.digest(Map.of("\ue000", 1L, "\ud800\udc00", 2L)));
  }

  @Test
  void listReaderSchemaIncludesNestedTypeAndNullabilityWithoutChangingValues() {
    var item =
        new org.apache.arrow.vector.types.pojo.Field(
            "item",
            org.apache.arrow.vector.types.pojo.FieldType.nullable(
                org.apache.arrow.vector.types.pojo.ArrowType.Utf8.INSTANCE),
            null);
    var list =
        new org.apache.arrow.vector.types.pojo.Field(
            "tags",
            org.apache.arrow.vector.types.pojo.FieldType.nullable(
                org.apache.arrow.vector.types.pojo.ArrowType.List.INSTANCE),
            List.of(item));
    assertEquals("list<item: string>", ParityRunner.arrowType(list));
    var element =
        new org.apache.arrow.vector.types.pojo.Field(
            "element",
            new org.apache.arrow.vector.types.pojo.FieldType(
                false,
                new org.apache.arrow.vector.types.pojo.ArrowType.FloatingPoint(
                    org.apache.arrow.vector.types.FloatingPointPrecision.SINGLE),
                null),
            null);
    var fixed =
        new org.apache.arrow.vector.types.pojo.Field(
            "vector",
            org.apache.arrow.vector.types.pojo.FieldType.nullable(
                new org.apache.arrow.vector.types.pojo.ArrowType.FixedSizeList(2)),
            List.of(element));
    assertEquals("fixed_size_list<item: float not null>[2]", ParityRunner.arrowType(fixed));
  }

  @Test
  void dateFixturePreservesNullablePreEpochAndLeapDayValues() {
    var values = java.util.Arrays.asList("1969-12-31", "2024-02-29", "2024-03-10", null);
    var rows = values.stream().map(value -> java.util.Arrays.asList(value)).toList();
    try (var allocator = new RootAllocator();
        var table =
            ParityRunner.table(
                Map.of(
                    "columns",
                    List.of("calendar_date"),
                    "types",
                    Map.of("calendar_date", "date32"),
                    "rows",
                    rows),
                allocator)) {
      assertEquals("date32[day]", ParityRunner.arrowType(table.getSchema().getFields().get(0)));
      var reader = table.getReader(0);
      reader.setPosition(0);
      assertEquals(-1, reader.readObject());
      reader.setPosition(1);
      assertEquals(19782, reader.readObject());
      reader.setPosition(2);
      assertEquals(19792, reader.readObject());
      reader.setPosition(3);
      assertNull(reader.readObject());
      assertEquals(
          Map.of("columns", List.of("calendar_date"), "rows", rows), ParityRunner.frame(table));
    }
  }

  @Test
  void routeCannotPassWithoutNativePhaseWitnesses() throws Exception {
    var record = new java.util.LinkedHashMap<String, Object>();
    record.put("status", "passed");
    record.put("semantic", Map.of("lifecycle", List.of("materialize")));
    try (var route = new ParityRoute()) {
      route.begin(record);
      route.end(record);
    }
    assertEquals("incomplete", record.get("status"));
    assertEquals("native operation witness missing for a required phase", record.get("reason"));
  }

  @Test
  void expectedOutcomesDoNotAllowArbitraryRefusals() {
    var error = new java.util.LinkedHashMap<String, Object>();
    error.put("stage", "restore");
    error.put("code", "UNSUPPORTED_REQUEST");
    error.put("rule", null);
    Map<String, Object> record = Map.of("status", "refused", "error", error);
    assertFalse(ParityRunner.accepted(record, Map.of("status", "supported")));
    var expected = new java.util.LinkedHashMap<>(error);
    expected.put("status", "refused");
    assertTrue(ParityRunner.accepted(record, expected));
    expected.put("stage", "predict");
    assertFalse(ParityRunner.accepted(record, expected));
    var local = Map.<String, Object>of("status", "passed", "execution", "local");
    assertTrue(ParityRunner.accepted(local, expected));
    assertFalse(
        ParityRunner.accepted(Map.of("status", "passed", "execution", "live"), expected));
    expected.put("stage", "materialize");
    assertFalse(ParityRunner.accepted(local, expected));
  }
}
