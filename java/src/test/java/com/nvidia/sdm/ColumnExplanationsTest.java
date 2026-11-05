// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import com.nvidia.sdm.internal.Json;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

class ColumnExplanationsTest {
  @Test void informationalConstantsMatchPublicAnalysisContract() {
    assertEquals(10, ColumnExplanations.NUM_COHORTS);
    assertEquals(500, ColumnExplanations.MAX_CATEGORIES);
    assertEquals(30, ColumnExplanations.MAX_CLASSES);
    assertEquals("COUNT(*)", ColumnExplanations.COUNT);
    assertEquals("OTHER(*)", ColumnExplanations.OTHER);
    assertEquals("N/A", ColumnExplanations.NA);
    assertEquals("instance_table", ColumnExplanations.TABULAR_TABLE);
  }

  private static Map<String, Object> row(Object label) throws Exception {
    Map<String, Object> row = new LinkedHashMap<>();
    row.put("table", "instance_table"); row.put("hop", 0L); row.put("column", "x");
    row.put("stype", "categorical"); row.put("cohort", "a");
    row.put("class_json", new String(Json.write(label), StandardCharsets.UTF_8));
    row.put("share", 0.5); row.put("mean", 1.0); row.put("score", 1.0);
    return row;
  }
  private static ColumnExplanations columns(Object... labels) throws Exception {
    List<Map<String, Object>> rows = new ArrayList<>();
    for (Object label : labels) rows.add(row(label));
    return ColumnExplanations.fromCore(Map.of("status", "ready", "rows", rows));
  }
  private static Map<String, Object> payload() {
    Map<String, Object> specs = new LinkedHashMap<>();
    specs.put("x", Map.of("stype", "categorical"));
    specs.put("y", Map.of("stype", "numerical"));
    return Map.of("task", Map.of("kind", "regression", "target", Map.of("column_name", "y")),
        "schema", Map.of("instance_table", Map.of("columns", specs)),
        "context", Map.of("instance_table", Map.of("columns", List.of("x", "y"),
            "rows", List.of(List.of("a", 0), List.of("a", 2), List.of("b", 8), List.of("b", 10)))));
  }

  @Test void publicConstructorUsesEngineAndMatchesIndependentMeans() {
    ColumnExplanations columns = new ColumnExplanations(payload());
    assertEquals(2, columns.rows().size());
    assertEquals(new ColumnExplanations.Row("instance_table", 0, "x", "categorical", "a", null, 0.5, 1, 1), columns.rows().get(0));
    assertEquals(9.0, columns.rows().get(1).mean());
    assertEquals("b", columns.rows().get(1).cohort());
    assertThrows(UnsupportedOperationException.class, () -> columns.rows().clear());
  }

  @Test void publicContainerDoesNotComputeAndRepresentsAbsence() throws Exception {
    Explanations empty = new Explanations();
    assertTrue(empty.gradients().entries().isEmpty()); assertNull(empty.columns());
    ColumnExplanations value = columns("yes");
    Explanations filled = new Explanations(empty.gradients(), value);
    assertSame(empty.gradients(), filled.gradients()); assertSame(value, filled.columns());
    assertNull(ColumnExplanations.fromCore(null));
    assertNull(ColumnExplanations.fromCore(Map.of("status", "failed", "reason", "nonfinite_statistics")));
    assertNull(ColumnExplanations.fromCore(Map.of("status", "unsupported", "reason", "ranking")));
  }

  @Test void viewsAreDetachedAndClosingArrowDoesNotCloseCallerAllocator() throws Exception {
    Map<String, Object> raw = row("yes");
    ColumnExplanations columns = ColumnExplanations.fromCore(Map.of("status", "ready", "rows", List.of(raw)));
    raw.put("mean", 999.0);
    assertEquals(1.0, columns.rows().get(0).mean());
    try (RootAllocator allocator = new RootAllocator()) {
      try (VectorSchemaRoot view = columns.toArrow(allocator)) {
        assertEquals(List.of("table", "hop", "column", "stype", "cohort", "class", "share", "mean", "score"),
            view.getSchema().getFields().stream().map(org.apache.arrow.vector.types.pojo.Field::getName).toList());
        assertEquals(1, view.getRowCount());
      }
      assertEquals(0, allocator.getAllocatedMemory());
      try (var buffer = allocator.buffer(8)) { buffer.setLong(0, 42); assertEquals(42, buffer.getLong(0)); }
    }
  }

  @Test void failedArrowAllocationReleasesPartialBuffers() throws Exception {
    ColumnExplanations columns = columns("yes");
    try (RootAllocator allocator = new RootAllocator(64)) {
      assertThrows(org.apache.arrow.memory.OutOfMemoryException.class, () -> columns.toArrow(allocator));
      assertEquals(0, allocator.getAllocatedMemory());
      try (var buffer = allocator.buffer(8)) { assertNotNull(buffer); }
    }
  }

  @Test void ordinaryClassLabelsUseNativeArrowTypes() throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        VectorSchemaRoot strings = columns("yes", "no").toArrow(allocator);
        VectorSchemaRoot integers = columns(1L, 9007199254740993L).toArrow(allocator);
        VectorSchemaRoot booleans = columns(false, true).toArrow(allocator)) {
      assertInstanceOf(VarCharVector.class, strings.getVector("class"));
      assertEquals("yes", strings.getVector("class").getObject(0).toString());
      assertInstanceOf(BigIntVector.class, integers.getVector("class"));
      assertEquals(9007199254740993L, integers.getVector("class").getObject(1));
      assertInstanceOf(BitVector.class, booleans.getVector("class"));
      assertEquals(true, booleans.getVector("class").getObject(1));
      assertFalse(integers.getSchema().getCustomMetadata().containsKey("sdm.class_encoding"));
    }
  }

  @Test void mixedWideAndDecimalClassLabelsRoundTripWithEncodingMetadata() throws Exception {
    Object[] labels = {1L, "1", true, new BigInteger("10000000000000000000000000000000000000001"), new BigDecimal("1.00000000000000000001"), null};
    ColumnExplanations columns = columns(labels);
    try (RootAllocator allocator = new RootAllocator(); VectorSchemaRoot view = columns.toArrow(allocator)) {
      assertEquals("json", view.getSchema().getCustomMetadata().get("sdm.class_encoding"));
      for (int i = 0; i < labels.length; i++) {
        assertEquals(labels[i], columns.rows().get(i).classLabel());
        Object raw = view.getVector("class").getObject(i);
        Object decoded = raw == null ? null : Json.read(raw.toString().getBytes(StandardCharsets.UTF_8));
        assertEquals(labels[i], decoded);
      }
    }
  }

  @Test void readyEmptyRetainsColumnsAndRegressionClassIsNull() throws Exception {
    ColumnExplanations empty = ColumnExplanations.fromCore(Map.of("status", "ready", "rows", List.of()));
    assertNotNull(empty); assertTrue(empty.rows().isEmpty());
    try (RootAllocator allocator = new RootAllocator(); VectorSchemaRoot view = empty.toArrow(allocator);
        VectorSchemaRoot regression = columns((Object) null).toArrow(allocator)) {
      assertEquals(0, view.getRowCount()); assertEquals(9, view.getFieldVectors().size());
      assertNull(regression.getVector("class").getObject(0));
    }
  }

  @Test void metadataRejectsNonScalarClassLabels() throws Exception {
    Map<String, Object> row = row(List.of("bad"));
    assertEquals("INTERNAL_ERROR", assertThrows(StructuredDataException.class,
        () -> ColumnExplanations.fromCore(Map.of("status", "ready", "rows", List.of(row)))).code());
  }

  @Test void malformedAnalysisMetadataDoesNotMasqueradeAsAbsence() throws Exception {
    for (Map<String, Object> analysis : List.<Map<String, Object>>of(
        Map.of("status", "unexpected"), Map.of("status", "ready"), Map.of("status", "failed"))) {
      assertEquals("INTERNAL_ERROR", assertThrows(StructuredDataException.class,
          () -> ColumnExplanations.fromCore(analysis)).code());
    }
    for (String field : List.of("table", "column", "cohort", "mean", "share", "score", "hop")) {
      Map<String, Object> row = row("yes"); row.remove(field);
      assertEquals("INTERNAL_ERROR", assertThrows(StructuredDataException.class,
          () -> ColumnExplanations.fromCore(Map.of("status", "ready", "rows", List.of(row)))).code());
    }
    Map<String, Object> row = row("yes"); row.put("mean", Double.POSITIVE_INFINITY);
    assertEquals("INTERNAL_ERROR", assertThrows(StructuredDataException.class,
        () -> ColumnExplanations.fromCore(Map.of("status", "ready", "rows", List.of(row)))).code());
  }

  @Test void directPredictionAndFitReuseContextAnalysisAndHonorHoldout() throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Table context = TabularTest.table(allocator, true, 1, 2, 3);
        Table predict = TabularTest.table(allocator, false, 4, 5);
        TabularTest.Endpoint endpoint = new TabularTest.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      endpoint.gradients = true;
      Request request = new Request(context, predict, "y", "regression");
      PredictionResult explained = client.predict(request, SdmClient.KUMO_TABULAR, List.of("prediction", "explanation"));
      assertNotNull(explained.explanations().columns());
      assertEquals(2, explained.explanations().gradients().entries().size());
      assertEquals(10.5, explained.explanations().columns().rows().stream().mapToDouble(ColumnExplanations.Row::mean).min().orElseThrow());
      try (FittedModel fitted = client.fit(request, SdmClient.KUMO_TABULAR, List.of(1L), List.of("prediction", "explanation"))) {
        PredictionResult scored = fitted.predict(request);
        assertNotNull(scored.explanations().columns());
        assertEquals(21.0, scored.explanations().columns().rows().stream().mapToDouble(ColumnExplanations.Row::mean).min().orElseThrow());
        try (Table noRows = TabularTest.table(allocator, false)) {
          PredictionResult empty = fitted.predict(new Request(context, noRows, "y", "regression"));
          assertEquals(scored.explanations().columns().rows(), empty.explanations().columns().rows());
        }
      }
    }
  }

  @Test void directRequestPersistenceRetainsExplanationInputs() throws Exception {
    Request request;
    try (RootAllocator allocator = new RootAllocator(); Table context = TabularTest.table(allocator, true, 1, 2, 3);
        Table predict = TabularTest.table(allocator, false, 4)) {
      request = new Request(context, predict, "y", "regression");
    }
    ByteArrayOutputStream saved = new ByteArrayOutputStream(); request.writeTo(saved);
    Request restored = Request.readFrom(new ByteArrayInputStream(saved.toByteArray()));
    try (TabularTest.Endpoint endpoint = new TabularTest.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      assertEquals(client.predict(request, SdmClient.KUMO_TABULAR, List.of("prediction", "explanation")).explanations().columns().rows(),
          client.predict(restored, SdmClient.KUMO_TABULAR, List.of("prediction", "explanation")).explanations().columns().rows());
    }
  }
  @Test void mergePreservesEqualAnalysesRejectsDifferentAndAcceptsExplicitOverride() throws Exception {
    try (RootAllocator allocator = new RootAllocator(); Table context = TabularTest.table(allocator, true, 1, 2, 3);
        Table predict = TabularTest.table(allocator, false, 4);
        TabularTest.Endpoint endpoint = new TabularTest.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      PredictionResult original = client.predict(new Request(context, predict, "y", "regression"),
          SdmClient.KUMO_TABULAR, List.of("prediction", "explanation"));
      ColumnExplanations replacement = new ColumnExplanations(payload());
      PredictionResult replaced = PredictionResult.merge(List.of(original), replacement);
      assertEquals(replacement.rows(), replaced.explanations().columns().rows());
      assertEquals(original.explanations().columns().rows(),
          PredictionResult.merge(List.of(original, original)).explanations().columns().rows());
      assertEquals("INVALID_REQUEST", assertThrows(StructuredDataException.class,
          () -> PredictionResult.merge(List.of(original, replaced))).code());
      assertEquals(replacement.rows(), PredictionResult.merge(List.of(original, replaced), replacement).explanations().columns().rows());
    }
  }

  @Test void materializedRequestPersistenceAndLateExplainUseRetainedFitAnalysis() throws Exception {
    Schema schema = RelationalTest.fixture(true, false);
    MaterializeOptions options = RelationalTest.options(1L, 2L).build();
    Request plain = PredictiveQuery.parse(RelationalTest.SCALAR).materialize(schema, options);
    Request explained = PredictiveQuery.parse("EXPLAIN " + RelationalTest.SCALAR).materialize(schema, options);
    ByteArrayOutputStream saved = new ByteArrayOutputStream(); explained.writeTo(saved);
    Request restored = Request.readFrom(new ByteArrayInputStream(saved.toByteArray()));
    try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint(); SdmClient client = endpoint.client()) {
      PredictionResult original = client.predict(explained, SdmClient.KUMO_RELATIONAL);
      PredictionResult recovered = client.predict(restored, SdmClient.KUMO_RELATIONAL);
      assertNotNull(original.explanations().columns());
      assertFalse(original.explanations().columns().rows().isEmpty());
      assertEquals(original.explanations().columns().rows(), recovered.explanations().columns().rows());
      try (FittedModel fitted = client.fit(plain, SdmClient.KUMO_RELATIONAL)) {
        assertNull(fitted.predict(plain).explanations().columns());
        ColumnExplanations late = fitted.predict(explained).explanations().columns();
        assertNotNull(late); assertFalse(late.rows().isEmpty());
        assertEquals(late.rows(), fitted.predict(explained).explanations().columns().rows());
      }
    }
  }

}
