// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import org.junit.jupiter.api.Test;

class ReportsTest {
  @Test void metadataIsImmutableAndLocal() {
    Table table = new Table("users", null, List.of(new Column("id", "int64", "ID", null, null)),
        List.of("id"), null, null);
    TableMetadata report = table.metadataReport();
    assertEquals(List.of("id"), report.primaryKey());
    assertEquals("users\n  id: int64, ID (primary key)", report.toText());
    assertThrows(UnsupportedOperationException.class, () -> report.columns().clear());
    assertEquals("empty", new TableMetadata("empty", List.of(), List.of(), null).toText());
  }

  @Test void explanationFreezesUnknownsAndChecksFieldsLazily() {
    Map<String, Object> raw = new LinkedHashMap<>();
    raw.put("summary", "Why"); raw.put("unknown", null);
    raw.put("details", Map.of("warning", "Careful"));
    ExplanationView view = new ExplanationView(raw);
    raw.put("summary", "changed");
    assertEquals("Why\nWarning: Careful", view.toText());
    assertTrue(view.raw().containsKey("unknown"));
    assertNull(view.raw().get("unknown"));
    assertThrows(UnsupportedOperationException.class, () -> view.raw().clear());
    assertNull(view.featureImportance(false));
    ExplanationView malformed = new ExplanationView(Map.of("summary", 1, "cohorts", List.of()));
    assertEquals(List.of(), malformed.cohorts());
    StructuredDataException error = assertThrows(StructuredDataException.class, malformed::summary);
    assertEquals("INVALID_REQUEST", error.code());
    assertEquals("$.summary", error.details().get("path"));
    raw.put("cycle", raw);
    assertThrows(StructuredDataException.class, () -> new ExplanationView(raw));
  }

  @Test void sharedAttributionFixtures() throws Exception {
    Object parsed = com.nvidia.sdm.internal.Json.read(java.nio.file.Files.readAllBytes(
        java.nio.file.Path.of("../tests/fixtures/reports/attribution.json")));
    for (Object entry : (List<?>) parsed) {
      Map<?, ?> fixture = (Map<?, ?>) entry;
      @SuppressWarnings("unchecked")
      ExplanationView view = new ExplanationView((Map<String, Object>) fixture.get("payload"));
      boolean normalize = (Boolean) fixture.get("normalize");
      if (fixture.containsKey("error")) {
        StructuredDataException error = assertThrows(StructuredDataException.class, () -> view.featureImportance(normalize));
        assertEquals("INVALID_REQUEST", error.code(), fixture.get("name").toString());
        assertEquals(fixture.get("error"), error.details().get("path"));
      } else {
        FeatureImportance report = view.featureImportance(normalize);
        if (fixture.get("rows") == null) assertNull(report);
        else {
          List<?> expected = (List<?>) fixture.get("rows");
          assertEquals(normalize, report.normalized());
          assertEquals(expected.size(), report.rows().size());
          for (int i = 0; i < expected.size(); i++) {
            List<?> row = (List<?>) expected.get(i);
            assertEquals(row.get(0), report.rows().get(i).table());
            assertEquals(row.get(1), report.rows().get(i).column());
            assertEquals(((Number) row.get(2)).doubleValue(), report.rows().get(i).score());
          }
        }
      }
    }
  }

  @Test void nonfiniteScoresUseSharedErrors() {
    for (Number score : new Number[]{Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, java.math.BigInteger.TEN.pow(5000)}) {
      ExplanationView view = new ExplanationView(Map.of("format", "kumo_rfm_v2_1", "details",
          Map.of("subgraphs", List.of(Map.of("tables", Map.of("t", Map.of("n", Map.of("cells", Map.of("c", Map.of("score", score))))))))));
      StructuredDataException error = assertThrows(StructuredDataException.class, () -> view.featureImportance(false));
      assertEquals("INVALID_REQUEST", error.code());
      assertEquals("$.details.subgraphs[0].tables.t.n.cells.c.score", error.details().get("path"));
    }
  }

  @Test void sharedMetadataFixture() throws Exception {
    Map<?, ?> fixture = (Map<?, ?>) com.nvidia.sdm.internal.Json.read(java.nio.file.Files.readAllBytes(
        java.nio.file.Path.of("../tests/fixtures/reports/metadata.json")));
    List<TableMetadata> tables = new ArrayList<>();
    for (Object entry : (List<?>) fixture.get("tables")) {
      Map<?, ?> table = (Map<?, ?>) entry;
      List<ColumnMetadata> columns = new ArrayList<>();
      for (Object value : (List<?>) table.get("columns")) {
        Map<?, ?> column = (Map<?, ?>) value;
        columns.add(new ColumnMetadata((String) column.get("name"), (String) column.get("dtype"), (String) column.get("stype")));
      }
      tables.add(new TableMetadata((String) table.get("name"), columns,
          ((List<?>) table.get("primary_key")).stream().map(String.class::cast).toList(), (String) table.get("time_column")));
    }
    assertEquals(fixture.get("text"), new MetadataReport(tables).toText());
    List<LinkMetadata> links = new ArrayList<>();
    for (Object entry : (List<?>) fixture.get("links")) {
      Map<?, ?> link = (Map<?, ?>) entry;
      links.add(new LinkMetadata((String) link.get("source_table"), (String) link.get("source_column"),
          (String) link.get("target_table"), (String) link.get("target_column")));
    }
    assertEquals(fixture.get("link_text"), new LinkReport(links).toText());
  }
}
