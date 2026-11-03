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
}
