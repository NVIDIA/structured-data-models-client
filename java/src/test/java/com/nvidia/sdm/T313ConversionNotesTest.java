// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.nvidia.sdm.internal.Results;
import java.sql.Connection;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** View notes are the core's sentences, limited to what the resulting schema holds. */
class T313ConversionNotesTest {
  private static Map<String, Object> text(String value) {
    return Map.of("t", "utf8", "v", value);
  }

  @Test
  void conversionNotesKeepTheirOrderAndReferToTheTablesAndTimesActuallyTaken() throws Exception {
    Map<String, Object> measure = Map.of("sentence", "view_measure_note", "table", "orders",
        "fields", Map.of("measure", text("revenue"), "function", text("SUM"),
            "table", text("orders"), "column", text("price")));
    Map<String, Object> omitted = Map.of("sentence", "view_measure_note", "table", "missing",
        "fields", Map.of("measure", text("missing"), "function", text("COUNT"),
            "table", text("missing"), "column", text("*")));
    Map<String, Object> time = Map.of("sentence", "view_time_column_note", "table", "orders",
        "time_column", "ts", "fields", Map.of("column", text("ts"), "table", text("orders"),
            "reasons", text("it is what x truncates")));
    Map<String, Object> parameter = Map.of("sentence", "view_parameter_note", "fields",
        Map.of("parameter", text("floor"), "default", text("1")));
    Results.Described described = (Results.Described) Results.expect(
        Map.of("op", "describe_source", "description", 3,
            "notes", List.of(measure, omitted, time, parameter)), "describe_source");
    String measureWords = "'revenue' is SUM(orders.price, 0, 30, days) over the next 30 days";
    String timeWords = "Took 'ts' as the time column of table 'orders', since it is what x truncates";
    String parameterWords = "Read the parameter 'floor' as its default '1', as a query of the view without arguments does";
    try (Connection connection = T312Shop.open("duckdb")) {
      Schema schema = Schema.fromSource(DuckDbSource.of(connection));
      assertEquals(List.of(), schema.conversionNotes());
      Schema noted = schema.withConversionNotes(described.notes());
      assertEquals(List.of(measureWords, timeWords, parameterWords), noted.conversionNotes());
      assertEquals(List.of(measureWords, parameterWords),
          schema.withoutTimeColumn("orders").withConversionNotes(described.notes()).conversionNotes());
      assertEquals(noted.conversionNotes(), noted.withStype("users", "age", "categorical").conversionNotes());
      assertThrows(UnsupportedOperationException.class, () -> noted.conversionNotes().add("x"));
    }
    Results.Described noNotes = (Results.Described) Results.expect(
        Map.of("op", "describe_source", "description", 3), "describe_source");
    assertEquals(List.of(), noNotes.notes());
  }
}
