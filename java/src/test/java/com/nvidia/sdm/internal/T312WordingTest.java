// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

/**
 * The Java host's words for the core's sentences (T3.12; the structure design's fix B): every
 * advice the core's table names is spelled in Java, every sentence of the table is filled from
 * fields of the shapes its template names, and the value formatter writes times, decimals,
 * values the caller gave, pairs, databases and values only this host holds.
 */
class T312WordingTest {
  @Test
  void everyAdviceTheCoreNamesIsSpelledInJava() {
    assertEquals(
        new java.util.TreeSet<>(Wording.adviceNamed()),
        new java.util.TreeSet<>(Wording.adviceSpelled()));
  }

  @Test
  void adviceIsSaidAsJavaIsWritten() throws Wording.Unfilled {
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("table", "orders");
    fields.put("column", "ts");
    assertEquals(
        "FromSourceOptions.builder().noTimeColumn(\"orders\")",
        Wording.fill("{advice:clear_time_column(table)}", fields));
    assertEquals(
        "FromSourceOptions.builder().primaryKey(\"orders\", \"ts\")",
        Wording.fill("{advice:set_primary_key(table, column)}", fields));
    assertTrue(Wording.fill("{advice:skip_key_check()}", fields).contains("checkKeys(false)"));
    // A semantic model's advice (T3.4 fix 8, T3.27), as Java reads one.
    fields.put("metric_view", "c.s.sales");
    assertEquals(
        "source.semanticModel(\"c.s.sales\", false)",
        Wording.fill("{advice:semantic_model(metric_view)}", fields));
    assertEquals("semanticModel(name, true)", Wording.fill("{advice:trust_expressions()}", fields));
  }

  @Test
  void aKeySentenceNamingAdviceIsSaidWithIt() {
    String said = Wording.say("left_out_key", Map.of("table", "t", "column", "c"));
    assertFalse(said.startsWith("the core said"), said);
    assertTrue(said.contains("null"), said);
  }

  @Test
  void valuesAreWrittenAsJavaWritesThem() throws Wording.Unfilled {
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("day", Map.of("t", "date32", "v", "19723"));
    fields.put("stamp", Map.of("t", "ts", "v", "1709251200000000", "unit", "us"));
    fields.put("price", Map.of("t", "d128", "v", "150", "p", 5, "s", 2));
    fields.put("given", Map.of("t", "utf8", "v", "4.5"));
    fields.put("whole", Map.of("t", "utf8", "v", "12345"));
    fields.put("ones", List.of(Map.of("t", "utf8", "v", "a"), Map.of("t", "utf8", "v", "b")));
    fields.put("twos", List.of(Map.of("t", "i64", "v", "1"), Map.of("t", "i64", "v", "2")));
    fields.put("db", Map.of("t", "utf8", "v", "duckdb"));
    Map<String, Object> read = Wording.fields(Map.of("fields", fields));
    assertEquals(LocalDate.of(2024, 1, 1), read.get("day"));
    assertEquals(new BigDecimal("1.50"), read.get("price"));
    assertEquals("2024-01-01T00:00:00", Wording.fill("{day!t}", read));
    assertEquals("2024-03-01T00:00:00", Wording.fill("{stamp!t}", read));
    assertEquals("1.50", Wording.fill("{price!r}", read));
    assertEquals("4.5|Double|4.5", Wording.fill("{given!G}|{given!N}|{given!S}", read));
    assertEquals("12,345", Wording.fill("{whole!I:,}", read));
    assertEquals("\"a\" to 1, \"b\" to 2", Wording.fill("{ones:to(twos)}", read));
    assertEquals("[\"a\", \"b\"]", Wording.fill("{ones!p}", read));
    assertFalse(Wording.fill("{db:place}", read).isEmpty());
    Map<String, Function<Map<String, Object>, Object>> held = Map.of("source", f -> "the source");
    assertEquals("\"the source\" holds", Wording.fill("{host.source!r} holds", read, held));
  }

  @Test
  void everySentenceIsFilledFromFieldsOfItsShape() {
    for (Map.Entry<String, String> entry : Wording.templates().entrySet()) {
      String template = entry.getValue();
      assertFalse(template == null || template.isEmpty(), entry.getKey());
    }
    assertTrue(Wording.templates().size() > 100);
  }
}
