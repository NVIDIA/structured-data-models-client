// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigInteger;
import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * The Java host says the core's sentences from the shared templates (T3.17), choosing none: every
 * sentence validation (S0) can reach is filled, with no advice to spell, and one this host cannot
 * fill yet is the defect, named.
 */
class WordingTest {
  @Test
  void timestampUnitsUseTheirExactCountsIncludingNegativeFractions() {
    String[] units = {"s", "ms", "us", "ns"};
    int[] nanos = {0, 999_000_000, 999_999_000, 999_999_999};
    for (int i = 0; i < units.length; i++) {
      Object value = Wording.scalar(Map.of("t", "ts", "v", "-1", "unit", units[i]));
      assertEquals(LocalDateTime.of(1969, 12, 31, 23, 59, 59, nanos[i]), value);
    }
    assertEquals(LocalDateTime.of(2024, 1, 1, 0, 0),
        Wording.scalar(Map.of("t", "ts", "v", "1704067200", "unit", "s")));
  }

  @Test
  void unknownTimestampAndCoarseNonTimestampUnitsCannotMasqueradeAsNanoseconds() {
    for (String unit : List.of("", "h", "MS", "US")) {
      assertThrows(IllegalArgumentException.class,
          () -> Wording.scalar(Map.of("t", "ts", "v", "1", "unit", unit)));
    }
    for (String tag : List.of("duration", "time64")) {
      for (String unit : List.of("s", "ms")) {
        assertThrows(IllegalArgumentException.class,
            () -> Wording.scalar(Map.of("t", tag, "v", "1", "unit", unit)));
      }
    }
  }

  /**
   * Every sentence a {@code validate_query} run can end with or warn with: the start's checks
   * ({@code sdm_core::dispatch::start}, {@code attached}, {@code sdm_host::runs}), the validation's
   * own ({@code sdm_core::pql}: a packed date literal, the local zone and the clock, the day-first
   * warning), and the run's protocol errors and a contained panic ({@code sdm_core::run}, {@code
   * sdm_host::contain}).
   */
  static final List<String> S0 =
      List.of(
          "internal",
          "invalid_message",
          "host_clock_not_utc",
          "local_zone_unread",
          "clock_past_years",
          "date_literal_packed",
          "date_literal_day_first",
          "run_busy",
          "run_finished",
          "run_cancelled",
          "run_poisoned",
          "reply_missing",
          "wrong_reply",
          "unknown_reply");

  private static final Pattern PLACEHOLDER = Pattern.compile("(?<!\\{)\\{([^{}]+)\\}");

  /** A value of the shape each placeholder of {@code template} asks for. */
  private static Map<String, Object> fieldsFor(String template) {
    Map<String, Object> fields = new LinkedHashMap<>();
    Matcher found = PLACEHOLDER.matcher(template);
    while (found.find()) {
      String spec = found.group(1);
      String name = spec.split("[\\[!:]", 2)[0];
      boolean list = spec.contains("[") || spec.matches(".*![lajcRd].*");
      fields.put(name, list ? List.of("x", "y") : spec.endsWith(":,") ? (Object) 1234L : "x");
    }
    return fields;
  }

  @Test
  void everySentenceS0CanReachIsSaidAndNamesNoAdviceToSpell() {
    Map<String, String> table = Wording.templates();
    for (String sentence : S0) {
      String template = table.get(sentence);
      assertTrue(template != null, sentence + " is not in the core's table");
      // Every Advice S0 can reach has a Java spelling: it reaches none.
      assertFalse(template.contains("{advice:"), sentence + " names advice");
      assertFalse(template.contains("{host."), sentence + " names a host-held value");
      String said = Wording.say(sentence, fieldsFor(template));
      assertFalse(said.startsWith("the core said"), sentence + ": " + said);
      assertFalse(said.contains("{"), sentence + ": " + said);
    }
  }

  @Test
  void aPackedDateLiteralIsSaidWithTheQueryAsAJavaLiteral() {
    assertEquals(
        "\"PREDICT \\\"x\\\"\" is not a query a relational model can answer: 20240101/00 is not"
            + " a valid instance of time",
        Wording.say(
            "date_literal_packed", Map.of("query", "PREDICT \"x\"", "literal", "20240101/00")));
  }

  @Test
  void aSentenceThisHostCannotFillIsTheDefectNamed() {
    // Advice is spelled by the Java host from T3.12: a sentence naming it is filled.
    assertEquals(
        "t.c is the table's primary key; set that to something else, or null, before leaving"
            + " the column out",
        Wording.say("left_out_key", Map.of("table", "t", "column", "c")));
    // A field the template names and the error does not carry.
    assertEquals(
        String.format(Wording.DEFECT, "\"no_table\""), Wording.say("no_table", Map.of()));
    assertEquals(
        String.format(Wording.DEFECT, "\"no_such_sentence\""),
        Wording.say("no_such_sentence", Map.of()));
  }

  @Test
  void thePartIsSaidAfterTheErrorsOwnSentence() {
    Map<String, Object> error = new LinkedHashMap<>();
    error.put("sentence", "run_busy");
    error.put("part", "part_forecast_step");
    error.put(
        "fields",
        Map.of(
            "forecast_step", Map.of("t", "u64", "v", "2"),
            "forecast_steps", Map.of("t", "u64", "v", "3")));
    assertEquals(
        "the run was already in a call; a run takes one call at a time. This is forecast step 2"
            + " of 3.",
        Wording.said(error));
  }

  @Test
  void valuesAreWrittenAsJava() throws Wording.Unfilled {
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("text", "a\"b\\c\n\u0001\ud800");
    fields.put("names", List.of("a", "b", "c"));
    fields.put("two", List.of("a", "b"));
    fields.put("count", new BigInteger("12345678901234567890"));
    fields.put("small", 1234L);
    fields.put("flag", true);
    fields.put("none", null);
    assertEquals(
        "\"a\\\"b\\\\c\\n\\u0001\\ud800\"", Wording.fill("{text!r}", fields));
    assertEquals("\"a\", \"b\" and \"c\"", Wording.fill("{names!l}", fields));
    assertEquals("a and b", Wording.fill("{two!a}", fields));
    assertEquals("\"a\", \"b\", \"c\"|a, b, c|a; b; c|a.b.c",
        Wording.fill("{names!j}|{names!c}|{names!R}|{names!d}", fields));
    assertEquals("\"a\" a [\"a\", \"b\"]", Wording.fill("{names[0]!r} {names[0]} {two}", fields));
    assertEquals("12,345,678,901,234,567,890 1,234", Wording.fill("{count:,} {small:,}", fields));
    assertEquals("true null {x}", Wording.fill("{flag} {none} {{x}}", fields));
    assertEquals("\"\ud83d\ude00\"", Wording.fill("{x!r}", Map.of("x", "\ud83d\ude00")));
  }
}
