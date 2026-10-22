// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The host's JSON: what it writes the core reads, and what it refuses, by name. */
class JsonTest {
  private static Object read(String text) throws Json.JsonException {
    return Json.read(text.getBytes(StandardCharsets.UTF_8));
  }

  @Test
  void valuesRoundTrip() throws Exception {
    Map<String, Object> value = new LinkedHashMap<>();
    value.put("text", "a\"b\\c\n\u0000\u001f é \uD83D\uDE00");
    value.put("long", Long.MIN_VALUE);
    value.put("big", new BigInteger("123456789012345678901234567890"));
    value.put("decimal", new BigDecimal("1.5e-7"));
    value.put("list", List.of(true, false));
    value.put("null", null);
    Object back = Json.read(Json.write(value));
    assertEquals(value, back);
  }

  @Test
  void anUnpairedSurrogateIsRefusedNotReplaced() {
    for (String text : new String[] {"\uD83D", "a\uDE00", "\uDE00\uD83D"}) {
      assertThrows(Json.JsonException.class, () -> Json.write(text));
    }
  }

  @Test
  void aNonFiniteNumberIsRefused() {
    assertThrows(Json.JsonException.class, () -> Json.write(Double.NaN));
    assertThrows(Json.JsonException.class, () -> Json.write(Double.POSITIVE_INFINITY));
  }

  @Test
  void thousandsOfNestedLevelsAreRefusedByNameNeverByOverflow() {
    String deep = "[".repeat(100_000) + "]".repeat(100_000);
    Json.JsonException refused = assertThrows(Json.JsonException.class, () -> read(deep));
    assertTrue(refused.getMessage().contains("nesting"), refused.getMessage());
    String objects = "{\"a\":".repeat(5_000) + "1" + "}".repeat(5_000);
    assertThrows(Json.JsonException.class, () -> read(objects));
    List<Object> nested = new ArrayList<>();
    List<Object> at = nested;
    for (int i = 0; i < 10_000; i++) {
      List<Object> inner = new ArrayList<>();
      at.add(inner);
      at = inner;
    }
    assertThrows(Json.JsonException.class, () -> Json.write(nested));
  }

  @Test
  void aVeryWideObjectIsRead() throws Exception {
    StringBuilder wide = new StringBuilder("{");
    for (int i = 0; i < 200_000; i++) {
      wide.append(i == 0 ? "" : ",").append("\"k").append(i).append("\":").append(i);
    }
    wide.append('}');
    assertEquals(200_000, ((Map<?, ?>) read(wide.toString())).size());
  }

  @Test
  void textThatIsNotOneJsonValueIsRefused() {
    for (String text :
        new String[] {"", "{", "[1,]", "{\"a\":1,\"a\":2}", "1 2", "\"\\x\"", "tru", "\"a\nb\""}) {
      assertThrows(Json.JsonException.class, () -> read(text), text);
    }
    assertThrows(
        Json.JsonException.class, () -> Json.read(new byte[] {(byte) 0xff, (byte) 0xfe}));
  }
}
