// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.lang.reflect.Proxy;
import java.sql.Array;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/** Zoned text is a stable UTC feature, independent of a caller's session or JVM zone. */
class T39PostgresUtcTextTest {
  private static Object read(int oid, Object value) throws SQLException {
    ResultSet rows = (ResultSet) Proxy.newProxyInstance(
        ResultSet.class.getClassLoader(), new Class<?>[] {ResultSet.class},
        (proxy, method, args) -> {
          if (method.getName().equals("getString") || method.getName().equals("getArray")) {
            return value;
          }
          throw new UnsupportedOperationException(method.getName());
        });
    ConversionTable.Session session = new ConversionTable.Session() {
      @Override
      public String intervalStyle() {
        return "postgres";
      }

      public int offsetSeconds(long micros) {
        throw new AssertionError("canonical UTC text must not read a session timezone");
      }
    };
    return ConversionTable.REVIEWED.postgres(oid, 0, 0).read(rows, 1, session);
  }

  private static Object readArray(String... values) throws SQLException {
    int[] at = {-1};
    ResultSet elements = (ResultSet) Proxy.newProxyInstance(
        ResultSet.class.getClassLoader(), new Class<?>[] {ResultSet.class},
        (proxy, method, args) -> switch (method.getName()) {
          case "next" -> ++at[0] < values.length;
          case "getObject", "getString" -> values[at[0]];
          case "close" -> null;
          default -> throw new UnsupportedOperationException(method.getName());
        });
    Array array = (Array) Proxy.newProxyInstance(
        Array.class.getClassLoader(), new Class<?>[] {Array.class},
        (proxy, method, args) -> switch (method.getName()) {
          case "getResultSet" -> elements;
          case "free" -> null;
          default -> throw new UnsupportedOperationException(method.getName());
        });
    return read(1185, array);
  }

  @Test
  void zonedRangesAndArrayElementsCrossDstInUtc() throws Exception {
    String lower = "2024-03-10 01:30:00.123456-08";
    String upper = "2024-03-10 03:30:00.654321-07";
    String range = "[\"" + lower + "\",\"" + upper + "\")";
    String expected =
        "[2024-03-10 09:30:00.123456+00:00, 2024-03-10 10:30:00.654321+00:00)";
    assertEquals(expected, read(3910, range));
    assertEquals("{" + expected + "}", read(4534, "{" + range + "}"));
    assertEquals(
        Arrays.asList("2024-03-10 09:30:00.123456+00:00", null),
        readArray(lower, null));
    assertEquals("empty", read(3910, "empty"));
    assertEquals("{}", read(4534, "{}"));
    assertNull(read(3910, null));
  }

  @Test
  void aReadableWallDateOutsidePythonsUtcYearsKeepsItsInstant() throws Exception {
    assertEquals(
        Arrays.asList("0000-12-31 15:00:00+00:00", "10000-01-01 02:00:00+00:00"),
        readArray("0001-01-01 05:00:00+14", "9999-12-31 23:00:00-03"));
    assertEquals(
        "[0000-12-31 15:00:00+00:00, None)",
        read(3910, "[\"0001-01-01 05:00:00+14\",)"));
  }

  @Test
  void timezoneFreeBoundsKeepTheirWallFields() throws Exception {
    assertEquals(
        "[2024-03-10 02:30:00, 2024-03-10 03:00:00)",
        read(3908, "[\"2024-03-10 02:30:00\",\"2024-03-10 03:00:00\")"));
  }
}
