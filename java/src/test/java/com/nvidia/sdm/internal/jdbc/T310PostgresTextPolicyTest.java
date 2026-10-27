// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.reflect.Proxy;
import java.sql.Connection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class T310PostgresTextPolicyTest {
  private static final List<Map<String, Object>> UTC =
      List.of(Map.of("conversion", "zoned_text_utc"));

  private static Connection noIo() {
    return (Connection) Proxy.newProxyInstance(
        Connection.class.getClassLoader(), new Class<?>[] {Connection.class},
        (proxy, method, arguments) -> { throw new AssertionError("driver I/O: " + method.getName()); });
  }

  private static Map<String, Object> step(String dialect, Object conversions) {
    Map<String, Object> plan = new LinkedHashMap<>();
    plan.put("plan", "sql");
    plan.put("statements", List.of(Map.of("dialect", dialect, "text", "SELECT 1", "params", List.of())));
    if (conversions != null) {
      plan.put("result_conversions", conversions);
    }
    return Map.of("plan", plan);
  }

  @Test
  void postgresRefusesMissingUnknownExtraAndDuplicatePolicyBeforeIo() {
    JdbcProvider provider = new JdbcProvider(noIo(), JdbcProvider.Driver.POSTGRES);
    for (Object conversions : new Object[] {
        null, List.of(), List.of(Map.of("conversion", "unknown")),
        List.of(Map.of("conversion", "zoned_text_utc", "column", 0)), List.of(UTC.get(0), UTC.get(0))
    }) {
      assertThrows(JdbcProvider.PlanNotRun.class,
          () -> provider.answer(step("postgres", conversions), Map.of(), null));
    }
    assertThrows(JdbcProvider.PlanNotRun.class,
        () -> provider.answer(step("sqlite", UTC), Map.of(), null));
  }

  @Test
  void noOtherDriverAcceptsThePostgresPolicyBeforeIo() {
    for (JdbcProvider.Driver driver : List.of(
        JdbcProvider.Driver.DUCKDB, JdbcProvider.Driver.SQLITE, JdbcProvider.Driver.DATABRICKS)) {
      JdbcProvider provider = new JdbcProvider(noIo(), driver);
      String dialect = driver == JdbcProvider.Driver.DATABRICKS ? "databricks" : driver.name().toLowerCase(java.util.Locale.ROOT);
      assertThrows(JdbcProvider.PlanNotRun.class,
          () -> provider.answer(step(dialect, UTC), Map.of(), null));
    }
  }
}
