// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Instant;
import org.junit.jupiter.api.Test;

class SnowflakeValuesTest {
  @Test
  void snowflakePlanBindsNullWithoutChangingDatabricksScalarRules() throws Exception {
    var statements =
        java.util.List.<Object>of(
            java.util.Map.of(
                "dialect",
                "snowflake",
                "text",
                "SELECT ?",
                "params",
                java.util.List.of(
                    java.util.Map.of("param", "value", "value", java.util.Map.of("t", "null")))));
    var plan =
        JdbcProvider.qmarkPlanned(statements, java.util.Map.of(), JdbcProvider.Driver.SNOWFLAKE);
    assertEquals(1, plan.get(0).values().size());
    assertNull(plan.get(0).values().get(0));
    assertThrows(
        JdbcProvider.PlanNotRun.class,
        () -> JdbcProvider.databricksPlanned(statements, java.util.Map.of()));
  }

  @Test
  void timestampChecksFinalValueRatherThanOverflowingTheIntermediateSeconds() {
    assertEquals(
        Long.MIN_VALUE,
        SnowflakeValues.ticks(Instant.ofEpochSecond(-9223372037L, 145224192), 1_000_000_000L));
    assertEquals(
        Long.MAX_VALUE,
        SnowflakeValues.ticks(Instant.ofEpochSecond(9223372036L, 854775807), 1_000_000_000L));
    assertThrows(
        ArithmeticException.class,
        () ->
            SnowflakeValues.ticks(Instant.ofEpochSecond(-9223372037L, 145224191), 1_000_000_000L));
    assertThrows(
        ArithmeticException.class,
        () -> SnowflakeValues.ticks(Instant.ofEpochSecond(9223372036L, 854775808), 1_000_000_000L));
  }
}
