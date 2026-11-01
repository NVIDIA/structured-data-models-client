// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;

class ParityHarnessTest {
  @Test
  void fixtureVerificationRejectsChangedValuesAndAllowsDriverRowOrder() throws Exception {
    Map<String, Object> declared =
        Map.of(
            "columns", List.of("id", "value"), "rows", List.of(List.of(1L, 2.0), List.of(2L, 3.0)));
    Map<String, Object> fixture =
        Map.of("tables", Map.of("users", declared), "primary_keys", Map.of("users", "id"));
    try (var allocator = new RootAllocator();
        var reordered =
            ParityRunner.table(
                Map.of(
                    "columns",
                    List.of("id", "value"),
                    "rows",
                    List.of(List.of(2L, 3.0), List.of(1L, 2.0))),
                allocator);
        var changed =
            ParityRunner.table(
                Map.of(
                    "columns",
                    List.of("id", "value"),
                    "rows",
                    List.of(List.of(1L, 2.0), List.of(2L, 4.0))),
                allocator)) {
      assertDoesNotThrow(() -> ParityRunner.verify(Map.of("users", reordered), fixture));
      assertThrows(
          ParityRunner.Incomplete.class,
          () -> ParityRunner.verify(Map.of("users", changed), fixture));
    }
  }

  @Test
  void canonicalFixtureDigestMatchesPythonSortedCompactJson() throws Exception {
    assertEquals(
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777",
        ParityRunner.digest(Map.of("b", 2L, "a", 1L)));
  }

  @Test
  void routeCannotPassWithoutNativePhaseWitnesses() throws Exception {
    var record = new java.util.LinkedHashMap<String, Object>();
    record.put("status", "passed");
    record.put("semantic", Map.of("lifecycle", List.of("materialize")));
    try (var route = new ParityRoute()) {
      route.begin(record);
      route.end(record);
      route.finish(List.of(record));
    }
    assertEquals("incomplete", record.get("status"));
    assertEquals("native operation witness missing for a required phase", record.get("reason"));
  }

  @Test
  void expectedOutcomesDoNotAllowArbitraryRefusals() {
    var error = new java.util.LinkedHashMap<String, Object>();
    error.put("stage", "restore");
    error.put("code", "UNSUPPORTED_REQUEST");
    error.put("rule", null);
    Map<String, Object> record = Map.of("status", "refused", "error", error);
    assertFalse(ParityRunner.accepted(record, Map.of("status", "supported")));
    var expected = new java.util.LinkedHashMap<>(error);
    expected.put("status", "refused");
    assertTrue(ParityRunner.accepted(record, expected));
    expected.put("stage", "predict");
    assertFalse(ParityRunner.accepted(record, expected));
  }
}
