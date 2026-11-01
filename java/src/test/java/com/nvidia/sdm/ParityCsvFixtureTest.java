// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class ParityCsvFixtureTest {
  @ParameterizedTest
  @ValueSource(strings = {"csv", "zip"})
  void rawDateTextMatchesFrozenPandasFixtureWriter(String connector) throws Exception {
    var fixture = ParityRunner.read(Path.of("../tests/parity/fixtures/shop.json"));
    try (var allocator = new RootAllocator();
        var inputs = new ParityRunner.Inputs()) {
      inputs.open(connector, Map.of(), fixture, allocator);
      var data = ParityRunner.frame(inputs.tables.get("events"));
      int at = ParityRunner.list(data.get("columns")).indexOf("event_time");
      assertEquals(
          "2024-01-01", ParityRunner.list(ParityRunner.list(data.get("rows")).get(0)).get(at));
    }
    var events = ParityRunner.map(ParityRunner.map(fixture.get("tables")).get("events"));
    int at = ParityRunner.list(events.get("columns")).indexOf("event_time");
    var changed =
        new java.util.ArrayList<>(ParityRunner.list(ParityRunner.list(events.get("rows")).get(0)));
    changed.set(at, "2024-01-01T12:34:56");
    var invalid =
        Map.<String, Object>of(
            "tables",
            Map.of(
                "events",
                Map.of("columns", events.get("columns"), "rows", java.util.List.of(changed))));
    try (var allocator = new RootAllocator();
        var inputs = new ParityRunner.Inputs()) {
      assertThrows(
          IllegalStateException.class, () -> inputs.open(connector, Map.of(), invalid, allocator));
    }
  }
}
