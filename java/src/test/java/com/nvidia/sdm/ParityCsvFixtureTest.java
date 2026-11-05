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

  @org.junit.jupiter.api.Test
  void allExpandedShopFixturesKeepTheirExpectedIdentity() throws Exception {
    Path base = Path.of("../tests/parity");
    var manifest = ParityRunner.read(base.resolve("manifest.json"));
    var hashes = new java.util.LinkedHashMap<String, String>();
    for (Object capability : ParityRunner.list(manifest.get("capabilities"))) {
      for (Object required : ParityRunner.list(ParityRunner.map(capability).get("requirements"))) {
        for (var entry : ParityRunner.map(ParityRunner.map(required).get("fixture_hashes")).entrySet()) {
          String previous = hashes.putIfAbsent(entry.getKey(), entry.getValue().toString());
          if (previous != null) assertEquals(previous, entry.getValue());
        }
      }
    }
    try (var files = java.nio.file.Files.list(base.resolve("fixtures"))) {
      assertEquals(files.filter(path -> path.toString().endsWith(".json")).count(), hashes.size());
    }
    for (var entry : hashes.entrySet())
      assertEquals(entry.getValue(), ParityRunner.digest(ShopFixtures.load(base.resolve(entry.getKey()))), entry.getKey());
  }

  @org.junit.jupiter.api.Test
  void shopVariantsKeepExactValuesNullsAndColumnOrder() throws Exception {
    var fixture = ShopFixtures.load(Path.of("../tests/parity/fixtures/shop-numeric.json"));
    var users = ParityRunner.map(ParityRunner.map(fixture.get("tables")).get("users"));
    assertEquals(java.util.List.of("id", "value", "class", "flag", "narrow", "wide", "fraction", "single"), users.get("columns"));
    var rows = ParityRunner.list(users.get("rows"));
    var first = ParityRunner.list(rows.get(0));
    assertNull(first.get(4));
    assertEquals("9007199254740993", first.get(5).toString());
    assertNull(first.get(6));
    assertEquals("1.123456", ParityRunner.list(rows.get(1)).get(6));
    assertEquals("decimal128(20,6)", ParityRunner.map(users.get("types")).get("fraction"));
    var tags = ShopFixtures.load(Path.of("../tests/parity/fixtures/shop-tags.json"));
    var tagRows = ParityRunner.list(ParityRunner.map(ParityRunner.map(tags.get("tables")).get("users")).get("rows"));
    assertEquals(java.util.List.of("red", "blue"), ParityRunner.list(tagRows.get(0)).get(4));
    assertEquals(java.util.List.of(), ParityRunner.list(tagRows.get(1)).get(4));
    assertNull(ParityRunner.list(tagRows.get(2)).get(4));
    assertNull(ParityRunner.list(ParityRunner.list(tagRows.get(3)).get(4)).get(1));
    var upper = ShopFixtures.load(Path.of("../tests/parity/fixtures/shop-snowflake-semantic.json"));
    assertEquals("ID", ParityRunner.map(upper.get("primary_keys")).get("users"));
    assertEquals(java.util.List.of(java.util.List.of("events.ID", "users.ID"), java.util.List.of("events.ITEM", "items.ITEM")), upper.get("links"));
    first.set(0, -123);
    var clean = ShopFixtures.load(Path.of("../tests/parity/fixtures/shop-numeric.json"));
    var cleanRows = ParityRunner.list(ParityRunner.map(ParityRunner.map(clean.get("tables")).get("users")).get("rows"));
    assertEquals("0", ParityRunner.list(cleanRows.get(0)).get(0).toString());
  }
}
