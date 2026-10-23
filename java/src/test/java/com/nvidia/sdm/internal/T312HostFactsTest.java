// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * T3.12: the local zone a run's facts name is read by the core's one rule ({@code
 * sdm_host::local_zone}), which the Python host reads too, and the Java host holds no copy of it
 * (T3.11 review, P2 2); the model capabilities come from the core's table.
 */
class T312HostFactsTest {
  private static List<Map<String, Object>> roots(Path... paths) {
    List<Map<String, Object>> roots = new ArrayList<>();
    for (Path path : paths) {
      Map<String, Object> root = new LinkedHashMap<>();
      root.put("path", path.toString());
      root.put("version", null);
      roots.add(root);
    }
    return roots;
  }

  @Test
  void theLocalZoneIsTheCoresAnswer(@TempDir Path root) throws Exception {
    Files.createDirectories(root.resolve("Europe"));
    Files.write(root.resolve("Europe/Berlin"), "TZif".getBytes(StandardCharsets.US_ASCII));
    List<Map<String, Object>> roots = roots(root);
    List<String> paths = List.of(root.toString());
    String[] settings = {
      null, "Europe/Berlin", ":Europe/Berlin", "utc", "EST5EDT,M3.2.0,M11.1.0", "Mars/Olympus",
      "../Europe/Berlin", ""
    };
    for (String setting : settings) {
      assertEquals(
          NativeCore.localZone(setting, paths), HostFacts.localZone(roots, setting), setting);
    }
    assertEquals(Map.of("iana", "Europe/Berlin"), HostFacts.localZone(roots, ":Europe/Berlin"));
    assertEquals(Map.of("iana", "utc"), HostFacts.localZone(roots, "utc"));
    assertEquals(
        Map.of("posix", "EST5EDT,M3.2.0,M11.1.0"),
        HostFacts.localZone(roots, "EST5EDT,M3.2.0,M11.1.0"));
    assertNull(HostFacts.localZone(roots, "Mars/Olympus"));
    assertNull(HostFacts.localZone(roots, "../Europe/Berlin"));
    Map<String, Object> unset = HostFacts.localZone(roots, null);
    assertTrue(unset == null || unset.keySet().equals(java.util.Set.of("iana")), "" + unset);
  }

  @Test
  void aRunsFactsNameTheZoneByTheSameRule() {
    Map<String, Object> facts = HostFacts.now();
    @SuppressWarnings("unchecked")
    List<Map<String, Object>> roots = (List<Map<String, Object>>) facts.get("zone_roots");
    assertEquals(HostFacts.localZone(roots, System.getenv("TZ")), facts.get("local_zone"));
  }

  @Test
  void theHostHoldsNoCopyOfTheZoneRule() throws Exception {
    // Run from java/ (Maven's base directory).
    String source =
        Files.readString(
            Path.of("src/main/java/com/nvidia/sdm/internal/HostFacts.java"),
            StandardCharsets.UTF_8);
    assertFalse(source.contains("/etc/localtime"));
    assertFalse(source.contains("\"zoneinfo\""));
    assertFalse(source.contains("\"..\""));
    assertFalse(source.contains("isPosixZone"));
    assertFalse(source.contains("toRealPath"));
  }

  @Test
  void theCapabilitiesAreTheCoresTable() throws Exception {
    @SuppressWarnings("unchecked")
    Map<String, Object> read = (Map<String, Object>) Json.read(NativeCore.capabilities());
    @SuppressWarnings("unchecked")
    List<Map<String, Object>> models = (List<Map<String, Object>>) read.get("models");
    assertEquals("kumo-relational", models.get(0).get("model"));
    assertEquals("kumo-tabular", models.get(1).get("model"));
    for (Map<String, Object> model : models) {
      assertEquals("sdk", model.get("source"));
      assertEquals(Boolean.TRUE, model.get("single_table"));
    }
  }
}
