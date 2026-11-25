// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SchemaVisualizationTest {
  @TempDir Path directory;

  @Test void emptySchemaExportsWithoutNativeOrNetwork() throws Exception {
    Schema schema = Schema.held(Map.of("tables", List.of(), "links", List.of()), null);
    assertEquals("flowchart LR\n  empty[\"Empty schema\"]", schema.toMermaid());
    String html = schema.toHtml();
    assertTrue(html.startsWith("<!doctype html>"));
    assertTrue(html.contains("securityLevel:\"strict\""));
    assertFalse(html.contains("<script src="));
    Path file = directory.resolve("schema.html");
    schema.visualize(file);
    assertEquals(html, Files.readString(file));
    schema.visualize(directory.resolve("schema.mmd"), false);
    assertEquals(schema.toMermaid(false) + "\n", Files.readString(directory.resolve("schema.mmd")));
    assertThrows(IllegalArgumentException.class, () -> schema.visualize(directory.resolve("schema.exe")));
    assertThrows(IllegalArgumentException.class, () -> schema.visualize(null));
  }
  @Test @SuppressWarnings("unchecked")
  void sharedMetadataRetainsKeysAndEscapesNames() throws Exception {
    Map<String, Object> value = (Map<String, Object>) com.nvidia.sdm.internal.Json.read(
        Files.readAllBytes(Path.of("../tests/fixtures/visualization/schema.json")));
    List<Object> links = new java.util.ArrayList<>();
    for (Object entry : (List<Object>) value.get("links")) {
      Map<String, Object> link = (Map<String, Object>) entry;
      links.add(List.of(Map.of("table", link.get("source_table"), "column", link.get("source_column")),
          Map.of("table", link.get("target_table"), "column", link.get("target_column"))));
    }
    value = new java.util.LinkedHashMap<>(value);
    value.put("links", links);
    Schema schema = Schema.held(value, null);
    String source = schema.toMermaid();
    String decoded = java.util.regex.Pattern.compile("#(\\d+);").matcher(source)
        .replaceAll(m -> java.util.regex.Matcher.quoteReplacement(
            new String(Character.toChars(Integer.parseInt(m.group(1))))));
    assertTrue(decoded.contains("tenant: ID [PK]"));
    assertTrue(decoded.contains("id: ID [PK]"));
    assertTrue(decoded.contains("customer_id: ID [FK]"));
    assertTrue(decoded.contains("created: timestamp [TIME]"));
    assertTrue(decoded.contains("customer_id → id (many to one)"));
    assertFalse(source.contains("</script>"));
    assertFalse(source.contains("attacker"));
    assertEquals(source, schema.toMermaid());
    Path output = Path.of("target/visualization-fixture.html");
    schema.visualize(output);
    Files.writeString(Path.of("target/visualization-fixture.mmd"), source);
  }

  @Test void packagedRendererMatchesPinnedUpstream() throws Exception {
    try (var input = Schema.class.getResourceAsStream("visualization/mermaid.min.js")) {
      assertNotNull(input);
      assertEquals("70137e77bb273bb2ef972b86e8b0400cca8be53cb25bfc45911a186dc98665de",
          java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256")
              .digest(input.readAllBytes())));
    }
    try (var input = Schema.class.getResourceAsStream("visualization/THIRD-PARTY-NOTICES-MERMAID.txt")) {
      assertNotNull(input);
      assertTrue(new String(input.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8).contains("MIT License"));
    }
  }
}
