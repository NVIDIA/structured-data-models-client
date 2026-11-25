// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.Json;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Host-only schema presentation. No execution or schema decisions. */
final class SchemaVisualization {
  private SchemaVisualization() {}

  private static String label(String value) {
    StringBuilder result = new StringBuilder();
    value.codePoints().forEach(c -> result.append('#').append(c).append(';'));
    return result.toString();
  }

  static String mermaid(Schema schema, boolean showColumns) {
    Map<String, String> identifiers = new LinkedHashMap<>();
    for (Table table : schema.tables()) identifiers.put(table.name(), "t" + identifiers.size());
    List<String> lines = new ArrayList<>();
    lines.add("flowchart LR");
    if (identifiers.isEmpty()) lines.add("  empty[\"Empty schema\"]");
    for (Table table : schema.tables()) {
      List<String> labels = new ArrayList<>();
      labels.add(label(table.name()));
      for (Column column : table.columns()) {
        List<String> marks = new ArrayList<>();
        if (table.primaryKey().contains(column.name())) marks.add("PK");
        if (schema.relationships().stream().anyMatch(r -> r.sourceTable().equals(table.name())
            && r.sourceColumn().equals(column.name()))) marks.add("FK");
        if (table.timeColumn().filter(column.name()::equals).isPresent()) marks.add("TIME");
        if (showColumns || !marks.isEmpty()) {
          String type = column.stype() != null ? column.stype()
              : column.dtype() != null ? column.dtype() : "unknown";
          String description = column.name() + ": " + type;
          if (!marks.isEmpty()) description += " [" + String.join(", ", marks) + "]";
          labels.add(label(description));
        }
      }
      lines.add("  " + identifiers.get(table.name()) + "[\"" + String.join("<br/>", labels) + "\"]");
    }
    for (Relationship link : schema.relationships()) {
      String description = link.sourceColumn() + " → " + link.targetColumn() + " (many to one)";
      lines.add("  " + identifiers.get(link.sourceTable()) + " -->|\"" + label(description)
          + "\"| " + identifiers.get(link.targetTable()));
    }
    return String.join("\n", lines);
  }

  private static String resource(String name) {
    try (InputStream input = SchemaVisualization.class.getResourceAsStream("visualization/" + name)) {
      if (input == null) throw new IllegalStateException("missing visualization resource: " + name);
      return new String(input.readAllBytes(), StandardCharsets.UTF_8);
    } catch (IOException error) {
      throw new IllegalStateException("cannot read visualization resource", error);
    }
  }

  private static final class Assets {
    static final String TEMPLATE = resource("schema.html");
    static final String RENDERER = resource("mermaid.min.js");
  }

  static String html(String source) {
    String escaped = source.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\"", "&quot;").replace("'", "&#x27;");
    return Assets.TEMPLATE.replace("@@RENDERER@@", Assets.RENDERER).replace("@@SOURCE@@", escaped);
  }

  static void export(Schema schema, Path path, boolean showColumns) throws IOException {
    if (path == null || path.getFileName() == null)
      throw new IllegalArgumentException("a file path is required");
    String name = path.getFileName().toString();
    int dot = name.lastIndexOf('.');
    String suffix = dot < 0 ? "" : name.substring(dot).toLowerCase(Locale.ROOT);
    String source = mermaid(schema, showColumns);
    switch (suffix) {
      case ".mmd" -> Files.writeString(path, source + "\n", StandardCharsets.UTF_8);
      case ".html" -> Files.writeString(path, html(source), StandardCharsets.UTF_8);
      case ".png", ".svg" -> Files.write(path, image(source, suffix));
      default -> throw new IllegalArgumentException("use an .mmd, .html, .png or .svg path");
    }
  }

  static byte[] image(String source, String suffix) throws IOException {
    byte[] json;
    try {
      json = Json.write(Map.of("code", source,
          "mermaid", Map.of("theme", "neutral", "securityLevel", "strict")));
    } catch (Json.JsonException error) {
      throw new IOException("cannot encode schema diagram", error);
    }
    String encoded = Base64.getUrlEncoder().encodeToString(json);
    String route = suffix.equals(".svg") ? "svg" : "img";
    HttpRequest request = HttpRequest.newBuilder(URI.create("https://mermaid.ink/" + route + "/base64:" + encoded
        + (suffix.equals(".png") ? "?type=png" : "")))
        .timeout(Duration.ofSeconds(30)).GET().build();
    try {
      HttpResponse<byte[]> response = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30))
          .build().send(request, HttpResponse.BodyHandlers.ofByteArray());
      if (response.statusCode() != 200) throw new IOException("renderer returned HTTP " + response.statusCode());
      return response.body();
    } catch (InterruptedException error) {
      Thread.currentThread().interrupt();
      throw new IOException("schema image export interrupted", error);
    } catch (IOException error) {
      throw new IOException("schema image export failed at the external renderer ("
          + error.getClass().getSimpleName() + ")");
    }
  }
}
