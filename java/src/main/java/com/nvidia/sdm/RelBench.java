// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.Json;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.table.Table;

/** Dataset I/O only; the ordinary Schema APIs own admission and sampling. */
final class RelBench {
  private RelBench() {}
  private static final HttpClient HTTP = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).connectTimeout(Duration.ofSeconds(60)).build();

  static Schema load(String name, RelBenchOptions options) {
    try {
      return load(name, options, resource("registry.json"), resource("overrides.json"),
          URI.create("https://huggingface.co/datasets/"));
    } catch (StructuredDataException failure) { throw failure;
    } catch (Exception failure) { throw error("could not read RelBench catalog", "PROVIDER_ERROR", Map.of(), failure); }
  }

  static Map<String, Object> resource(String name) throws Exception {
    try (var input = RelBench.class.getResourceAsStream("/com/nvidia/sdm/relbench/" + name)) {
      if (input == null) throw new IOException("missing packaged RelBench " + name);
      return map(Json.read(input.readAllBytes()));
    }
  }

  static Schema load(String name, RelBenchOptions options, Map<String, Object> registry,
      Map<String, Object> overrides, URI base) {
    if (name == null || options == null) throw error("name and options are required", "INVALID_REQUEST", Map.of(), null);
    Map<String, Object> datasets = map(registry.get("datasets"));
    String dataset = name.toLowerCase(Locale.ROOT);
    if (!datasets.containsKey(dataset)) dataset = "rel-" + dataset;
    if (!datasets.containsKey(dataset)) throw error("unknown RelBench dataset " + name + "; available: " + datasets.keySet(), "INVALID_REQUEST", Map.of(), null);
    Map<String, Object> entry = map(datasets.get(dataset));
    Map<String, Object> specs = map(entry.get("tables"));
    String repository = (String) entry.get("repository");
    String revision = (String) map(registry.get("revisions")).get(repository);
    Map<String, Object> declarations = map(overrides.getOrDefault(dataset, Map.of()));
    Map<String, Table> tables = new LinkedHashMap<>();
    List<Table> readers = new ArrayList<>();
    Schema schema;
    try (RootAllocator allocator = new RootAllocator()) {
      try {
        for (var table : specs.entrySet()) {
          String tableName = table.getKey();
          Map<String, Object> spec = map(table.getValue());
          Path relative = relative(repository, revision, dataset, tableName);
          Path path = options.cacheDirectory().resolve(relative);
          try {
            fetch(path, base.resolve(repository + "/resolve/" + revision + "/" + dataset + "/db/" + tableName + ".parquet"), (String) spec.get("sha256"), options.offline());
            List<?> excluded = (List<?>) map(declarations.getOrDefault(tableName, Map.of())).getOrDefault("exclude", List.of());
            {
              Table raw = Readers.read(path, allocator);
              readers.add(raw); // Keep dictionary owners alive through Schema.fromData snapshots.
              List<FieldVector> kept = new ArrayList<>();
              try {
                for (var field : raw.getSchema().getFields()) if (!excluded.contains(field.getName())) kept.add(raw.getVectorCopy(field.getName()));
                tables.put(tableName, new Table(kept, Math.toIntExact(raw.getRowCount()), raw.getDictionaryProvider()));
              } finally { for (var vector : kept) vector.close(); }
            }
          } catch (Exception failure) {
            if (failure instanceof InterruptedException) Thread.currentThread().interrupt();
            throw error("could not load RelBench " + dataset + "." + tableName + ": " + failure.getMessage(), "PROVIDER_ERROR", Map.of("dataset", dataset, "table", tableName), failure);
          }
        }
        schema = Schema.fromData(tables).withRelationships(List.of());
      } finally {
        for (Table table : tables.values()) table.close();
        for (Table reader : readers) reader.close();
      }
    }
    List<Relationship> links = new ArrayList<>();
    for (var table : specs.entrySet()) {
      String tableName = table.getKey();
      Map<String, Object> spec = map(table.getValue());
      String key = (String) spec.get("primary_key");
      String time = (String) spec.get("time_column");
      schema = key == null ? schema.withoutPrimaryKey(tableName) : schema.withPrimaryKey(tableName, key);
      schema = time == null ? schema.withoutTimeColumn(tableName) : schema.withTimeColumn(tableName, time);
      for (var link : map(spec.get("foreign_keys")).entrySet()) {
        String target = (String) link.getValue();
        links.add(new Relationship(tableName, link.getKey(), target, (String) map(specs.get(target)).get("primary_key")));
      }
    }
    schema = schema.withRelationships(links);
    for (var table : declarations.entrySet()) {
      for (var column : map(map(table.getValue()).getOrDefault("stypes", Map.of())).entrySet()) schema = schema.withStype(table.getKey(), column.getKey(), (String) column.getValue());
    }
    return schema;
  }

  static Path relative(String repository, String revision, String dataset, String table) {
    List<String> parts = new ArrayList<>(List.of(repository.split("/", -1)));
    if (parts.size() != 2) throw error("unsafe RelBench manifest path", "INVALID_REQUEST", Map.of(), null);
    parts.addAll(List.of(revision, dataset, "db", table + ".parquet"));
    for (String part : parts) if (!part.matches("[A-Za-z0-9][A-Za-z0-9_.-]*")) throw error("unsafe RelBench manifest path", "INVALID_REQUEST", Map.of(), null);
    return Path.of(parts.get(0), parts.subList(1, parts.size()).toArray(String[]::new));
  }

  private static String digest(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var input = Files.newInputStream(path)) {
      byte[] buffer = new byte[1024 * 1024];
      for (int count; (count = input.read(buffer)) != -1;) digest.update(buffer, 0, count);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  static void fetch(Path path, URI uri, String expected, boolean offline) throws Exception {
    if (Files.isRegularFile(path) && digest(path).equals(expected)) return;
    if (offline) throw new IOException("offline cache entry is missing or has an invalid SHA-256");
    Files.createDirectories(path.getParent());
    Path temporary = Files.createTempFile(path.getParent(), "." + path.getFileName(), ".part");
    try {
      var request = HttpRequest.newBuilder(uri).timeout(Duration.ofMinutes(10)).GET().build();
      var response = HTTP.send(request, HttpResponse.BodyHandlers.ofFile(temporary));
      if (response.statusCode() != 200) throw new IOException("HTTP " + response.statusCode());
      if (!digest(temporary).equals(expected)) throw new IOException("download SHA-256 does not match the pinned manifest");
      Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } finally { Files.deleteIfExists(temporary); }
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> map(Object value) { return (Map<String, Object>) value; }
  private static StructuredDataException error(String message, String code, Map<String, ?> details, Throwable cause) {
    return new StructuredDataException(message, code, details, cause);
  }
}
