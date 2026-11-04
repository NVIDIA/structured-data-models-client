// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import com.nvidia.sdm.internal.Json;
import java.nio.file.Files;
import java.nio.file.Path;
import java.net.URI;
import java.util.Map;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RelBenchTest {
  @TempDir Path cache;
  static final Path FIXTURE = Path.of("../tests/fixtures/relbench");
  @SuppressWarnings("unchecked")
  static Map<String, Object> json(String name) throws Exception {
    return (Map<String, Object>) Json.read(Files.readAllBytes(FIXTURE.resolve(name)));
  }
  void seed() throws Exception {
    Path target = cache.resolve("fixture/repo/" + "a".repeat(40) + "/rel-fixture/db");
    Files.createDirectories(target);
    for (String table : List.of("customers", "orders")) Files.copy(FIXTURE.resolve("rel-fixture/db/" + table + ".parquet"), target.resolve(table + ".parquet"));
  }
  @Test void offlineOwnedSchemaPreservesRowsAndDeclarations() throws Exception {
    seed();
    Schema schema = RelBench.load("FIXTURE", RelBenchOptions.builder().cacheDirectory(cache).offline(true).build(), json("registry.json"), json("overrides.json"), URI.create("http://127.0.0.1:1/"));
    assertEquals(List.of("id"), schema.table("customers").primaryKey());
    assertEquals(List.of(), schema.table("orders").primaryKey());
    assertTrue(schema.table("customers").timeColumn().isEmpty());
    assertEquals("time", schema.table("orders").timeColumn().orElseThrow());
    assertEquals(2, schema.table("customers").columns().size());
    assertEquals(1, schema.relationships().size());
    assertEquals("id", schema.relationships().get(0).targetColumn());
    // Snapshot remains accessible after every loader Arrow allocation has closed.
    try (var allocator = new org.apache.arrow.memory.RootAllocator();
        var reader = new org.apache.arrow.vector.ipc.ArrowStreamReader(new java.io.ByteArrayInputStream(schema.data().get("orders")), allocator)) {
      assertTrue(reader.loadNextBatch());
      var rows = reader.getVectorSchemaRoot();
      assertEquals(2, rows.getRowCount());
      assertEquals(2.5d, rows.getVector("amount").getObject(0));
      assertEquals(3.5d, rows.getVector("amount").getObject(1));
      assertEquals(1L, rows.getVector("customer_id").getObject(0));
    }
    var saved = new java.io.ByteArrayOutputStream();
    schema.writeTo(saved);
    assertEquals(schema.relationships(), Schema.readFrom(new java.io.ByteArrayInputStream(saved.toByteArray())).relationships());
  }
  @Test void loadedSchemaComposesReportsExportsAndPersistedRequests() throws Exception {
    seed();
    var options = RelBenchOptions.builder().cacheDirectory(cache).offline(true).build();
    Schema schema = RelBench.load("fixture", options, json("registry.json"), json("overrides.json"), URI.create("http://127.0.0.1:1/"));
    MetadataReport metadata = schema.metadataReport();
    assertEquals(List.of("customers", "orders"), metadata.tables().stream().map(TableMetadata::name).toList());
    assertEquals(List.of("id"), metadata.tables().get(0).primaryKey());
    assertEquals(List.of(), metadata.tables().get(1).primaryKey());
    assertEquals("time", metadata.tables().get(1).timeColumn());
    assertEquals(List.of(new LinkMetadata("orders", "customer_id", "customers", "id")), schema.linkReport().links());
    schema.visualize(cache.resolve("loaded.html"));
    schema.visualize(cache.resolve("loaded.mmd"));
    assertEquals(schema.toHtml(), Files.readString(cache.resolve("loaded.html")));
    assertEquals(schema.toMermaid(), Files.readString(cache.resolve("loaded.mmd")).strip());
    Schema changed = schema.withStype("customers", "name", "categorical");
    assertEquals("text", metadata.tables().get(0).columns().get(1).stype());
    Request request = PredictiveQuery.parse("PREDICT SUM(orders.amount, 0, 1, days) FOR EACH customers.id")
        .materialize(changed, MaterializeOptions.builder().model(SdmClient.KUMO_RELATIONAL)
            .indices(1L).anchorTime(java.time.LocalDate.of(2024, 1, 2)).numContextSamples(1).randomSeed(7).build());
    SanitizationReport report = request.sanitizationReport();
    assertEquals("available", report.status());
    assertEquals(List.of(new TableSanitizationReport("customers", 2, 2, 0, 0, 0),
        new TableSanitizationReport("orders", 2, 2, 0, 0, 0)), report.tables());
    var bytes = new java.io.ByteArrayOutputStream();
    request.writeTo(bytes);
    Request restored = Request.readFrom(new java.io.ByteArrayInputStream(bytes.toByteArray()));
    assertEquals(report, restored.sanitizationReport());
    assertEquals(request.predictRows(), restored.predictRows());
    Schema second = RelBench.load("rel-fixture", options, json("registry.json"), json("overrides.json"), URI.create("http://127.0.0.1:1/"));
    assertEquals("text", second.table("customers").columns().stream().filter(c -> c.name().equals("name")).findFirst().orElseThrow().stype());
  }
  @Test void corruptOfflineCacheAndUnknownNameAreStructured() throws Exception {
    seed();
    Files.writeString(cache.resolve("fixture/repo/" + "a".repeat(40) + "/rel-fixture/db/orders.parquet"), "corrupt");
    var failure = assertThrows(StructuredDataException.class, () -> RelBench.load("fixture", RelBenchOptions.builder().cacheDirectory(cache).offline(true).build(), json("registry.json"), json("overrides.json"), URI.create("http://127.0.0.1:1/")));
    assertEquals("PROVIDER_ERROR", failure.code());
    assertEquals("INVALID_REQUEST", assertThrows(StructuredDataException.class, () -> Schema.fromRelbench("unknown-dataset")).code());
  }
  @Test void registryIsPackagedAndUnsafePathsFail() throws Exception {
    assertEquals(29, ((Map<?, ?>) RelBench.resource("registry.json").get("datasets")).size());
    assertThrows(StructuredDataException.class, () -> RelBench.relative("fixture/repo", "a", "../escape", "table"));
  }
  @org.junit.jupiter.params.ParameterizedTest
  @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
  void httpCacheReuseAndFailedDownloadRecovery(boolean wrongDigest) throws Exception {
    var calls = new java.util.concurrent.atomic.AtomicInteger();
    var broken = new java.util.concurrent.atomic.AtomicBoolean();
    var server = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext("/", exchange -> {
      calls.incrementAndGet();
      String file = Path.of(exchange.getRequestURI().getPath()).getFileName().toString();
      byte[] bytes = Files.readAllBytes(FIXTURE.resolve("rel-fixture/db/" + file));
      exchange.sendResponseHeaders(200, bytes.length);
      try (var output = exchange.getResponseBody()) { output.write(broken.get() && wrongDigest ? new byte[bytes.length] : bytes, 0, broken.get() && !wrongDigest ? 10 : bytes.length); }
    });
    server.start();
    try {
      URI base = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/");
      var options = RelBenchOptions.builder().cacheDirectory(cache).build();
      RelBench.load("fixture", options, json("registry.json"), json("overrides.json"), base);
      assertEquals(2, calls.get());
      RelBench.load("fixture", options, json("registry.json"), json("overrides.json"), base);
      assertEquals(2, calls.get());
      Path orders = cache.resolve("fixture/repo/" + "a".repeat(40) + "/rel-fixture/db/orders.parquet");
      Files.writeString(orders, "invalid cached bytes");
      broken.set(true);
      assertEquals("PROVIDER_ERROR", assertThrows(StructuredDataException.class, () -> RelBench.load("fixture", options, json("registry.json"), json("overrides.json"), base)).code());
      assertEquals("invalid cached bytes", Files.readString(orders));
      try (var paths = Files.list(orders.getParent())) { assertEquals(2, paths.count()); }
      broken.set(false);
      assertEquals(1, RelBench.load("fixture", options, json("registry.json"), json("overrides.json"), base).relationships().size());
    } finally { server.stop(0); }
  }
}
