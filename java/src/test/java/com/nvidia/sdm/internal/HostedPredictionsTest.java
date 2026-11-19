// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.StructuredDataException;
import com.sun.net.httpserver.HttpServer;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The API catalog's hosted relational model is posted to at its exact URL with a bearer token; a
 * base URL keeps {@code /v1/predictions} and {@code X-API-Key}.
 */
class HostedPredictionsTest {
  private static final String CATALOG =
      "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions";
  private static final byte[] OBJECT = "{\"predictions\": []}".getBytes(StandardCharsets.UTF_8);

  /** What one request carried. */
  private record Seen(String method, String path, String bearer, String apiKey) {}

  @Test
  void onlyTheCatalogsPredictionsUrlIsTakenAsItIs() {
    assertTrue(HttpTransport.isHostedPredictions(CATALOG));
    assertTrue(HttpTransport.isHostedPredictions(CATALOG + "/"));
    assertTrue(HttpTransport.isHostedPredictions(CATALOG.replace("ai.api", "AI.API")));
    assertFalse(HttpTransport.isHostedPredictions(CATALOG.replace("https", "http")));
    assertFalse(HttpTransport.isHostedPredictions(CATALOG.replace(".com/", ".com:8443/")));
    assertFalse(HttpTransport.isHostedPredictions(CATALOG + "?x=1"));
    assertFalse(HttpTransport.isHostedPredictions(CATALOG + "/extra"));
    assertFalse(HttpTransport.isHostedPredictions("https://ai.api.nvidia.com"));
    assertFalse(HttpTransport.isHostedPredictions("https://ai.api.nvidia.com/v1/predictions"));
    assertFalse(HttpTransport.isHostedPredictions("http://localhost:8000"));
    assertFalse(HttpTransport.isHostedPredictions(null));
    assertFalse(new HttpTransport(CATALOG, "key", Duration.ofSeconds(5), 0).servesSessions());
    assertTrue(new HttpTransport("https://nim.example", "key", Duration.ofSeconds(5), 0).servesSessions());
  }

  @Test
  void anExactUrlIsPostedToAsItIsWithABearerToken() throws Exception {
    List<Seen> seen = Collections.synchronizedList(new ArrayList<>());
    HttpServer server = serve(seen);
    try {
      String url = "http://127.0.0.1:" + server.getAddress().getPort() + "/hosted/predictions";
      HttpTransport transport = new HttpTransport(url, "secret", Duration.ofSeconds(30), 0, true);
      transport.predict("{}".getBytes(StandardCharsets.UTF_8));
      assertEquals(List.of(new Seen("POST", "/hosted/predictions", "Bearer secret", null)), seen);
      assertFalse(transport.servesSessions());
      assertEquals("SESSIONS_UNSUPPORTED", assertThrows(StructuredDataException.class,
          () -> transport.call("create_session", new byte[0])).code());
      assertEquals("UNSUPPORTED_REQUEST",
          assertThrows(StructuredDataException.class, transport::models).code());
      assertEquals("UNSUPPORTED_REQUEST",
          assertThrows(StructuredDataException.class, transport::healthReady).code());
      assertEquals(1, seen.size());
    } finally {
      server.stop(0);
    }
  }

  @Test
  void aBaseUrlKeepsItsRoutesAndHeader() throws Exception {
    List<Seen> seen = Collections.synchronizedList(new ArrayList<>());
    HttpServer server = serve(seen);
    try {
      String url = "http://127.0.0.1:" + server.getAddress().getPort() + "/";
      HttpTransport transport = new HttpTransport(url, "secret", Duration.ofSeconds(30), 0);
      transport.predict("{}".getBytes(StandardCharsets.UTF_8));
      assertTrue(transport.servesSessions());
      Seen only = seen.get(0);
      assertEquals(new Seen("POST", "/v1/predictions", null, "secret"), only);
      assertNull(only.bearer());
    } finally {
      server.stop(0);
    }
  }

  private static HttpServer serve(List<Seen> seen) throws java.io.IOException {
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext("/", exchange -> {
      try (InputStream in = exchange.getRequestBody()) {
        in.readAllBytes();
      }
      seen.add(new Seen(exchange.getRequestMethod(), exchange.getRequestURI().getPath(),
          exchange.getRequestHeaders().getFirst("Authorization"),
          exchange.getRequestHeaders().getFirst("X-API-Key")));
      exchange.sendResponseHeaders(200, OBJECT.length);
      try (OutputStream out = exchange.getResponseBody()) {
        out.write(OBJECT);
      }
    });
    server.start();
    return server;
  }
}
