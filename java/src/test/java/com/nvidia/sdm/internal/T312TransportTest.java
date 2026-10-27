// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.ModelEndpointException;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * The model transport over {@code java.net.http} (T3.12): the call is sent as the bytes given,
 * one past 16 MB included; a transient status is tried again; a refusal is the endpoint's; an
 * answer that is not JSON, or one past 64 MiB, is a failed transport.
 */
class T312TransportTest {
  /** An endpoint answering {@code statuses} in turn, then 200, keeping every body. */
  private static final class Endpoint implements AutoCloseable {
    final List<byte[]> received = Collections.synchronizedList(new ArrayList<>());
    private final HttpServer server;

    Endpoint(byte[] answer, int... statuses) throws IOException {
      AtomicInteger call = new AtomicInteger();
      server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
      server.createContext(
          "/",
          exchange -> {
            try (InputStream in = exchange.getRequestBody()) {
              received.add(in.readAllBytes());
            }
            int at = call.getAndIncrement();
            int status = at < statuses.length ? statuses[at] : 200;
            byte[] body = status == 200 ? answer : "{\"detail\": \"later\"}".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(status, body.length);
            try (OutputStream out = exchange.getResponseBody()) {
              out.write(body);
            }
          });
      server.start();
    }

    HttpTransport transport(int retries) {
      return new HttpTransport(
          "http://127.0.0.1:" + server.getAddress().getPort(), null, Duration.ofSeconds(30), retries);
    }

    @Override
    public void close() {
      server.stop(0);
    }
  }

  private static final byte[] OBJECT = "{\"predictions\": []}".getBytes(StandardCharsets.UTF_8);

  @Test
  void aCallPast16MiBIsSentAsTheBytesGiven() throws Exception {
    byte[] body = new byte[17 * 1024 * 1024];
    Arrays.fill(body, (byte) ' ');
    body[0] = '{';
    body[body.length - 1] = '}';
    try (Endpoint endpoint = new Endpoint(OBJECT)) {
      assertArrayEquals(OBJECT, endpoint.transport(0).predict(body));
      assertArrayEquals(body, endpoint.received.get(0));
    }
  }

  @Test
  void aTransientStatusIsTriedAgain() throws Exception {
    try (Endpoint endpoint = new Endpoint(OBJECT, 503, 429)) {
      assertArrayEquals(OBJECT, endpoint.transport(3).predict("{}".getBytes(StandardCharsets.UTF_8)));
      assertEquals(3, endpoint.received.size());
    }
  }

  @Test
  void aRefusalIsTheEndpointsAndIsNotTriedAgain() throws Exception {
    try (Endpoint endpoint = new Endpoint(OBJECT, 500)) {
      ModelEndpointException refused =
          assertThrows(
              ModelEndpointException.class,
              () -> endpoint.transport(3).predict("{}".getBytes(StandardCharsets.UTF_8)));
      assertEquals(500, refused.status());
      assertEquals("later", refused.getMessage());
      assertEquals(1, endpoint.received.size());
    }
  }

  @Test
  void anAnswerThatIsNoJsonObjectIsAFailedTransport() throws Exception {
    for (String answer : List.of("not json", "[1, 2]")) {
      try (Endpoint endpoint = new Endpoint(answer.getBytes(StandardCharsets.UTF_8))) {
        assertThrows(
            HttpTransport.Failed.class,
            () -> endpoint.transport(0).predict("{}".getBytes(StandardCharsets.UTF_8)));
      }
    }
  }

  @Test
  void anEmptyAnswerIsAnEmptyObject() throws Exception {
    try (Endpoint endpoint = new Endpoint(new byte[0])) {
      assertArrayEquals(
          "{}".getBytes(StandardCharsets.UTF_8),
          endpoint.transport(0).predict("{}".getBytes(StandardCharsets.UTF_8)));
    }
  }

  /**
   * Codex T3.12 r1 P1-6: the timeout bounds the whole attempt, the answer's body included: an
   * endpoint that sends its headers and then holds its body back, or never ends it, is a timed-out
   * transport within the timeout, its exchange given up.
   */
  @Test
  void anAnswerWhoseBodyIsHeldBackTimesOut() throws Exception {
    AtomicBoolean closed = new AtomicBoolean();
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.setExecutor(java.util.concurrent.Executors.newCachedThreadPool(runnable -> {
      Thread thread = new Thread(runnable);
      thread.setDaemon(true);
      return thread;
    }));
    server.createContext(
        "/",
        exchange -> {
          try (InputStream in = exchange.getRequestBody()) {
            in.readAllBytes();
          }
          exchange.sendResponseHeaders(200, 2);
          OutputStream out = exchange.getResponseBody();
          out.write('{');
          out.flush();
          // The rest of the body never comes while the test runs.
          for (int waited = 0; waited < 300 && !closed.get(); waited++) {
            try {
              Thread.sleep(100);
            } catch (InterruptedException e) {
              return;
            }
          }
        });
    server.start();
    try {
      HttpTransport transport =
          new HttpTransport(
              "http://127.0.0.1:" + server.getAddress().getPort(), null, Duration.ofMillis(300), 0);
      long started = System.nanoTime();
      HttpTransport.Failed failed =
          assertThrows(
              HttpTransport.Failed.class,
              () -> transport.predict("{}".getBytes(StandardCharsets.UTF_8)));
      long millis = (System.nanoTime() - started) / 1_000_000;
      assertTrue(millis < 5_000, "the call took " + millis + " ms");
      assertTrue(failed.getMessage().contains("timed out"), failed.getMessage());
    } finally {
      closed.set(true);
      server.stop(0);
    }
  }
}
