// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.StructuredDataException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** A run's one owner, its release, and the core's errors, on the test-only operations. */
class RunTest {
  @BeforeEach
  void testOps() {
    TestOps.required();
  }

  private static Map<String, Object> next(Run run) throws NativeError {
    try (NativeMessage step = run.next()) {
      return Messages.read(step);
    }
  }

  @Test
  void aRunClosedTwiceIsReleasedOnce() throws Exception {
    Run run = Run.start(TestOps.start("log", Map.of("hold_ms", 0)), Map.of());
    run.close();
    run.close();
    assertThrows(IllegalStateException.class, run::next);
    assertThrows(IllegalStateException.class, run::cancel);
  }

  @Test
  void aRunNeverClosedIsReleasedByTheCleaner() throws Exception {
    for (int i = 0; i < 200; i++) {
      Run.start(TestOps.start("log", Map.of("hold_ms", 0)), Map.of());
    }
    System.gc();
    // Nothing to assert in process but that the JVM is well: the cleaner
    // releases each handle once, on its own thread, holding the write lock.
    Run run = Run.start(TestOps.start("log", Map.of("hold_ms", 0)), Map.of());
    assertEquals("done", next(run).get("step"));
    run.close();
  }

  @Test
  void aSecondCallWhileOneIsInTheRunIsBusy() throws Exception {
    try (Run run = Run.start(TestOps.start("log", Map.of("hold_ms", 1500)), Map.of())) {
      CompletableFuture<Map<String, Object>> first =
          CompletableFuture.supplyAsync(
              () -> {
                try {
                  return next(run);
                } catch (NativeError e) {
                  throw new IllegalStateException(e);
                }
              });
      Thread.sleep(300);
      NativeError busy = assertThrows(NativeError.class, run::next);
      StructuredDataException raised = busy;
      assertEquals("RUN_BUSY", raised.code());
      assertEquals("protocol", raised.kind());
      assertEquals("done", first.get(30, TimeUnit.SECONDS).get("step"));
    }
  }

  @Test
  void closeWaitsForTheCallInFlight() throws Exception {
    Run run = Run.start(TestOps.start("log", Map.of("hold_ms", 800)), Map.of());
    CompletableFuture<Map<String, Object>> first =
        CompletableFuture.supplyAsync(
            () -> {
              try {
                return next(run);
              } catch (NativeError e) {
                throw new IllegalStateException(e);
              }
            });
    Thread.sleep(200);
    run.close();
    assertEquals("done", first.get(30, TimeUnit.SECONDS).get("step"));
    assertThrows(IllegalStateException.class, run::next);
  }

  @Test
  void anErrorDuringNextCarriesItsCodeAndFields() throws Exception {
    Map<String, Object> error = new LinkedHashMap<>();
    error.put("kind", "invalid_request");
    error.put("code", "INVALID_REQUEST");
    error.put("sentence", "no_table");
    error.put("message", "a test-only failure");
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("field", Map.of("t", "utf8", "v", "indices"));
    fields.put("table", Map.of("t", "utf8", "v", "orders"));
    fields.put(
        "tables", List.of(Map.of("t", "utf8", "v", "users"), Map.of("t", "utf8", "v", "a\"b")));
    error.put("fields", fields);
    try (Run run =
        Run.start(
            TestOps.start("fail", Map.of("error", 1)),
            Map.of(1, TestOps.json(error)))) {
      NativeError failed = assertThrows(NativeError.class, run::next);
      StructuredDataException raised = failed;
      assertEquals("INVALID_REQUEST", raised.code());
      assertEquals("invalid_request", raised.kind());
      assertEquals("indices", raised.details().get("field"));
      // The core's sentence, filled as Java writes the values: never its free text.
      assertEquals(
          "no table \"orders\" in this schema; it has [\"users\", \"a\\\"b\"]",
          raised.getMessage());
    }
  }

  @Test
  void anErrorDuringReplyAndCancelReachesJava() throws Exception {
    try (Run run =
        Run.start(
            TestOps.start("fetch", Map.of("fetch", 1)),
            Map.of(1, TestOps.json(TestOps.fetch(false))))) {
      assertEquals("fetch", next(run).get("step"));
      // A reply naming an attachment it does not carry: INVALID_MESSAGE.
      NativeError refused =
          assertThrows(
              NativeError.class, () -> run.reply(TestOps.fetchReply(1), Map.of()));
      assertEquals("INVALID_MESSAGE", refused.code());
      run.cancel();
      NativeError cancelled = assertThrows(NativeError.class, run::next);
      assertEquals("RUN_CANCELLED", cancelled.code());
    }
  }

  @Test
  void aPanicAtStartOrInTheBodyIsAnInternalErrorAndTheNextRunWorks() throws Exception {
    for (String at : new String[] {"start", "body", "adapter_stack"}) {
      StructuredDataException raised;
      try {
        Run run =
            Run.start(
                TestOps.start("panic", Map.of("at", at, "payload", "a test-only panic")),
                Map.of());
        try (run) {
          NativeError failed = assertThrows(NativeError.class, run::next);
          raised = failed;
          // The run is poisoned: a later call is refused, never a crash.
          NativeError poisoned = assertThrows(NativeError.class, run::next);
          assertEquals("RUN_POISONED", poisoned.code());
        }
      } catch (NativeError refused) {
        raised = refused;
      }
      assertEquals("INTERNAL_ERROR", raised.code(), at);
      assertEquals(Boolean.TRUE, raised.details().get("panic"), at);
      assertTrue(!raised.getMessage().contains("a test-only panic"), at);
    }
    try (Run run = Run.start(TestOps.start("log", Map.of("hold_ms", 0)), Map.of())) {
      assertEquals("done", next(run).get("step"));
    }
  }

  @Test
  void takingAnAttachmentTheRunDoesNotHoldIsNull() throws Exception {
    try (Run run = Run.start(TestOps.start("log", Map.of("hold_ms", 0)), Map.of())) {
      next(run);
      assertNull(run.takeAttachment(99));
      assertNull(run.takeWarnings());
    }
  }
}
