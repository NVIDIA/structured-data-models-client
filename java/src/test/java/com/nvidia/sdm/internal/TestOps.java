// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** The test-only operations, and a subprocess JVM, for the binding's tests. */
final class TestOps {
  private TestOps() {}

  /** Skips the test unless the library loaded is the test-only build. */
  static void required() {
    assumeTrue(
        Boolean.TRUE.equals(NativeCore.load().get("test_ops")),
        "needs the library built with the test-only operations (-Dsdm.native.library)");
  }

  static byte[] json(Object value) {
    try {
      return Json.write(value);
    } catch (Json.JsonException e) {
      throw new IllegalStateException(e);
    }
  }

  /** A start of the test-only operation {@code test}, with its fields. */
  static byte[] start(String test, Map<String, Object> fields) {
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "test");
    operation.put("test", test);
    operation.putAll(fields);
    Map<String, Object> start = new LinkedHashMap<>();
    start.put("message", "start");
    start.put("contract", Host.contract());
    start.put("facts", HostFacts.now());
    start.put("operation", operation);
    return json(start);
  }

  /** The test-only fetch of one scan, its data discovered, echoed when {@code echo}. */
  static Map<String, Object> fetch(boolean echo) {
    Map<String, Object> request = new LinkedHashMap<>();
    request.put("request", "scan");
    request.put("table", Map.of("source", List.of("t")));
    request.put("columns", List.of());
    request.put("filters", List.of());
    request.put("limit", null);
    Map<String, Object> fetch = new LinkedHashMap<>();
    fetch.put("request", request);
    fetch.put("plan", Map.of("plan", "host", "expect", Map.of("expect", "discover")));
    fetch.put("echo", echo);
    return fetch;
  }

  static byte[] fetchReply(int data) {
    Map<String, Object> reply = new LinkedHashMap<>();
    reply.put("message", "fetch_reply");
    reply.put("id", 1);
    reply.put("outcome", Map.of("outcome", "data", "data", data));
    return json(reply);
  }

  /** What a subprocess JVM printed, and how it ended. */
  record Ended(int status, String out, String err) {}

  /** Runs {@code main} in a new JVM with this one's class path and the runtime flags. */
  static Ended jvm(Class<?> main, String... properties) throws IOException, InterruptedException {
    return jvmWithout(null, main, properties);
  }

  /** {@link #jvm}, leaving out the runtime flag {@code left}. */
  static Ended jvmWithout(String left, Class<?> main, String... properties)
      throws IOException, InterruptedException {
    List<String> command = new ArrayList<>();
    command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
    String flags = System.getProperty("sdm.java.flags", "");
    for (String flag : flags.split(" ")) {
      if (!flag.isBlank() && !flag.equals(left)) {
        command.add(flag);
      }
    }
    String library = System.getProperty(NativeCore.LIBRARY_PROPERTY, "");
    if (!library.isEmpty()) {
      command.add("-D" + NativeCore.LIBRARY_PROPERTY + "=" + library);
    }
    for (String property : properties) {
      command.add("-D" + property);
    }
    command.add("-cp");
    command.add(System.getProperty("java.class.path"));
    command.add(main.getName());
    Process process =
        new ProcessBuilder(command).redirectInput(ProcessBuilder.Redirect.from(nullFile())).start();
    if (!process.waitFor(120, TimeUnit.SECONDS)) {
      process.destroyForcibly();
      throw new IllegalStateException("the subprocess JVM did not finish");
    }
    return new Ended(
        process.exitValue(),
        new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8),
        new String(process.getErrorStream().readAllBytes(), StandardCharsets.UTF_8));
  }

  private static java.io.File nullFile() {
    return new java.io.File(System.getProperty("os.name").startsWith("Windows") ? "NUL" : "/dev/null");
  }
}
