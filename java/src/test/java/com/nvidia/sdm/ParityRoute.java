// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Trace;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordingFile;

/** Test-only native phase witnesses and process observation around the entire journey. */
final class ParityRoute implements AutoCloseable {
  private final Recording recording = new Recording();
  private final Set<String> workers = ConcurrentHashMap.newKeySet();
  private Map<String, Object> current;
  private Map<String, Integer> calls;

  ParityRoute() {
    recording.enable("jdk.ProcessStart").withStackTrace();
    recording.enable("jdk.ThreadStart");
    recording.start();
    workers.add(ProcessHandle.current().pid() + ":" + Thread.currentThread().getId());
    Trace.install(
        new Trace() {
          @Override
          public void started(byte[] message, Map<Integer, Object> attachments) {
            synchronized (ParityRoute.this) {
              workers.add(ProcessHandle.current().pid() + ":" + Thread.currentThread().getId());
              if (current != null)
                calls.merge(
                    String.valueOf(current.getOrDefault("stage", "setup")), 1, Integer::sum);
            }
          }
        });
  }

  synchronized void begin(Map<String, Object> record) {
    current = record;
    calls = new LinkedHashMap<>();
  }

  synchronized void end(Map<String, Object> record) {
    var route = new LinkedHashMap<String, Object>();
    route.put("installed", true);
    route.put("scope", "whole-journey");
    route.put("legacy_hits", List.of());
    route.put("native_calls", new LinkedHashMap<>(calls));
    record.put("route", route);
    if ("passed".equals(record.get("status"))) {
      var semantic = ParityRunner.map(record.get("semantic"));
      var required = new HashSet<>(ParityRunner.list(semantic.get("lifecycle")));
      required.add("schema");
      if (required.stream().anyMatch(stage -> calls.getOrDefault(stage.toString(), 0) == 0)) {
        record.put("status", "incomplete");
        record.put("reason", "native operation witness missing for a required phase");
      }
    }
    current = null;
  }

  void finish(List<Map<String, Object>> records) throws Exception {
    recording.stop();
    Path file = Files.createTempFile("sdm-parity-route-", ".jfr");
    var processes = new ArrayList<Object>();
    var observed = new ArrayList<Object>();
    try {
      recording.dump(file);
      for (var event : RecordingFile.readAllEvents(file)) {
        if (event.getEventType().getName().equals("jdk.ProcessStart")) {
          String command = event.getString("command");
          boolean sqlite =
              event.getStackTrace() != null
                  && event.getStackTrace().getFrames().stream()
                      .anyMatch(
                          frame ->
                              frame
                                  .getMethod()
                                  .getType()
                                  .getName()
                                  .equals("org.sqlite.util.ProcessRunner"));
          if (sqlite && Set.of("uname -m", "uname -o").contains(command))
            observed.add(
                Map.of(
                    "pid",
                    event.getLong("pid"),
                    "command",
                    Arrays.asList(command.split(" ")),
                    "origin",
                    "org.sqlite.util.ProcessRunner"));
          else processes.add(Long.toString(event.getLong("pid")));
        }
        if (event.getEventType().getName().equals("jdk.ThreadStart")) {
          var thread = event.getThread("thread");
          if (thread != null)
            workers.add(ProcessHandle.current().pid() + ":" + thread.getJavaThreadId());
        }
      }
      for (var record : records) {
        var route = ParityRunner.map(record.get("route"));
        route.put("workers", new ArrayList<>(new TreeSet<>(workers)));
        route.put("unobserved_subprocesses", processes);
        route.put("observed_subprocesses", observed);
        route.put(
            "boundary", "packaged Java API with observed JNI operations; no Python runtime loaded");
        if (!processes.isEmpty()) {
          record.put("status", "incomplete");
          record.put("reason", "child process occurred outside observed Java/JNI boundary");
        }
      }
    } finally {
      Files.deleteIfExists(file);
    }
  }

  @Override
  public void close() {
    Trace.install(null);
    recording.close();
  }
}
