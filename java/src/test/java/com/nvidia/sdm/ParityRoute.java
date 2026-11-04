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
  private static final String DATABRICKS_SDK =
      "com.databricks.sdk.core.DatabricksCliCredentialsProvider";

  static boolean databricksAuthentication(String command, String executable, Set<String> callers) {
    if (!callers.contains(DATABRICKS_SDK)
        || !executable.matches("/[A-Za-z0-9_./-]+/databricks")) return false;
    String prefix = "/bin/bash -c " + executable + " ";
    if (!command.startsWith(prefix)) return false;
    String arguments = command.substring(prefix.length());
    return arguments.equals("version --output json")
        || (callers.contains("com.databricks.sdk.core.CliTokenSource")
            && arguments.matches("auth token --profile [A-Za-z0-9_.-]+ --force-refresh"));
  }

  static List<String> authenticationReceipt(String command) {
    return command.endsWith(" version --output json")
        ? List.of("databricks", "version", "--output", "json")
        : List.of("databricks", "auth", "token", "--profile", "<redacted>", "--force-refresh");
  }

  private static Path databricksExecutable() {
    for (String directory : System.getenv().getOrDefault("PATH", "").split(":")) {
      if (directory.isBlank()) continue;
      Path candidate = Path.of(directory, "databricks").toAbsolutePath().normalize();
      if (Files.isRegularFile(candidate) && Files.isExecutable(candidate)) return candidate;
    }
    return null;
  }

  private final Path authExecutable = databricksExecutable();
  private final Recording recording = new Recording();
  private final Set<String> workers = ConcurrentHashMap.newKeySet();
  private Map<String, Object> current;
  private Map<String, Integer> calls;
  private List<Map<String, Object>> transport;
  private Integer bodyId;
  private String operation;
  private Object sessionId;
  private Map<String, Object> lastCall;

  synchronized void capture(List<Map<String, Object>> target) {
    transport = target;
  }

  ParityRoute() {
    recording.enable("jdk.ProcessStart").withStackTrace();
    recording.enable("jdk.ThreadStart");
    recording.start();
    workers.add(ProcessHandle.current().pid() + ":" + Thread.currentThread().getId());
    Trace.install(
        new Trace() {
          @Override
          public void stepped(Map<String, Object> step) {
            synchronized (ParityRoute.this) {
              if (transport == null || !"call_model".equals(step.get("step"))) return;
              Object endpoint = step.get("endpoint");
              String kind =
                  endpoint instanceof Map<?, ?> value
                      ? value.keySet().iterator().next().toString()
                      : String.valueOf(endpoint);
              operation =
                  switch (kind) {
                    case "predict" -> "predict";
                    case "create_session" -> "fit";
                    case "session_predict" -> "score";
                    case "delete_session" -> "close";
                    default -> throw new IllegalStateException("unknown endpoint operation");
                  };
              sessionId =
                  endpoint instanceof Map<?, ?> fields
                      ? ParityRunner.map(fields.get(kind)).get("session_id")
                      : null;
              bodyId = ((Number) step.get("body")).intValue();
            }
          }

          @Override
          public void taken(int id, Object value, boolean arrow) {
            synchronized (ParityRoute.this) {
              if (transport == null || arrow || !Objects.equals(bodyId, id)) return;
              var call = new LinkedHashMap<String, Object>();
              call.put("operation", operation);
              if (sessionId != null) call.put("session_id", sessionId);
              byte[] bytes = (byte[]) value;
              // Trace sees the native sentinel before transport; HTTP DELETE sends no body.
              // ParitySupplementTest checks the actual server-received byte count.
              try {
                call.put(
                    "payload",
                    bytes.length == 0 || "close".equals(operation)
                        ? null
                        : com.nvidia.sdm.internal.Json.read(bytes));
              } catch (Exception error) {
                throw new IllegalStateException("invalid model request JSON", error);
              }
              transport.add(call);
              lastCall = call;
              bodyId = null;
            }
          }

          @Override
          public void modelAnswered(byte[] response) {
            synchronized (ParityRoute.this) {
              if (transport == null || lastCall == null) return;
              if (Set.of("fit", "predict", "score").contains(lastCall.get("operation"))) {
                try {
                  var responseBody = ParityRunner.map(com.nvidia.sdm.internal.Json.read(response));
                  if ("fit".equals(lastCall.get("operation")))
                    lastCall.put("session_id", responseBody.get("session_id"));
                  else lastCall.put("response_id", responseBody.get("id"));
                } catch (Exception error) {
                  throw new IllegalStateException("invalid model response JSON", error);
                }
              }
              lastCall = null;
            }
          }

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
    transport = null;
    bodyId = null;
    lastCall = null;
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
          else {
            var callers = new HashSet<String>();
            if (event.getStackTrace() != null)
              for (var frame : event.getStackTrace().getFrames())
                callers.add(frame.getMethod().getType().getName());
            if (authExecutable != null
                && databricksAuthentication(command, authExecutable.toString(), callers))
              observed.add(
                  Map.of(
                      "pid", event.getLong("pid"),
                      "command", authenticationReceipt(command),
                      "origin", DATABRICKS_SDK,
                      "executable_sha256", ParityRunner.fileHash(authExecutable),
                      "purpose", "Databricks SDK authentication only"));
            else processes.add(Long.toString(event.getLong("pid")));
          }
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
