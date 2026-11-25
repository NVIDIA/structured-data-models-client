// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Trace;
import java.util.*;

/** Test-only native lifecycle and model-request observations. */
final class ParityRoute implements AutoCloseable {
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

  @Override
  public void close() {
    Trace.install(null);
  }
}
