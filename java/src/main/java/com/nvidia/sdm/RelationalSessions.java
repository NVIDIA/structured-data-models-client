// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowTabular;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Json;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Maps Java request/session owners onto shared Rust frame-session operations. */
final class RelationalSessions {
  private RelationalSessions() {}

  static void requireRequest(Request request) {
    if (request.direct()) throw new StructuredDataException(
        "this session needs a materialized Relational request", "INVALID_REQUEST", Map.of(), null);
    if (request.frame() != null && request.frame().inParts()) throw new StructuredDataException(
        "a session pins one question; use predict() for ranking or forecast parts",
        "UNSUPPORTED_REQUEST", Map.of(), null);
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> read(byte[] bytes) {
    try { return (Map<String, Object>) Json.read(bytes); }
    catch (Json.JsonException e) { throw new IllegalStateException("invalid core payload", e); }
  }

  @SuppressWarnings("unchecked")
  static Map<String, Object> question(Request request) {
    Map<String, Object> task = (Map<String, Object>) read(request.payloadBytes().get(0)).get("task");
    Map<String, Object> target = (Map<String, Object>) task.get("target");
    Map<String, Object> question = new LinkedHashMap<>();
    question.put("query", request.query());
    question.put("kind", task.get("kind"));
    question.put("target", target.get("column_name"));
    question.put("dtype", target.get("dtype"));
    question.put("entity_table", request.entityTable());
    question.put("context", null);
    return question;
  }

  private static Host.Finished run(SdmClient client, Map<String, Object> input,
      Request request, byte[] pinned) {
    String action = (String) input.get("action");
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(input));
    List<Integer> ids = new ArrayList<>();
    if (request != null) {
      for (byte[] bytes : request.payloadBytes()) {
        int id = attachments.size() + 1;
        ids.add(id); attachments.put(id, bytes);
      }
    }
    Map<String, Object> op = new LinkedHashMap<>();
    op.put("op", "frame_session"); op.put("action", action); op.put("input", 1);
    if (!action.equals("close")) op.put("payloads", ids);
    if (pinned != null) {
      int id = attachments.size() + 1;
      attachments.put(id, pinned); op.put("pinned", id);
    }
    return Host.run(op, attachments, new Host.Call(null, client.tabularTransport(), Map.of()));
  }

  static FittedModel fit(SdmClient client, Request request, String model, List<?> holdout) {
    requireRequest(request);
    List<Object> keys = new ArrayList<>();
    for (Object key : holdout) {
      List<?> parts = key instanceof List<?> list ? list : java.util.Collections.singletonList(key);
      keys.add(parts.stream().map(ArrowTabular::key).toList());
    }
    Map<String, Object> input = new LinkedHashMap<>();
    input.put("action", "fit"); input.put("model", model); input.put("outputs", null);
    input.put("question", question(request)); input.put("holdout", Map.of("rows", keys));
    input.put("windowed", request.targetMovesWithTime()); input.put("request_id", SdmClient.requestId());
    input.put("inference_asked", request.inferenceAsked());
    try (Host.Finished finished = run(client, input, request, null)) {
      Map<String, Object> result = ArrowTabular.value(finished);
      return new FittedModel(client, result, finished.bytes(((Number) result.get("pinned")).intValue()));
    }
  }

  static PredictionResult predict(SdmClient client, Map<String, Object> state, byte[] pinned,
      Request request) {
    requireRequest(request);
    Map<String, Object> input = Map.of("action", "predict", "state", state,
        "question", question(request), "request_ids", request.payloadBytes().stream()
            .map(ignored -> SdmClient.requestId()).toList(), "inference_asked", request.inferenceAsked());
    try (Host.Finished finished = run(client, input, request, pinned)) {
      return SdmClient.predicted(finished);
    }
  }

  static Map<String, Object> close(SdmClient client, Map<String, Object> state) {
    try (Host.Finished finished = run(client, Map.of("action", "close", "state", state), null, null)) {
      return Map.of("state", ArrowTabular.value(finished));
    }
  }
}
