// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/** Retains the complete core RequestState, including every part's attachments. */
public final class FrameRequest {
  private final Map<String, Object> state;
  private final Map<Integer, Object> attachments = new LinkedHashMap<>();
  private final List<byte[]> payloads = new ArrayList<>();

  public FrameRequest(Map<String, Object> state, Host.Finished finished) {
    this.state = state;
    take(state, finished);
    for (Map<String, Object> part : parts(state)) take(part, finished);
  }

  private void take(Map<String, Object> part, Host.Finished finished) {
    int payload = ((Number) part.get("payload")).intValue();
    byte[] bytes = finished.bytes(payload);
    attachments.put(payload, bytes);
    payloads.add(bytes);
    if (part.get("query") instanceof Number query)
      attachments.put(query.intValue(), finished.bytes(query.intValue()));
  }

  @SuppressWarnings("unchecked")
  private static List<Map<String, Object>> parts(Map<String, Object> state) {
    if (state.get("forecast") instanceof List<?> parts) return (List<Map<String, Object>>) parts;
    if (state.get("batches") instanceof List<?> parts) return (List<Map<String, Object>>) parts;
    if (state.get("ranking") instanceof Map<?, ?> rank)
      return (List<Map<String, Object>>) rank.get("parts");
    return List.of();
  }

  public Map<Integer, Object> attachments() { return new LinkedHashMap<>(attachments); }
  public List<byte[]> payloads() { return List.copyOf(payloads); }
  public boolean inParts() { return state.containsKey("forecast") || state.containsKey("ranking"); }

  /** Host rendering only: the core names candidates by their positions. */
  public Map<String, java.util.function.Function<Map<String, Object>, Object>> held() {
    if (!(state.get("ranking") instanceof Map<?, ?> ranking)) return Map.of();
    Map<Integer, Object> values = new LinkedHashMap<>();
    List<Map<?, ?>> named = new ArrayList<>();
    named.add(ranking);
    for (Object part : (List<?>) ranking.get("parts")) named.add((Map<?, ?>) part);
    for (Map<?, ?> part : named) {
      try {
        values.put(((Number) part.get("candidate")).intValue(),
            Json.read(((String) part.get("value")).getBytes(java.nio.charset.StandardCharsets.UTF_8)));
      } catch (Json.JsonException e) { throw new IllegalStateException("invalid core candidate", e); }
    }
    return Map.of("candidates", fields -> ((List<?>) fields.get("candidates")).stream()
        .map(position -> values.get(((Number) position).intValue())).toList());
  }

  @SuppressWarnings("unchecked")
  public Map<String, Object> predictionState(Supplier<String> requestId) {
    Map<String, Object> copy = new LinkedHashMap<>(state);
    List<Map<String, Object>> parts = new ArrayList<>();
    for (Map<String, Object> part : parts(state)) {
      Map<String, Object> sent = new LinkedHashMap<>(part);
      sent.put("request_id", requestId.get());
      parts.add(sent);
    }
    if (copy.containsKey("forecast")) copy.put("forecast", parts);
    else if (copy.containsKey("batches")) copy.put("batches", parts);
    else if (copy.get("ranking") instanceof Map<?, ?> rank) {
      Map<String, Object> ranking = new LinkedHashMap<>((Map<String, Object>) rank);
      ranking.put("parts", parts);
      copy.put("ranking", ranking);
    }
    return copy;
  }
}
