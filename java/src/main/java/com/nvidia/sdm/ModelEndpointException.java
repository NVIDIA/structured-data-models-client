// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Json;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * The model endpoint refused a call, as Python's {@code NimRequestError}: the endpoint's status
 * ({@link #status}), its own code ({@link #code}, which may be null), its message (its {@code
 * detail} or {@code title}, else the start of its answer) and its other fields in {@link
 * #details}, as the endpoint wrote them (decision E47). When the core had answered earlier calls
 * of the same request, {@code details} also holds how many ({@code calls_answered}).
 */
public final class ModelEndpointException extends StructuredDataException {
  private static final long serialVersionUID = 1L;

  /** The longest message taken from an answer that has no {@code detail} or {@code title}. */
  private static final int SNIPPET = 512;

  private static final Set<String> SAID = Set.of("code", "detail", "title", "type", "status", "instance");

  private final int status;

  private ModelEndpointException(int status, String message, String code, Map<String, ?> details) {
    super(message, code, details, null);
    this.status = status;
  }

  /** The endpoint's HTTP status. */
  public int status() {
    return status;
  }

  /** The refusal an answer of {@code status} with {@code content} is. */
  public static ModelEndpointException of(int status, byte[] content) {
    Map<?, ?> body = Map.of();
    try {
      if (Json.read(content) instanceof Map<?, ?> read) {
        body = read;
      }
    } catch (Json.JsonException e) {
      // Not JSON: its text is the message.
    }
    Object reported = body.get("detail");
    if (reported == null || "".equals(reported)) {
      reported = body.get("title");
    }
    if (reported == null || "".equals(reported)) {
      reported = new String(content, StandardCharsets.UTF_8);
    }
    String message = snippet(String.valueOf(reported));
    if (message.isEmpty()) {
      message = "NIM request failed with status " + status;
    }
    Map<String, Object> details = new LinkedHashMap<>();
    for (Map.Entry<?, ?> entry : body.entrySet()) {
      String key = String.valueOf(entry.getKey());
      if (!SAID.contains(key)) {
        details.put(key, entry.getValue());
      }
    }
    Object code = body.get("code");
    return new ModelEndpointException(
        status, message, code instanceof String text ? text : null, details);
  }

  /** This refusal carrying {@code more} in its details. */
  ModelEndpointException with(Map<String, Object> more) {
    Map<String, Object> details = new LinkedHashMap<>(details());
    details.putAll(more);
    return new ModelEndpointException(status, getMessage(), code(), details);
  }

  private static String snippet(String text) {
    String flat = String.join(" ", text.strip().split("\\s+"));
    return flat.length() <= SNIPPET
        ? flat
        : flat.substring(0, SNIPPET) + "... [truncated, " + flat.length() + " chars total]";
  }
}
