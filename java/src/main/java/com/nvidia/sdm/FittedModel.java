// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowTabular;
import com.nvidia.sdm.internal.Host;
import java.util.Map;

/** A server-held context, not a trained model. Close it to release server resources. */
public final class FittedModel implements AutoCloseable {
  private final SdmClient client;
  private Map<String, Object> session;

  FittedModel(SdmClient client, Map<String, Object> session) {
    this.client = client;
    this.session = session;
  }

  @SuppressWarnings("unchecked")
  private Map<String, Object> state() { return (Map<String, Object>) session.get("state"); }

  public synchronized String sessionId() { return (String) state().get("session_id"); }
  public synchronized boolean isClosed() { return Boolean.TRUE.equals(state().get("closed")); }

  /** Score using the schema and examples uploaded at fit; this request's context is not sent. */
  public synchronized PredictionResult predict(Request request) {
    client.requireOpen();
    if (!request.direct()) throw new StructuredDataException("fitted prediction needs an in-memory request",
        "INVALID_REQUEST", Map.of(), null);
    Map<String, Object> input = Map.of("session", session, "request", request.directInput(),
        "request_id", SdmClient.requestId());
    try (Host.Finished finished = ArrowTabular.run("score", input,
        Map.of("predict", request.predictData()), client.tabularTransport())) {
      return SdmClient.predicted(finished);
    }
  }

  /** A failed remote delete leaves this object open so cleanup can be retried. */
  @Override
  public synchronized void close() {
    if (isClosed()) return;
    client.requireOpen();
    try (Host.Finished finished = ArrowTabular.run("close", session, Map.of(), client.tabularTransport())) {
      session = ArrowTabular.value(finished);
      client.released(this);
    }
  }
}
