// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.HttpTransport;
import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.NativeCore;
import com.nvidia.sdm.internal.PythonText;
import com.nvidia.sdm.internal.Results;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A connection to one model endpoint, as Python's {@code SDMClient}: {@link #predict} answers a
 * request the core gathered, on the core, sending the core's call to the endpoint as the bytes it
 * wrote, over {@code java.net.http} (design choice 10, "Model calls"). No request is made until a
 * prediction. Close it, or use it in a try-with-resources.
 *
 * <pre>{@code
 * try (SdmClient client = SdmClient.connect("http://localhost:8000")) {
 *   PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL);
 * }
 * }</pre>
 */
public final class SdmClient implements AutoCloseable {
  /** The relational model's id, as Python's {@code KumoRelational()} names it. */
  public static final String KUMO_RELATIONAL = "kumo-relational";

  private static final SecureRandom IDS = new SecureRandom();

  private final HttpTransport transport;
  private volatile boolean closed;

  private SdmClient(HttpTransport transport) {
    this.transport = transport;
  }

  /**
   * A client of the endpoint at {@code url}, its base URL (for example {@code
   * http://localhost:8000}).
   *
   * @throws StructuredDataException {@code INVALID_CONFIGURATION} for a URL that is not a base
   *     endpoint, or a credential over plaintext HTTP to another host
   */
  public static SdmClient connect(String url) {
    return connect(url, ClientOptions.defaults());
  }

  /** A client of the endpoint at {@code url}, reached as {@code options} say. */
  public static SdmClient connect(String url, ClientOptions options) {
    Objects.requireNonNull(options, "options");
    return new SdmClient(
        new HttpTransport(url, options.apiKey(), options.timeout(), options.maxRetries()));
  }

  /** Where this client sends requests. */
  public String url() {
    return transport.url();
  }

  /**
   * What {@code model} supports, as far as this client knows: the core's table of the models, not
   * the endpoint's answer, as Python's {@code capabilities()}.
   *
   * @throws StructuredDataException {@code UNKNOWN_MODEL} for a model the contract does not carry
   */
  @SuppressWarnings("unchecked")
  public ModelCapabilities capabilities(String model) {
    Map<String, Object> table;
    try {
      table = (Map<String, Object>) Json.read(NativeCore.capabilities());
    } catch (Json.JsonException e) {
      throw new StructuredDataException(
          "the core's table of models cannot be read", "INTERNAL_ERROR", Map.of(), null);
    }
    for (Object item : (List<Object>) table.get("models")) {
      Map<String, Object> each = (Map<String, Object>) item;
      if (each.get("model").equals(model)) {
        return new ModelCapabilities(
            (String) each.get("model"),
            (List<String>) each.get("tasks"),
            (List<String>) each.get("outputs"),
            Boolean.TRUE.equals(each.get("single_table")),
            Boolean.TRUE.equals(each.get("multiple_tables")),
            Boolean.TRUE.equals(each.get("relationships")),
            (List<String>) each.get("explanation_formats"),
            Boolean.TRUE.equals(each.get("sessions")),
            (String) each.get("source"));
      }
    }
    throw new StructuredDataException(
        "unknown model " + model, "UNKNOWN_MODEL", Map.of("model", String.valueOf(model)), null);
  }

  /** The answer to {@code request} by {@code model}, with the outputs the model gives by default. */
  public PredictionResult predict(Request request, String model) {
    return predict(request, model, null);
  }

  /**
   * The answer to {@code request} by {@code model}, asking for {@code outputs} (null: what the
   * model gives for this task by default), predicted on the core as Python predicts a request
   * materialized on the core: the core checks the model and the outputs, makes each call the
   * request is asked as, reads each answer and lays out the result.
   *
   * @throws StructuredDataException the core's refusal ({@code UNKNOWN_MODEL}, {@code
   *     UNSUPPORTED_REQUEST} naming the rule, {@code INVALID_RESPONSE} for an answer it cannot
   *     read), {@code TRANSPORT_ERROR} when the endpoint could not be reached; a {@link
   *     ModelEndpointException} when the endpoint refused a call
   */
  public PredictionResult predict(Request request, String model, List<String> outputs) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(model, "model");
    if (closed) {
      throw new StructuredDataException(
          "this client was closed", "INVALID_CONFIGURATION", Map.of(), null);
    }
    List<byte[]> payloads = request.payloadBytes();
    Map<String, Object> state = new LinkedHashMap<>();
    state.put("version", request.version());
    state.put("payload", 1);
    state.put("query", 2);
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, payload(payloads.get(0)));
    attachments.put(2, Host.attachment(request.query()));
    if (request.batchRows() != null) {
      List<Object> batches = new ArrayList<>();
      for (int at = 0; at < request.batchRows().size(); at++) {
        int id = 3 + at;
        attachments.put(id, payload(payloads.get(at + 1)));
        Map<String, Object> batch = new LinkedHashMap<>();
        batch.put("payload", id);
        batch.put("rows", request.batchRows().get(at));
        batch.put("request_id", requestId());
        batches.add(batch);
      }
      state.put("batches", batches);
    }
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "predict");
    operation.put("state", state);
    operation.put("model", model);
    operation.put("request_id", requestId());
    operation.put("outputs", outputs == null ? List.of() : List.copyOf(outputs));
    operation.put("inference_asked", request.inferenceAsked());
    if (outputs == null) {
      operation.put("default_outputs", true);
    }
    try (Host.Finished finished =
        Host.run(operation, attachments, new Host.Call(null, this::call, Map.of()))) {
      Results.Predicted predicted =
          (Results.Predicted) Results.expect(finished.result(), "predict");
      Object metadata = read(finished.bytes(predicted.metadata()));
      byte[] rows = finished.arrow(predicted.result());
      return new PredictionResult(
          predicted.model(),
          predicted.requestId(),
          metadata,
          predicted.outputs(),
          rows);
    }
  }

  /** One call of the core's, through the transport, its refusal kept for the run's end. */
  private byte[] call(byte[] body) throws Host.TransportFailed, Host.Refused {
    try {
      return transport.predict(body);
    } catch (HttpTransport.Failed failed) {
      throw new Host.TransportFailed();
    } catch (ModelEndpointException refused) {
      throw new Host.Refused(refused::with, refused.status());
    }
  }

  /**
   * A payload as the Python host sends it: the JSON the core wrote, read and written back by
   * Python's {@code json.dumps}, a value it has no form for refused as its transport refuses it.
   */
  private static byte[] payload(byte[] written) {
    try {
      return PythonText.dumpsDefault(written).getBytes(StandardCharsets.UTF_8);
    } catch (IllegalArgumentException e) {
      throw new StructuredDataException(
          "this request carries a value that cannot be sent as JSON",
          "INVALID_REQUEST",
          Map.of(),
          null);
    }
  }

  /** A call's id, as the adapter draws one: {@code sdm-} and twelve hexadecimal digits. */
  private static String requestId() {
    byte[] random = new byte[6];
    IDS.nextBytes(random);
    return "sdm-" + HexFormat.of().formatHex(random);
  }

  private static Object read(byte[] json) {
    try {
      return Json.read(json);
    } catch (Json.JsonException e) {
      throw new StructuredDataException(
          "the core answered JSON this host cannot read", "INTERNAL_ERROR", Map.of(), null);
    }
  }

  /** Closes the client; a later prediction is refused. */
  @Override
  public void close() {
    closed = true;
  }

  @Override
  public String toString() {
    return "SdmClient(" + transport.url() + ")";
  }
}
