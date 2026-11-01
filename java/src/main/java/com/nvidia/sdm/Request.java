// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.ArrowTabular;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Results;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A request the core gathered for one query ({@link PredictiveQuery#materialize}), as Python's
 * {@code Request}: its payloads, one per call the endpoint is asked, and what the core resolved of
 * the query. It holds what {@link SdmClient#predict} hands back to the core, and is plain data:
 * the schema it was gathered from never changes under it.
 */
public final class Request {
  private byte[] contextData;
  private byte[] predictData;
  private Map<String, Object> direct;
  private com.nvidia.sdm.internal.MaterializedRequest materialized;

  /** A Tabular request from Arrow tables. The constructor takes an owned snapshot. */
  public Request(org.apache.arrow.vector.table.Table context,
      org.apache.arrow.vector.table.Table predict, String target, String task) {
    this(context, predict, target, task, RequestOptions.builder().build());
  }

  /** Direct Tabular request with optional explicit schema and entity-table name. */
  public Request(org.apache.arrow.vector.table.Table context,
      org.apache.arrow.vector.table.Table predict, String target, String task, RequestOptions options) {
    this(ArrowTabular.snapshot(context), ArrowTabular.snapshot(predict),
        directOptions(target, task, options), "");
    this.inference = options.inference();
    this.taskOptions = options.taskOptions();
  }

  private Map<String, Object> inference = Map.of();
  private Map<String, Object> taskOptions = Map.of();
  public Map<String, Object> inference() { return inference; }
  public Map<String, Object> taskOptions() { return taskOptions; }

  private static Map<String, Object> directOptions(String target, String task, RequestOptions options) {
    Map<String, Object> input = ArrowTabular.request(target, task, options.entityTable(),
        options.schema() == null ? null : options.schema().toHeld());
    input.put("inference", com.nvidia.sdm.internal.DirectOptions.encode(options.inference()));
    input.put("task_options", com.nvidia.sdm.internal.DirectOptions.encode(options.taskOptions()));
    return input;
  }

  Request(byte[] context, byte[] predict, Map<String, Object> request, String query) {
    this.contextData = context;
    this.predictData = predict;
    try (Host.Finished finished = ArrowTabular.run("prepare", request,
        Map.of("context", context, "predict", predict), null)) {
      Map<String, Object> description = ArrowTabular.value(finished);
      this.schema = Schema.held(description.get("schema"), null);
      this.direct = new java.util.LinkedHashMap<>(request);
      this.direct.put("target", description.get("target"));
      this.direct.put("task", description.get("task"));
      this.direct.put("entity_table", description.get("entity_table"));
      this.direct.put("schema", description.get("schema"));
    }
    if (!query.isEmpty()) {
      this.direct.put("query", query);
      try {
        byte[] digest = java.security.MessageDigest.getInstance("SHA-256").digest(context);
        this.direct.put("context_identity", new java.math.BigInteger(1, digest).toString());
      } catch (java.security.NoSuchAlgorithmException error) { throw new IllegalStateException(error); }
    }
    this.payloads = List.of();
    this.query = query;
    this.inferenceAsked = List.of();
    this.facts = null;
  }

  /** The held context rows and related tables; inspection never samples again. */
  public RequestSplit context() { return new RequestSplit(this, "context"); }

  /** The held prediction rows and related tables; each reader is independently owned. */
  public RequestSplit predict() { return new RequestSplit(this, "predict"); }

  boolean direct() { return direct != null; }
  Map<String, Object> directInput() { return direct; }
  byte[] contextData() { return contextData; }
  byte[] predictData() { return predictData; }

  private final Schema schema;
  private final List<byte[]> payloads;
  private final String query;
  private final Results.QueryFacts facts;
  private final List<String> inferenceAsked;

  Request(Schema schema, com.nvidia.sdm.internal.MaterializedRequest materialized, String query,
      Results.QueryFacts facts, List<String> inferenceAsked) {
    this.schema = schema; this.materialized = materialized; this.payloads = materialized.payloads();
    this.query = query; this.facts = facts; this.inferenceAsked = List.copyOf(inferenceAsked);
  }

  com.nvidia.sdm.internal.MaterializedRequest materialized() { return materialized; }

  /** Save an in-memory request's owned snapshots; leaves the caller's stream open. */
  public void writeTo(java.io.OutputStream output) throws java.io.IOException {
    Snapshots.Writer writer = new Snapshots.Writer();
    Map<String, Object> saved = new java.util.LinkedHashMap<>();
    saved.put("schema", schema.saved(writer));
    saved.put("query", query); saved.put("inference", inference); saved.put("task_options", taskOptions);
    if (direct()) {
      saved.put("kind", "direct_request"); saved.put("input", direct);
      saved.put("context", writer.blob(contextData)); saved.put("predict", writer.blob(predictData));
    } else {
      saved.put("kind", "materialized_request"); saved.put("state", materialized.snapshotState());
      saved.put("inference_asked", inferenceAsked);
      List<Object> attached = new ArrayList<>();
      materialized.attachments().forEach((id, bytes) -> attached.add(Map.of("id", id, "blob", writer.blob((byte[]) bytes))));
      saved.put("attachments", attached);
    }
    writer.write(saved, output);
  }

  /** Restore snapshots without sampling or HTTP; core validation still governs prediction. */
  public static Request readFrom(java.io.InputStream input) throws java.io.IOException {
    Snapshots.Reader reader = Snapshots.read(input);
    Map<String, Object> saved = reader.manifest;
    String kind = Snapshots.text(saved.get("kind"));
    if (!kind.equals("direct_request") && !kind.equals("materialized_request")) {
      throw Snapshots.invalid("snapshot is not a request", null);
    }
    Schema schema = Schema.restored(saved.get("schema"), reader);
    String query = Snapshots.text(saved.get("query"));
    Map<String, Object> inference = com.nvidia.sdm.internal.DirectOptions.snapshot(Snapshots.object(saved.get("inference")));
    Map<String, Object> taskOptions = com.nvidia.sdm.internal.DirectOptions.snapshot(Snapshots.object(saved.get("task_options")));
    Request request;
    if (kind.equals("direct_request")) {
      byte[] context = reader.blob(saved.get("context")), predict = reader.blob(saved.get("predict"));
      Snapshots.validateArrow(context); Snapshots.validateArrow(predict);
      Map<String, Object> direct = Snapshots.object(saved.get("input"));
      Map<String, Object> empty = Map.of("entries", List.of());
      if (!java.util.Objects.equals(direct.getOrDefault("inference", empty), com.nvidia.sdm.internal.DirectOptions.encode(inference))
          || !java.util.Objects.equals(direct.getOrDefault("task_options", empty), com.nvidia.sdm.internal.DirectOptions.encode(taskOptions))) {
        throw Snapshots.invalid("snapshot option metadata disagrees with its request", null);
      }
      request = new Request(context, predict, direct, query);
      if (!request.schema().toHeld().equals(schema.toHeld())) throw Snapshots.invalid("snapshot schema disagrees with its request", null);
    } else {
      Map<String, Object> state = Snapshots.object(saved.get("state"));
      int version = Snapshots.integer(state.get("version"));
      Object supportedVersion = com.nvidia.sdm.internal.NativeCore.load().get("request_state_version");
      if (!(supportedVersion instanceof Number supportedNumber)) {
        throw new StructuredDataException("native library does not support request restoration", "UNSUPPORTED_REQUEST", Map.of(), null);
      }
      int supported = supportedNumber.intValue();
      if (version != supported) throw Snapshots.invalid("unsupported materialized request state version", null);
      Map<Integer, byte[]> attachments = new java.util.LinkedHashMap<>();
      for (Object item : Snapshots.list(saved.get("attachments"))) {
        Map<String, Object> entry = Snapshots.object(item);
        int id = Snapshots.integer(entry.get("id"));
        if (id < 0 || attachments.putIfAbsent(id, reader.blob(entry.get("blob"))) != null) {
          throw Snapshots.invalid("duplicate or invalid request attachment id", null);
        }
      }
      validateStateAttachments(state, attachments);
      Map<String, Object> operation = Map.of("op", "validate_query", "query", 1, "schema", 2,
          "profile", Map.of("name", "RELATIONAL"));
      try (Host.Finished finished = Host.run(operation,
          Map.of(1, Host.attachment(query), 2, Host.attachment(schema.queryShape())))) {
        Results.Validated result = (Results.Validated) Results.expect(finished.result(), "validate_query");
        if (!result.accepted() || result.query() == null) throw Snapshots.invalid("snapshot query is not accepted", null);
        request = new Request(schema, com.nvidia.sdm.internal.MaterializedRequest.restored(state, attachments),
            query, result.query(), Snapshots.texts(saved.get("inference_asked")));
      }
    }
    request.inference = inference;
    request.taskOptions = taskOptions;
    reader.complete();
    return request;
  }

  /** Framing and JSON shapes only; native RequestState remains the semantic authority. */
  private static void validateStateAttachments(Map<String, Object> state, Map<Integer, byte[]> attachments)
      throws java.io.IOException {
    List<Map<String, Object>> parts = new ArrayList<>(); parts.add(state);
    int decompositions = 0;
    for (String name : List.of("forecast", "batches")) {
      if (state.containsKey(name)) {
        decompositions++;
        for (Object item : Snapshots.list(state.get(name))) parts.add(Snapshots.object(item));
      }
    }
    if (state.containsKey("ranking")) {
      decompositions++;
      Map<String, Object> ranking = Snapshots.object(state.get("ranking"));
      for (Object item : Snapshots.list(ranking.get("parts"))) parts.add(Snapshots.object(item));
    }
    if (decompositions > 1) throw Snapshots.invalid("snapshot contains conflicting request decompositions", null);
    java.util.Set<Integer> used = new java.util.HashSet<>();
    for (Map<String, Object> part : parts) {
      int payload = Snapshots.integer(part.get("payload"));
      byte[] bytes = attachments.get(payload);
      if (bytes == null) throw Snapshots.invalid("snapshot request is missing its payload", null);
      used.add(payload);
      try { Snapshots.object(Json.read(bytes)); }
      catch (Json.JsonException malformed) { throw Snapshots.invalid("invalid request payload JSON", malformed); }
      if (part == state || !state.containsKey("batches")) {
        int query = Snapshots.integer(part.get("query"));
        byte[] text = attachments.get(query);
        if (text == null) throw Snapshots.invalid("snapshot request is missing its query", null);
        used.add(query);
        try { Snapshots.text(Json.read(text)); }
        catch (Json.JsonException malformed) { throw Snapshots.invalid("invalid request query JSON", malformed); }
      }
    }
    if (!used.equals(attachments.keySet())) throw Snapshots.invalid("snapshot has unreferenced request attachments", null);
  }

  /** The schema the request was gathered from. */
  public Schema schema() {
    return schema;
  }

  /** The query the request answers, as the core resolved it. */
  public String query() {
    return query;
  }

  /** Each call's payload, as the JSON text the core wrote, in the order the calls are made. */
  public List<String> payloads() {
    List<String> texts = new ArrayList<>();
    for (byte[] payload : payloads) {
      texts.add(new String(payload, StandardCharsets.UTF_8));
    }
    return texts;
  }

  /** The table whose entities the query is about. */
  public String entityTable() {
    return direct() ? (String) direct.get("entity_table") : facts.entityTable();
  }

  /** The task the query is, as the core resolved it, when it resolved one. */
  public Optional<String> problemType() {
    return direct() ? Optional.of((String) direct.get("task")) : Optional.ofNullable(facts.problemType());
  }

  /** Original query when this request is decomposed into candidate or forecast questions. */
  public Optional<String> asked() {
    return !direct() && materialized.inParts() ? Optional.of(query) : Optional.empty();
  }

  /** Each independent candidate/forecast request, without sampling again. */
  public Optional<List<Request>> parts() {
    if (direct() || !materialized.inParts()) return Optional.empty();
    List<Request> result = new ArrayList<>();
    for (com.nvidia.sdm.internal.MaterializedRequest part : materialized.questions()) {
      String text = part.query();
      Map<String, Object> operation = Map.of("op", "validate_query", "query", 1, "schema", 2,
          "profile", Map.of("name", "RELATIONAL"));
      try (Host.Finished finished = Host.run(operation,
          Map.of(1, Host.attachment(text), 2, Host.attachment(schema.queryShape())))) {
        Results.Validated validated = (Results.Validated) Results.expect(finished.result(), "validate_query");
        if (!validated.accepted() || validated.query() == null)
          throw new StructuredDataException("core produced an invalid request part", "INTERNAL_ERROR", Map.of(), null);
        result.add(new Request(schema, part, text, validated.query(), inferenceAsked));
      }
    }
    return Optional.of(List.copyOf(result));
  }

  /** Whether the answer is a window rather than a standing value, as Python's attribute. */
  public boolean targetMovesWithTime() {
    return !direct() && facts.windowed();
  }

  /**
   * Whether the request needs a model that reads related tables, as Python's attribute: its
   * context carries related tables, or its schema holds a relationship or more than one table.
   */
  public boolean isRelational() {
    if (direct()) return false;
    if (schema.tables().size() > 1 || !schema.relationships().isEmpty()) {
      return true;
    }
    for (byte[] payload : payloads) {
      try {
        if (Json.read(payload) instanceof Map<?, ?> envelope
            && envelope.get("context") instanceof Map<?, ?> context
            && context.get("related_tables") instanceof Map<?, ?> related
            && !related.isEmpty()) {
          return true;
        }
      } catch (Json.JsonException e) {
        return false;
      }
    }
    return false;
  }

  /** How many rows the model learns the answer from, across every call. */
  public long contextRows() {
    return direct() ? directRows(contextData) : rows("context");
  }

  /** How many rows are asked about, across every call. */
  public long predictRows() {
    return direct() ? directRows(predictData) : rows("predict");
  }

  /** Whether the request is known to ask about no row at all. */
  public boolean nothingToScore() {
    if (direct()) return predictRows() == 0;
    for (byte[] payload : payloads) {
      Long rows = rows(payload, "predict");
      if (rows == null || rows != 0) {
        return false;
      }
    }
    return true;
  }

  private static long directRows(byte[] data) {
    try (org.apache.arrow.vector.ipc.ArrowReader reader =
        com.nvidia.sdm.internal.ArrowData.reader(data, com.nvidia.sdm.internal.ArrowData.allocator())) {
      long rows = 0;
      while (reader.loadNextBatch()) rows += reader.getVectorSchemaRoot().getRowCount();
      return rows;
    } catch (java.io.IOException e) { throw new IllegalStateException(e); }
  }

  private long rows(String half) {
    long total = 0;
    for (byte[] payload : payloads) {
      Long rows = rows(payload, half);
      total += rows == null ? 0 : rows;
    }
    return total;
  }

  /** How many instance rows one payload carries in {@code half}, or null when it cannot say. */
  private static Long rows(byte[] payload, String half) {
    try {
      Object read = Json.read(payload);
      if (read instanceof Map<?, ?> envelope
          && envelope.get(half) instanceof Map<?, ?> part
          && part.get("instance_table") instanceof Map<?, ?> table
          && table.get("rows") instanceof List<?> rows) {
        return (long) rows.size();
      }
    } catch (Json.JsonException e) {
      return null;
    }
    return null;
  }

  List<byte[]> payloadBytes() {
    return payloads;
  }

  List<String> inferenceAsked() {
    return inferenceAsked;
  }

  @Override
  public String toString() {
    return "Request(" + query + ", " + predictRows() + " rows to predict, " + contextRows()
        + " context rows in " + payloads.size() + " call(s))";
  }
}
