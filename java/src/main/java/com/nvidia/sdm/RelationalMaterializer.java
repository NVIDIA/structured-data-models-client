// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.ArrowFrames;
import com.nvidia.sdm.internal.MaterializedRequest;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Indices;
import com.nvidia.sdm.internal.NativeFrameGraph;
import com.nvidia.sdm.internal.Results;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * An in-memory Relational graph and its continuing sampler state.
 * The schema already owns snapshots, so callers may close their Arrow tables.
 * Reuse this owner across queries; close releases its native graph.
 * Materialization and close are serialized, including failed calls.
 */
public final class RelationalMaterializer implements AutoCloseable {
  private final Schema schema;
  private NativeFrameGraph graph;
  private boolean closed;

  public RelationalMaterializer(Schema schema) {
    this.schema = Objects.requireNonNull(schema, "schema");
    if (schema.data().isEmpty()) throw new StructuredDataException(
        "an in-memory materializer needs Schema.fromData()", "INVALID_REQUEST", Map.of(), null);
  }

  private NativeFrameGraph graph() {
    if (graph != null) return graph;
    List<String> names = new ArrayList<>(schema.data().keySet());
    List<Integer> ids = new ArrayList<>();
    Map<Integer, byte[]> tables = new LinkedHashMap<>();
    for (String name : names) {
      int id = tables.size() + 3;
      ids.add(id); tables.put(id, schema.data().get(name));
    }
    Map<String, Object> op = Map.of("op", "prepare_frame_graph", "schema", 1, "names", 2, "tables", ids);
    try (Host.Finished finished = ArrowFrames.run(op,
        Map.of(1, Host.attachment(schema.toHeld()), 2, Host.attachment(names)), tables, Host.Call.none())) {
      graph = finished.graph(((Number) finished.result().get("graph")).intValue());
      return graph;
    }
  }

  synchronized Request materialize(PredictiveQuery query, MaterializeOptions options) {
    if (closed) throw new StructuredDataException("this materializer was closed", "INVALID_CONFIGURATION", Map.of(), null);
    if (options.model() != null && !SdmClient.KUMO_RELATIONAL.equals(options.model()))
      throw new StructuredDataException("a Relational materializer needs kumo-relational", "INVALID_REQUEST", Map.of(), null);
    if (options.indices() == null) throw new StructuredDataException(
        "materialize() needs indices naming the rows to score", "INVALID_REQUEST", Map.of(), null);
    Map<String, Object> input = new LinkedHashMap<>();
    input.put("query", query.text());
    input.put("candidates", PredictiveQuery.candidates(options.candidates()));
    input.put("anchor_time", options.anchorTime() == MaterializeOptions.EACH_ENTITY ? null : PredictiveQuery.timestamp(options.anchorTime()));
    input.put("anchor_entity", options.anchorTime() == MaterializeOptions.EACH_ENTITY);
    input.put("seed", PredictiveQuery.seed(options.options().get("random_seed")));
    input.put("unseeded", options.options().containsKey("random_seed") && options.options().get("random_seed") == null);
    input.put("options", List.of());
    Map<String, Object> op = Map.of("op", "materialize_frames", "graph", 1, "input", 2, "indices", 3, "values", 4);
    Map<Integer, Object> attached = new LinkedHashMap<>();
    attached.put(1, graph());
    attached.put(2, Host.attachment(input));
    attached.put(3, Indices.column(options.indices(), ArrowData.allocator()));
    attached.put(4, Host.optionAttachment(PredictiveQuery.values(options)));
    try (Host.Finished finished = Host.run(op, attached)) {
      Results.Gathered gathered = (Results.Gathered) Results.expect(finished.result(), "materialize");
      @SuppressWarnings("unchecked") Map<String, Object> state = (Map<String, Object>) finished.result().get("state");
      MaterializedRequest retained = new MaterializedRequest(state, finished);
      return new Request(schema, retained, query.text(), gathered.query(),
          PredictiveQuery.inferenceAsked(options.options().get("inference_config")));
    }
  }

  @Override public synchronized void close() {
    if (!closed) {
      closed = true;
      if (graph != null) { graph.close(); graph = null; }
    }
  }
}
