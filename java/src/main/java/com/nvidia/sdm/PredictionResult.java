// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * The answer to one request, as Python's {@code PredictionResult}: the model that answered, the
 * endpoint's id for the call, its metadata, the outputs asked for, and the rows, read as an Arrow
 * {@link VectorSchemaRoot} in the layout the core gives them ({@link #read}): one row per answer,
 * ordered by {@code row_index}, under the entity it is about ({@code id}), with its prediction and
 * the other outputs it carries. Join on {@code id}, or on {@code row_index}, never on row order.
 */
public final class PredictionResult {
  private final String model;
  private final String requestId;
  private final Object metadata;
  private final List<String> outputs;
  private final byte[] rows;
  private final List<Object> unscoredCandidates;
  private record CandidateOwner(long start, long end, Map<Long, Object> values) {}
  private final List<CandidateOwner> candidateOwners;

  PredictionResult(
      String model, String requestId, Object metadata, List<String> outputs, byte[] rows) {
    this(model, requestId, metadata, outputs, rows, List.of());
  }

  PredictionResult(String model, String requestId, Object metadata, List<String> outputs,
      byte[] rows, List<Object> unscoredCandidates) {
    this(model, requestId, metadata, outputs, rows, unscoredCandidates, List.of());
  }

  PredictionResult(String model, String requestId, Object metadata, List<String> outputs,
      byte[] rows, List<Object> unscoredCandidates, Map<Long, Object> candidates) {
    this(model, requestId, metadata, outputs, rows, unscoredCandidates,
        candidates.isEmpty() ? List.of() : List.of(new CandidateOwner(0, Long.MAX_VALUE,
            java.util.Collections.unmodifiableMap(new java.util.LinkedHashMap<>(candidates)))));
  }

  @SuppressWarnings("unchecked")
  private PredictionResult(String model, String requestId, Object metadata, List<String> outputs,
      byte[] rows, List<Object> unscoredCandidates, List<CandidateOwner> owners) {
    this.model = model;
    this.requestId = requestId;
    this.metadata = metadata instanceof Map<?, ?> map
        ? com.nvidia.sdm.internal.DirectOptions.snapshot((Map<String, ?>) map) : Map.of();
    this.outputs = List.copyOf(outputs);
    this.rows = rows;
    this.candidateOwners = List.copyOf(owners);
    this.unscoredCandidates = java.util.Collections.unmodifiableList(new ArrayList<>(unscoredCandidates));
  }

  /** Combine already computed slices using the same native merge rules as prediction batching. */
  public static PredictionResult merge(List<PredictionResult> parts) {
    java.util.Objects.requireNonNull(parts, "parts");
    if (parts.size() == 1) return java.util.Objects.requireNonNull(parts.get(0));
    List<CandidateOwner> owners = new ArrayList<>();
    long ownerOffset = 0;
    List<Object> descriptions = new ArrayList<>();
    List<Object> candidates = new ArrayList<>();
    Map<Integer, byte[]> tables = new java.util.LinkedHashMap<>();
    try {
      for (PredictionResult part : parts) {
        long count = part.rows();
        for (CandidateOwner owner : part.candidateOwners) {
          long end = Math.min(owner.end(), count);
          if (owner.start() < end) owners.add(new CandidateOwner(
              Math.addExact(ownerOffset, owner.start()), Math.addExact(ownerOffset, end), owner.values()));
        }
        ownerOffset = Math.addExact(ownerOffset, count);
        Map<String, Object> description = new java.util.LinkedHashMap<>();
        description.put("model", part.model()); description.put("request_id", part.requestId());
        description.put("outputs", part.outputs());
        description.put("metadata", new String(Json.write(part.metadata()), StandardCharsets.UTF_8));
        List<String> unscored = new ArrayList<>();
        for (Object value : part.unscoredCandidates()) unscored.add(new String(Json.write(value), StandardCharsets.UTF_8));
        description.put("unscored_candidates", unscored);
        descriptions.add(description); candidates.addAll(part.unscoredCandidates());
        tables.put(tables.size() + 2, part.rows);
      }
    } catch (Json.JsonException error) {
      throw new StructuredDataException("result metadata cannot be encoded", "INTERNAL_ERROR", Map.of(), error);
    }
    Map<String, Object> operation = Map.of("op", "merge_results", "input", 1, "tables", new ArrayList<>(tables.keySet()));
    try (com.nvidia.sdm.internal.Host.Finished finished = com.nvidia.sdm.internal.ArrowFrames.run(operation,
        Map.of(1, com.nvidia.sdm.internal.Host.attachment(Map.of("parts", descriptions))), tables,
        com.nvidia.sdm.internal.Host.Call.none())) {
      com.nvidia.sdm.internal.Results.Predicted result = (com.nvidia.sdm.internal.Results.Predicted)
          com.nvidia.sdm.internal.Results.expect(finished.result(), "predict");
      Object metadata;
      try { metadata = Json.read(finished.bytes(result.metadata())); }
      catch (Json.JsonException error) { throw new IllegalStateException("invalid merged metadata", error); }
      return new PredictionResult(result.model(), result.requestId(), metadata, result.outputs(),
          finished.arrow(result.result()), result.unscoredCandidates().stream().map(i -> candidates.get(Math.toIntExact(i))).toList(), owners);
    }
  }

  /** Candidate values omitted because their sampled context could not teach the question. */
  public List<Object> unscoredCandidates() { return unscoredCandidates; }

  /** The model that answered. */
  public String model() {
    return model;
  }

  /** The endpoint's id for the call, useful when reading its logs. */
  public String requestId() {
    return requestId;
  }

  /** What the endpoint reported beside the predictions, as JSON values. */
  @SuppressWarnings("unchecked")
  public Map<String, Object> metadata() {
    return metadata instanceof Map<?, ?> map ? (Map<String, Object>) map : Map.of();
  }

  /** The outputs asked for. */
  public List<String> outputs() {
    return outputs;
  }

  /**
   * The rows, in a reader whose buffers are {@code allocator}'s: the caller closes it. Each call
   * reads the rows afresh.
   */
  public ArrowReader read(BufferAllocator allocator) {
    return ArrowData.reader(rows, allocator);
  }

  /** Semantic answers in endpoint answer order, with original ranking candidates restored. */
  public List<PredictionItem> items() {
    Map<String, Object> descriptor = new java.util.LinkedHashMap<>();
    descriptor.put("model", model); descriptor.put("request_id", requestId);
    descriptor.put("outputs", outputs); descriptor.put("unscored_candidates", List.of());
    try { descriptor.put("metadata", new String(Json.write(metadata), StandardCharsets.UTF_8)); }
    catch (Json.JsonException error) { throw com.nvidia.sdm.internal.ItemValues.invalid("result metadata cannot be encoded"); }
    var operation = Map.<String, Object>of("op", "read_result_items", "input", 1, "table", 2);
    try (var finished = com.nvidia.sdm.internal.ArrowFrames.run(operation,
        Map.of(1, com.nvidia.sdm.internal.Host.attachment(descriptor)), Map.of(2, rows),
        com.nvidia.sdm.internal.Host.Call.none())) {
      var projected = (com.nvidia.sdm.internal.Results.Items) com.nvidia.sdm.internal.Results.expect(finished.result(), "result_items");
      List<PredictionItem> items = new ArrayList<>();
      long ordinal = 0;
      for (var segment : projected.segments()) {
        try (ArrowReader reader = finished.arrowReader(segment.table(), ArrowData.allocator())) {
          var root = reader.getVectorSchemaRoot();
          if (!root.getSchema().getFields().stream().map(org.apache.arrow.vector.types.pojo.Field::getName).toList()
              .equals(segment.fields().stream().map(com.nvidia.sdm.internal.Results.ItemField::name).toList()))
            throw com.nvidia.sdm.internal.ItemValues.invalid("projected item descriptors differ from Arrow columns");
          while (reader.loadNextBatch()) for (int row = 0; row < root.getRowCount(); row++, ordinal++) {
            Map<String, Object> fields = new java.util.LinkedHashMap<>();
            for (int col = 0; col < segment.fields().size(); col++) {
              var field = segment.fields().get(col);
              Object value = root.getVector(col).getObject(row);
              fields.put(field.name(), itemValue(field.encoding(), root.getSchema().getFields().get(col), value, ordinal));
            }
            items.add(new PredictionItem(fields));
          }
        }
      }
      return List.copyOf(items);
    } catch (IOException error) {
      throw new StructuredDataException("projected result items cannot be read", "INTERNAL_ERROR", Map.of(), error);
    }
  }

  private Object originalCandidate(Object value, long ordinal) {
    if (!(value instanceof Number number)) throw com.nvidia.sdm.internal.ItemValues.invalid("candidate position is not an integer");
    long position;
    try { position = new java.math.BigDecimal(number.toString()).longValueExact(); }
    catch (ArithmeticException error) { throw com.nvidia.sdm.internal.ItemValues.invalid("candidate position is outside integer range"); }
    if (position < 0) throw com.nvidia.sdm.internal.ItemValues.invalid("candidate position is negative");
    for (CandidateOwner owner : candidateOwners)
      if (ordinal >= owner.start() && ordinal < owner.end() && owner.values().containsKey(position))
        return owner.values().get(position);
    throw com.nvidia.sdm.internal.ItemValues.invalid("candidate position has no retained original owner");
  }

  private Object itemValue(String encoding, org.apache.arrow.vector.types.pojo.Field field, Object raw, long ordinal) {
    if (raw == null) return null;
    return switch (encoding) {
      case "native" -> com.nvidia.sdm.internal.ItemValues.nativeValue(field, raw);
      case "value" -> com.nvidia.sdm.internal.ItemValues.primitive(raw.toString());
      case "mapping" -> com.nvidia.sdm.internal.ItemValues.mapping(com.nvidia.sdm.internal.ItemValues.primitive(raw.toString()));
      case "mappings" -> ((List<?>) com.nvidia.sdm.internal.ItemValues.primitive(raw.toString())).stream()
          .map(com.nvidia.sdm.internal.ItemValues::mapping).toList();
      case "candidate_list" -> ((List<?>) raw).stream().map(v -> originalCandidate(v, ordinal)).toList();
      case "candidate_rankings" -> ((List<?>) raw).stream().map(v -> {
        if (!(v instanceof Map<?, ?> entry)) throw com.nvidia.sdm.internal.ItemValues.invalid("candidate ranking is not a struct");
        Map<Object, Object> ranked = new java.util.LinkedHashMap<>();
        ranked.put("candidate", originalCandidate(entry.get("position"), ordinal)); ranked.put("score", entry.get("score"));
        return ranked;
      }).toList();
      default -> throw com.nvidia.sdm.internal.ItemValues.invalid("unknown projected item encoding");
    };
  }

  /** How many rows the answer holds. */
  public long rows() {
    long count = 0;
    try (ArrowReader reader = read(ArrowData.allocator())) {
      while (reader.loadNextBatch()) {
        count += reader.getVectorSchemaRoot().getRowCount();
      }
    } catch (IOException e) {
      throw new StructuredDataException(
          "the result's rows cannot be read", "INTERNAL_ERROR", Map.of(), null);
    }
    return count;
  }

  /**
   * The explanation, when it was asked for and the model gave one; null otherwise, as Python's
   * {@code explanation}.
   */
  public Map<String, Object> explanation() {
    try (ArrowReader reader = read(ArrowData.allocator())) {
      Map<String, String> metadata = reader.getVectorSchemaRoot().getSchema().getCustomMetadata();
      String text = metadata == null ? null : metadata.get("explanation");
      if (text == null) {
        return null;
      }
      @SuppressWarnings("unchecked")
      Map<String, Object> read =
          (Map<String, Object>) Json.read(text.getBytes(StandardCharsets.UTF_8));
      return read;
    } catch (IOException | Json.JsonException | ClassCastException e) {
      throw new StructuredDataException(
          "the result's explanation cannot be read", "INTERNAL_ERROR", Map.of(), null);
    }
  }

  /**
   * Which of {@code requested} the answer does not carry, as Python's {@code absent()}, each once,
   * in the order named: the core's reading of the answer, which the result names beside its rows
   * (every output it looks for that no row holds, the explanation where none was given, nothing
   * for an answer of no rows); a name it does not look for is taken as carried.
   */
  public List<String> absent(List<String> requested) {
    Set<String> leftOut = new LinkedHashSet<>(leftOut());
    List<String> missing = new ArrayList<>();
    for (String name : new LinkedHashSet<>(requested)) {
      if (leftOut.contains(name)) {
        missing.add(name);
      }
    }
    return missing;
  }

  /** The outputs the core read the answer as leaving out, from the result's schema. */
  private List<String> leftOut() {
    try (ArrowReader reader = read(ArrowData.allocator())) {
      Map<String, String> named = reader.getVectorSchemaRoot().getSchema().getCustomMetadata();
      String text = named == null ? null : named.get("absent");
      if (text == null) {
        throw new StructuredDataException(
            "the result does not say which outputs the answer left out", "INTERNAL_ERROR",
            Map.of(), null);
      }
      List<String> read = new ArrayList<>();
      for (Object each : (List<?>) Json.read(text.getBytes(StandardCharsets.UTF_8))) {
        read.add((String) each);
      }
      return read;
    } catch (IOException | Json.JsonException | ClassCastException e) {
      throw new StructuredDataException(
          "the result's outputs left out cannot be read", "INTERNAL_ERROR", Map.of(), null);
    }
  }

  @Override
  public String toString() {
    return "PredictionResult(" + model + ", " + rows() + " rows)";
  }
}
