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

  PredictionResult(
      String model, String requestId, Object metadata, List<String> outputs, byte[] rows) {
    this.model = model;
    this.requestId = requestId;
    this.metadata = metadata;
    this.outputs = List.copyOf(outputs);
    this.rows = rows;
  }

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
