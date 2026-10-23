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
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.BaseListVector;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * The answer to one request, as Python's {@code PredictionResult}: the model that answered, the
 * endpoint's id for the call, its metadata, the outputs asked for, and the rows, read as an Arrow
 * {@link VectorSchemaRoot} in the layout the core gives them ({@link #read}): one row per answer,
 * ordered by {@code row_index}, under the entity it is about ({@code id}), with its prediction and
 * the other outputs it carries. Join on {@code id}, or on {@code row_index}, never on row order.
 */
public final class PredictionResult {
  /** The outputs a row carries under a column of its own name. */
  private static final Set<String> ITEM_FIELDS =
      Set.of("prediction", "probabilities", "quantiles", "scores", "rankings", "embeddings");

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
   * Which of {@code requested} the answer does not carry, as Python's {@code absent()}: a field
   * no row holds a value of, the explanation where none was given; nothing for an answer of no
   * rows, and a name this does not look for is taken as carried.
   */
  public List<String> absent(List<String> requested) {
    List<String> missing = new ArrayList<>();
    if (rows() == 0) {
      return missing;
    }
    for (String name : new LinkedHashSet<>(requested)) {
      if ("explanation".equals(name)) {
        if (explanation() == null) {
          missing.add(name);
        }
      } else if (ITEM_FIELDS.contains(name) && !carries(name)) {
        missing.add(name);
      }
    }
    return missing;
  }

  private boolean carries(String name) {
    try (ArrowReader reader = read(ArrowData.allocator())) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        FieldVector column = root.getVector(name);
        if (column == null) {
          return false;
        }
        for (int row = 0; row < root.getRowCount(); row++) {
          if (column.isNull(row)) {
            continue;
          }
          if (column instanceof BaseListVector list
              && list.getElementEndIndex(row) == list.getElementStartIndex(row)) {
            continue;
          }
          return true;
        }
      }
      return false;
    } catch (IOException e) {
      throw new StructuredDataException(
          "the result's rows cannot be read", "INTERNAL_ERROR", Map.of(), null);
    }
  }

  @Override
  public String toString() {
    return "PredictionResult(" + model + ", " + rows() + " rows)";
  }
}
