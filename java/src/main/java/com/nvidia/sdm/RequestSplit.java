// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Results;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * A lightweight view of a request's instance rows and named related tables. Every read returns an
 * independent reader owned by the caller. Keep its allocator alive until that reader closes.
 * Reading never changes the request, samples data, or calls a source/model.
 */
public final class RequestSplit {
  private final Request request;
  private final String half;

  RequestSplit(Request request, String half) {
    this.request = request;
    this.half = half;
  }

  /** Read the instance table, retaining the request's column and row order. */
  public ArrowReader read(BufferAllocator allocator) {
    Objects.requireNonNull(allocator, "allocator");
    if (request.direct())
      return ArrowData.reader(
          half.equals("context") ? request.contextData() : request.predictData(), allocator);
    return selected(Map.of("kind", "instance"), null, allocator);
  }

  /** Related table names in their first appearance order; no Arrow rows are allocated. */
  public List<String> relatedTables() {
    if (request.direct()) return List.of();
    try (Host.Finished finished = run(Map.of("kind", "names"))) {
      return result(finished).relatedTables().stream().map(Results.SplitTable::name).toList();
    }
  }

  /** Read one related table; unknown names are INVALID_REQUEST. */
  public ArrowReader readRelated(String name, BufferAllocator allocator) {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(allocator, "allocator");
    if (request.direct())
      throw new StructuredDataException(
          "a direct Tabular request has no related tables", "INVALID_REQUEST", Map.of(), null);
    return selected(Map.of("kind", "related", "name", name), name, allocator);
  }

  private ArrowReader selected(
      Map<String, Object> selection, String name, BufferAllocator allocator) {
    try (Host.Finished finished = run(selection)) {
      Results.Split split = result(finished);
      Integer id =
          name == null
              ? split.instance()
              : split.relatedTables().stream()
                  .filter(t -> t.name().equals(name))
                  .map(Results.SplitTable::data)
                  .findFirst()
                  .orElse(null);
      if (id == null)
        throw new StructuredDataException(
            "core omitted the selected request table", "INTERNAL_ERROR", Map.of(), null);
      return finished.arrowReader(id, allocator);
    }
  }

  private Results.Split result(Host.Finished finished) {
    return (Results.Split) Results.expect(finished.result(), "request_split");
  }

  private Host.Finished run(Map<String, Object> selection) {
    Map<Integer, byte[]> attachments = new LinkedHashMap<>();
    for (byte[] payload : request.payloadBytes()) attachments.put(attachments.size() + 1, payload);
    return Host.run(
        Map.of(
            "op",
            "read_request_split",
            "payloads",
            List.copyOf(attachments.keySet()),
            "half",
            half,
            "selection",
            selection),
        attachments);
  }
}
