// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static com.nvidia.sdm.ParityRunner.*;

import com.nvidia.sdm.internal.Json;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.apache.arrow.memory.RootAllocator;

/** Inspect actual public request data and actual uploaded holdout context. */
final class ParityHoldout {
  private ParityHoldout() {}

  static Map<String, Object> payload(Request request) throws Exception {
    return map(Json.read(request.payloads().get(0).getBytes(StandardCharsets.UTF_8)));
  }

  static Map<String, Object> instance(Map<String, Object> payload) {
    return map(map(payload.get("context")).get("instance_table"));
  }

  static String entity(Map<String, Object> payload) {
    var links =
        list(map(payload.get("schema")).get("relationships")).stream()
            .map(ParityRunner::map)
            .filter(
                link ->
                    !link.containsKey("source_table") && "users".equals(link.get("target_table")))
            .toList();
    if (links.size() != 1) throw new IllegalStateException("expected one entity link");
    return text(list(links.get(0).get("source_columns")).get(0));
  }

  static Object key(Request request, RootAllocator allocator) throws Exception {
    if (!request.payloads().isEmpty()) {
      var payload = payload(request);
      var table = instance(payload);
      return list(list(table.get("rows")).get(0))
          .get(list(table.get("columns")).indexOf(entity(payload)));
    }
    try (var reader = request.context().read(allocator)) {
      if (!reader.loadNextBatch() || reader.getVectorSchemaRoot().getRowCount() == 0)
        throw new IllegalStateException("empty holdout context");
      var root = reader.getVectorSchemaRoot();
      // Tabular fixture has its explicit entity key as a feature as well as the wire index.
      return scalar(root.getVector("id").getObject(0));
    }
  }

  static void check(Request request, List<?> held, List<Map<String, Object>> calls)
      throws Exception {
    if (!calls.stream()
        .map(call -> call.get("operation"))
        .toList()
        .equals(List.of("fit", "score", "score", "close")))
      throw new IllegalStateException("unexpected holdout lifecycle calls");
    var fitted = map(calls.get(0).get("payload"));
    var table = instance(fitted);
    if (!request.payloads().isEmpty()) {
      var original = payload(request);
      var prior = instance(original);
      int entityAt = list(prior.get("columns")).indexOf(entity(original));
      int idAt = list(prior.get("columns")).indexOf("instance_id");
      var removed = new HashSet<Object>();
      for (Object row : list(prior.get("rows")))
        if (held.contains(list(row).get(entityAt))) removed.add(list(row).get(idAt));
      if (removed.isEmpty()) throw new IllegalStateException("holdout removed no context");
      for (Object row : list(table.get("rows")))
        if (held.contains(list(row).get(entityAt)))
          throw new IllegalStateException("held key retained");
      if (list(prior.get("rows")).size() - list(table.get("rows")).size() != removed.size())
        throw new IllegalStateException("wrong holdout count");
      for (Object related : map(map(fitted.get("context")).get("related_tables")).values()) {
        var value = map(related);
        int at = list(value.get("columns")).indexOf("instance_id");
        for (Object row : list(value.get("rows")))
          if (removed.contains(list(row).get(at)))
            throw new IllegalStateException("held related row retained");
      }
    } else {
      Object primary = map(map(fitted.get("schema")).get("instance_table")).get("primary_key");
      String name = primary instanceof List<?> keys ? text(keys.get(0)) : text(primary);
      int at = list(table.get("columns")).indexOf(name);
      for (Object row : list(table.get("rows")))
        for (Object key : held)
          if (text(key).equals(text(list(row).get(at))))
            throw new IllegalStateException("held Tabular key retained");
      if (request.contextRows() - list(table.get("rows")).size() != held.size())
        throw new IllegalStateException("wrong Tabular holdout count");
    }
  }
}
