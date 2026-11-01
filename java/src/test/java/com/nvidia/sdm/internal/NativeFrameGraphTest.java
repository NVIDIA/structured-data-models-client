// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

/** Native graph ownership remains valid after the producing run and original owner close. */
class NativeFrameGraphTest {
  private static NativeFrameGraph graph() throws Exception {
    byte[] data;
    try (RootAllocator allocator = new RootAllocator()) {
      BigIntVector id = new BigIntVector("id", allocator);
      Float8Vector value = new Float8Vector("value", allocator);
      id.allocateNew(3);
      value.allocateNew(3);
      for (int at = 0; at < 3; at++) {
        id.setSafe(at, at);
        value.setSafe(at, 2.5 * at);
      }
      id.setValueCount(3);
      value.setValueCount(3);
      try (Table table = new Table(List.of(id, value))) {
        data = ArrowTabular.snapshot(table);
      }
    }
    Map<String, Object> schema;
    try (Host.Finished described = ArrowTabular.run("describe", Map.of("entity_table", "rows"),
        Map.of("table", data), null)) {
      schema = ArrowTabular.value(described);
    }
    try (ArrowReader reader = ArrowData.reader(data, ArrowData.allocator());
        ArrowArrayStream stream = ArrowExchange.export(ArrowData.allocator(), reader)) {
      Map<String, Object> operation = Map.of("op", "prepare_frame_graph", "schema", 1,
          "names", 2, "tables", List.of(3));
      try (Host.Finished prepared = Host.run(operation,
          Map.of(1, Host.attachment(schema), 2, Host.attachment(List.of("rows")), 3, stream),
          Host.Call.none())) {
        return prepared.graph(((Number) prepared.result().get("graph")).intValue());
      }
    }
  }

  @Test
  void closingTheOwnerTwiceRejectsFurtherAdmission() throws Exception {
    NativeFrameGraph graph = graph();
    graph.close();
    graph.close();
    assertThrows(IllegalStateException.class, () -> Attachments.of(Map.of(1, graph)));
  }

  @Test
  void admittedLeaseOutlivesTheOriginalOwner() throws Exception {
    NativeFrameGraph graph = graph();
    try (Attachments lease = Attachments.of(Map.of(1, graph))) {
      graph.close();
      long retained = NativeCore.retainFrameGraph(lease.graphs()[0]);
      try (NativeFrameGraph nextOwner = new NativeFrameGraph(retained);
          Attachments nextLease = Attachments.of(Map.of(2, nextOwner))) {
        assertTrue(nextLease.graphs()[0] != 0);
      }
    } finally {
      graph.close();
    }
  }

  @Test
  void failedAttachmentAdmissionDoesNotConsumeTheOriginalOwner() throws Exception {
    try (NativeFrameGraph graph = graph()) {
      Map<Integer, Object> invalid = new LinkedHashMap<>();
      invalid.put(1, graph);
      invalid.put(2, new Object());
      assertThrows(IllegalArgumentException.class, () -> Attachments.of(invalid));
      try (Attachments valid = Attachments.of(Map.of(1, graph))) {
        assertTrue(valid.graphs()[0] != 0);
      }
    }
  }
}
