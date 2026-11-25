// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.table.Table;

/** Arrow transport and owned request snapshots, with no model policy. */
public final class ArrowTabular {
  private ArrowTabular() {}

  /**
   * Snapshot a Table without consuming it. Arrow 19 exportTable transfers its vectors;
   * public vector copies plus IPC also preserve dictionaries and require no public owner class.
   */
  public static byte[] snapshot(Table table) {
    java.util.Objects.requireNonNull(table, "table");
    List<FieldVector> vectors = new ArrayList<>();
    try {
      for (int at = 0; at < table.getVectorCount(); at++) vectors.add(table.getVectorCopy(at));
      try (VectorSchemaRoot root = new VectorSchemaRoot(table.getSchema(), vectors,
          Math.toIntExact(table.getRowCount()))) {
        vectors = new ArrayList<>(); // root now owns every copied vector
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (ArrowStreamWriter writer = new ArrowStreamWriter(root,
            table.getDictionaryProvider(), Channels.newChannel(out))) {
          writer.start();
          writer.writeBatch();
          writer.end();
        }
        return out.toByteArray();
      }
    } catch (IOException | ArithmeticException e) {
      throw new StructuredDataException("Arrow table cannot be imported", "INVALID_REQUEST",
          Map.of(), e);
    } finally {
      for (FieldVector vector : vectors) vector.close();
    }
  }

  public static Host.Finished run(String action, Object input, Map<String, byte[]> tables,
      Host.Model model) {
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "tabular_arrow");
    operation.put("action", action);
    operation.put("input", 1);
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(input));
    List<ArrowReader> readers = new ArrayList<>();
    List<ArrowArrayStream> streams = new ArrayList<>();
    try {
      for (Map.Entry<String, byte[]> entry : tables.entrySet()) {
        int id = attachments.size() + 1;
        ArrowReader reader = ArrowData.reader(entry.getValue(), ArrowData.allocator());
        readers.add(reader);
        ArrowArrayStream stream = ArrowExchange.export(ArrowData.allocator(), reader);
        streams.add(stream);
        operation.put(entry.getKey(), id);
        attachments.put(id, stream);
      }
      return Host.run(operation, attachments, new Host.Call(null, model, Map.of()));
    } finally {
      for (ArrowArrayStream stream : streams) stream.close();
      for (ArrowReader reader : readers) {
        try { reader.close(); } catch (IOException e) { throw new IllegalStateException(e); }
      }
    }
  }

  @SuppressWarnings("unchecked")
  public static Map<String, Object> value(Host.Finished finished) {
    Object id = finished.result().get("value");
    if (!(id instanceof Number number)) throw new IllegalStateException("missing core value");
    try { return (Map<String, Object>) Json.read(finished.bytes(number.intValue())); }
    catch (Json.JsonException e) { throw new IllegalStateException("invalid core value", e); }
  }

  public static Map<String, Object> request(String target, String task, String table, Object schema) {
    Map<String, Object> value = new LinkedHashMap<>();
    value.put("target", target); value.put("task", task);
    value.put("entity_table", table); value.put("schema", schema);
    return value;
  }

  public static Map<String, Object> key(Object value) {
    if (value == null) return Map.of("kind", "missing");
    if (value instanceof String s) return Map.of("kind", "spelling", "value", s);
    if (value instanceof Boolean b) return Map.of("kind", "boolean", "value", b);
    if (value instanceof Number n) return Map.of("kind", "numeric", "decimal", n.toString());
    throw new StructuredDataException("unsupported entity key", "INVALID_REQUEST", Map.of(), null);
  }
}
