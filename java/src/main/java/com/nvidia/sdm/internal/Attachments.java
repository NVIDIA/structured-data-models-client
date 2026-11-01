// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;

/**
 * The attachments of a message the host sends, as the native library takes them: parallel arrays
 * of ids and exactly one of bytes, Arrow stream addresses or temporary graph leases per id.
 * Graph leases are released after native admission, including when a call fails. Not public API.
 */
final class Attachments implements AutoCloseable {
  private final int[] ids;
  private final byte[][] bytes;
  private final long[] streams;
  private final long[] graphs;

  private Attachments(int size) {
    ids = new int[size];
    bytes = new byte[size][];
    streams = new long[size];
    graphs = new long[size];
  }

  /** Bytes, Arrow streams or retained frame graphs, by id; leases last through the JNI call. */
  static Attachments of(Map<Integer, ?> attachments) {
    Attachments read = new Attachments(attachments.size());
    int at = 0;
    try {
      for (Map.Entry<Integer, ?> each : attachments.entrySet()) {
        Integer id = each.getKey();
        Object value = each.getValue();
        if (id == null || id < 0) {
          throw new IllegalArgumentException("an attachment's id is a non-negative integer");
        }
        read.ids[at] = id;
        if (value instanceof byte[] data) {
          read.bytes[at] = data;
        } else if (value instanceof ArrowArrayStream stream) {
          read.streams[at] = stream.memoryAddress();
        } else if (value instanceof NativeFrameGraph graph) {
          read.graphs[at] = graph.retainHandle();
        } else {
          throw new IllegalArgumentException(
              "attachment "
                  + id
                  + " is "
                  + (value == null ? "null" : "a " + value.getClass().getName())
                  + ": an attachment is bytes, an Arrow stream or a native frame graph");
        }
        at++;
      }
      return read;
    } catch (RuntimeException | Error failure) {
      read.close();
      throw failure;
    }
  }

  int[] ids() {
    return ids;
  }

  byte[][] bytes() {
    return bytes;
  }

  long[] streams() {
    return streams;
  }

  long[] graphs() {
    return graphs;
  }

  @Override
  public void close() {
    for (int at = 0; at < graphs.length; at++) {
      long graph = graphs[at];
      graphs[at] = 0;
      if (graph != 0) NativeCore.releaseFrameGraph(graph);
    }
  }
}
