// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.vector.ipc.ArrowReader;

/** Owns temporary Arrow import resources while a core operation consumes snapshots. */
public final class ArrowFrames {
  private ArrowFrames() {}

  public static Host.Finished run(Map<String, Object> operation, Map<Integer, Object> supplied,
      Map<Integer, byte[]> tables, Host.Call call) {
    Map<Integer, Object> attachments = new LinkedHashMap<>(supplied);
    List<ArrowReader> readers = new ArrayList<>();
    List<ArrowArrayStream> streams = new ArrayList<>();
    Throwable failure = null;
    try {
      for (Map.Entry<Integer, byte[]> table : tables.entrySet()) {
        ArrowReader reader = ArrowData.reader(table.getValue(), ArrowData.allocator());
        readers.add(reader);
        ArrowArrayStream stream = ArrowExchange.export(ArrowData.allocator(), reader);
        streams.add(stream);
        attachments.put(table.getKey(), stream);
      }
      return Host.run(operation, attachments, call);
    } catch (RuntimeException | Error e) {
      failure = e;
      throw e;
    } finally {
      RuntimeException cleanup = null;
      List<AutoCloseable> owned = new ArrayList<>(streams);
      owned.addAll(readers);
      for (AutoCloseable resource : owned) {
        try { resource.close(); }
        catch (Exception e) {
          if (failure != null) failure.addSuppressed(e);
          else if (cleanup == null) cleanup = new IllegalStateException("could not release Arrow import", e);
          else cleanup.addSuppressed(e);
        }
      }
      if (cleanup != null) throw cleanup;
    }
  }
}
