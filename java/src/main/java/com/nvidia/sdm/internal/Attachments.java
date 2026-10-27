// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;

/**
 * The attachments of a message the host sends, as the native library takes them: parallel arrays
 * of ids, bytes (null for an Arrow one) and the addresses of the Arrow ones' stream structs (0 for
 * a bytes one). Not part of the API.
 */
final class Attachments {
  private final int[] ids;
  private final byte[][] bytes;
  private final long[] streams;

  private Attachments(int size) {
    ids = new int[size];
    bytes = new byte[size][];
    streams = new long[size];
  }

  /** {@code attachments}, each a {@code byte[]} or an {@link ArrowArrayStream}, by id. */
  static Attachments of(Map<Integer, ?> attachments) {
    Attachments read = new Attachments(attachments.size());
    int at = 0;
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
      } else {
        throw new IllegalArgumentException(
            "attachment "
                + id
                + " is "
                + (value == null ? "null" : "a " + value.getClass().getName())
                + ": an attachment is a byte[] or an ArrowArrayStream");
      }
      at++;
    }
    return read;
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
}
