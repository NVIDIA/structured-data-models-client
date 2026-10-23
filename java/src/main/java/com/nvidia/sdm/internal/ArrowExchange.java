// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * Arrow across the binding through Arrow Java's C Data module, with no copy, and the named error
 * for a JVM started without the flag Arrow Java needs: {@code
 * --add-opens=java.base/java.nio=ALL-UNNAMED} (design choice 10, "Runtime flags"), which Arrow
 * reports only as an {@link UnsupportedOperationException} of its own internals. Not part of the
 * API.
 */
public final class ArrowExchange {
  /** The flag Arrow Java's memory module needs. */
  public static final String FLAG = "--add-opens=java.base/java.nio=ALL-UNNAMED";

  private ArrowExchange() {}

  /**
   * {@code reader}'s data exported into a new stream struct, an Arrow attachment of {@link
   * Run#start} or {@link Run#reply}; the caller closes the struct after the call.
   */
  public static ArrowArrayStream export(BufferAllocator allocator, ArrowReader reader) {
    ArrowArrayStream stream = null;
    try {
      stream = ArrowArrayStream.allocateNew(allocator);
      Data.exportArrayStream(allocator, reader, stream);
      return stream;
    } catch (UnsupportedOperationException e) {
      if (stream != null) {
        stream.close();
      }
      throw flagMissing(e);
    }
  }

  /** The data the native library exports into a new stream struct, as a reader. */
  static ArrowReader importing(BufferAllocator allocator, long handle) {
    try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
      NativeCore.exportArrow(handle, stream.memoryAddress());
      return Data.importArrayStream(allocator, stream);
    } catch (UnsupportedOperationException e) {
      throw flagMissing(e);
    }
  }

  static StructuredDataException flagMissing(Throwable cause) {
    return new StructuredDataException(
        "Arrow Java cannot reach native memory: start the JVM with " + FLAG,
        "INVALID_CONFIGURATION",
        Map.of("flag", FLAG),
        cause);
  }
}
