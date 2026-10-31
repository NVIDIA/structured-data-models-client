// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Json;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Versioned data-only framing; schema and prediction policy stay in the core. */
final class Snapshots {
  private Snapshots() {}
  private static final byte[] MAGIC = {'S', 'D', 'M', 'S', 'N', 'A', 'P', 0};
  private static final int VERSION = 1;

  static final class Writer {
    private final IdentityHashMap<byte[], Integer> ids = new IdentityHashMap<>();
    private final List<byte[]> blobs = new ArrayList<>();
    int blob(byte[] bytes) {
      Integer present = ids.get(bytes);
      if (present != null) return present;
      int id = blobs.size();
      blobs.add(bytes); ids.put(bytes, id);
      return id;
    }
    void write(Map<String, Object> manifest, OutputStream output) throws IOException {
      byte[] metadata;
      try { metadata = Json.write(manifest); }
      catch (Json.JsonException failure) { throw invalid("snapshot metadata cannot be encoded", failure); }
      DataOutputStream data = new DataOutputStream(java.util.Objects.requireNonNull(output, "output"));
      data.write(MAGIC); data.writeInt(VERSION);
      data.writeInt(metadata.length); data.write(metadata); data.writeInt(blobs.size());
      for (int id = 0; id < blobs.size(); id++) {
        byte[] bytes = blobs.get(id);
        data.writeInt(id); data.writeLong(bytes.length); data.write(bytes);
      }
      // No close/flush: the caller owns buffering and stream lifetime.
    }
  }

  static final class Reader {
    final Map<String, Object> manifest;
    private final Map<Integer, byte[]> blobs;
    private final java.util.Set<Integer> used = new java.util.HashSet<>();
    Reader(Map<String, Object> manifest, Map<Integer, byte[]> blobs) {
      this.manifest = manifest; this.blobs = blobs;
    }
    byte[] blob(Object value) throws IOException {
      int id = integer(value);
      byte[] bytes = blobs.get(id);
      if (bytes == null) throw invalid("snapshot refers to a missing attachment", null);
      used.add(id);
      return bytes;
    }
    void complete() throws IOException {
      if (used.size() != blobs.size()) throw invalid("snapshot contains an unreferenced attachment", null);
    }
  }

  static Reader read(InputStream input) throws IOException {
    DataInputStream data = new DataInputStream(java.util.Objects.requireNonNull(input, "input"));
    try {
      if (!java.util.Arrays.equals(readBytes(data, MAGIC.length), MAGIC)) throw invalid("not an SDM snapshot", null);
      if (data.readInt() != VERSION) throw invalid("unsupported SDM snapshot version", null);
      int length = data.readInt();
      Map<String, Object> manifest = object(Json.read(readBytes(data, length)));
      int count = data.readInt();
      if (count < 0) throw invalid("negative attachment count", null);
      Map<Integer, byte[]> blobs = new LinkedHashMap<>();
      for (int at = 0; at < count; at++) {
        int id = data.readInt();
        if (id < 0 || id >= count || blobs.containsKey(id)) throw invalid("duplicate or invalid attachment id", null);
        long size = data.readLong();
        if (size < 0 || size > Integer.MAX_VALUE) throw invalid("attachment exceeds supported Java array size", null);
        blobs.put(id, readBytes(data, (int) size));
      }
      return new Reader(manifest, blobs);
    } catch (Json.JsonException failure) { throw invalid("invalid snapshot metadata JSON", failure); }
    catch (EOFException failure) { throw invalid("truncated SDM snapshot", failure); }
  }

  private static byte[] readBytes(DataInputStream input, int length) throws IOException {
    if (length < 0) throw invalid("negative snapshot section length", null);
    // readNBytes grows only as input arrives, rather than allocating a declared huge missing blob.
    byte[] bytes = input.readNBytes(length);
    if (bytes.length != length) throw new EOFException("truncated snapshot section");
    return bytes;
  }

  @SuppressWarnings("unchecked")
  static Map<String, Object> object(Object value) throws IOException {
    if (!(value instanceof Map<?, ?> map) || map.keySet().stream().anyMatch(key -> !(key instanceof String))) {
      throw invalid("snapshot field must be an object", null);
    }
    return (Map<String, Object>) map;
  }
  static List<?> list(Object value) throws IOException {
    if (!(value instanceof List<?> list)) throw invalid("snapshot field must be a list", null);
    return list;
  }
  static String text(Object value) throws IOException {
    if (!(value instanceof String text)) throw invalid("snapshot field must be text", null);
    return text;
  }
  static List<String> texts(Object value) throws IOException {
    List<String> result = new ArrayList<>();
    for (Object item : list(value)) result.add(text(item));
    return List.copyOf(result);
  }
  static int integer(Object value) throws IOException {
    try {
      if (value instanceof Long number) return Math.toIntExact(number);
      if (value instanceof Integer number) return number;
      if (value instanceof java.math.BigInteger number) return number.intValueExact();
    } catch (ArithmeticException failure) { throw invalid("snapshot integer is out of range", failure); }
    throw invalid("snapshot field must be an integer", null);
  }
  static IOException invalid(String message, Throwable cause) { return new IOException(message, cause); }

  static void validateArrow(byte[] bytes) throws IOException {
    try (var allocator = new org.apache.arrow.memory.RootAllocator();
        var reader = com.nvidia.sdm.internal.ArrowData.reader(bytes, allocator)) {
      reader.getVectorSchemaRoot();
      while (reader.loadNextBatch()) { /* Fully validate every IPC batch without retaining another table. */ }
    } catch (RuntimeException failure) { throw invalid("invalid Arrow snapshot attachment", failure); }
  }
}
