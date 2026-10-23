// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorLoader;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Arrow data in the host: the one allocator the library's Arrow lives in, an in-memory reader
 * over batches the host built, and an Arrow IPC stream's bytes, which the contract runner's
 * listener is told ({@link Trace}) and a result keeps. Not part of the API.
 */
public final class ArrowData {
  private ArrowData() {}

  /**
   * The allocator every Arrow buffer the host makes or imports comes from. It lives as long as
   * the library: an exported stream's buffers are the core's until it releases them, which may be
   * after the call that handed them over returns, so no shorter-lived allocator could be closed.
   */
  public static BufferAllocator allocator() {
    return Root.ROOT;
  }

  private static final class Root {
    static final BufferAllocator ROOT = new RootAllocator();

    private Root() {}
  }

  /** {@code reader}'s batches as the bytes of one Arrow IPC stream; the reader is closed. */
  public static byte[] ipc(ArrowReader reader) {
    try (reader) {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      try (ArrowStreamWriter writer =
          new ArrowStreamWriter(root, null, Channels.newChannel(out))) {
        writer.start();
        while (reader.loadNextBatch()) {
          writer.writeBatch();
        }
        writer.end();
      }
      return out.toByteArray();
    } catch (IOException e) {
      throw Messages.internal("Arrow data could not be read whole: " + e.getMessage());
    }
  }

  /**
   * A reader of the Arrow IPC stream {@code bytes}, its buffers from {@code allocator}, each
   * buffer an allocation of its own: an IPC stream lays a batch's buffers out in one body at
   * 8-byte offsets, which is no alignment a 128-bit value (a decimal) may be read at across the C
   * Data interface, so each is copied to its own, which the allocator aligns.
   */
  public static ArrowReader reader(byte[] bytes, BufferAllocator allocator) {
    return new Aligned(new ArrowStreamReader(new ByteArrayInputStream(bytes), allocator), allocator);
  }

  /** A reader handing over another's batches, each buffer copied to an allocation of its own. */
  private static final class Aligned extends ArrowReader {
    private final ArrowReader source;

    Aligned(ArrowReader source, BufferAllocator allocator) {
      super(allocator);
      this.source = source;
    }

    @Override
    public boolean loadNextBatch() throws IOException {
      if (!source.loadNextBatch()) {
        return false;
      }
      try (ArrowRecordBatch read =
          new VectorUnloader(source.getVectorSchemaRoot()).getRecordBatch()) {
        List<ArrowBuf> copies = new ArrayList<>();
        try {
          for (ArrowBuf buffer : read.getBuffers()) {
            long size = buffer.readableBytes();
            ArrowBuf copy = allocator.buffer(Math.max(size, 1));
            copy.setBytes(0, buffer, 0, size);
            copy.writerIndex(size);
            copies.add(copy);
          }
          try (ArrowRecordBatch aligned =
              new ArrowRecordBatch(read.getLength(), read.getNodes(), copies)) {
            new VectorLoader(getVectorSchemaRoot()).load(aligned);
          }
        } finally {
          copies.forEach(ArrowBuf::close);
        }
      }
      return true;
    }

    @Override
    public long bytesRead() {
      return source.bytesRead();
    }

    @Override
    protected void closeReadSource() throws IOException {
      source.close();
    }

    @Override
    protected Schema readSchema() throws IOException {
      return source.getVectorSchemaRoot().getSchema();
    }
  }

  /**
   * A reader handing over {@code batches}, each a root of {@code schema}, which it owns and closes:
   * the host's own data (a key set it read, a result it built) as a stream to export.
   */
  public static ArrowReader reader(
      Schema schema, List<VectorSchemaRoot> batches, BufferAllocator allocator) {
    return new Batches(schema, batches, allocator);
  }

  /** An in-memory reader of batches the host built. */
  private static final class Batches extends ArrowReader {
    private final Schema schema;
    private final List<VectorSchemaRoot> batches;
    private int next;

    Batches(Schema schema, List<VectorSchemaRoot> batches, BufferAllocator allocator) {
      super(allocator);
      this.schema = schema;
      this.batches = new ArrayList<>(batches);
    }

    @Override
    public boolean loadNextBatch() throws IOException {
      if (next >= batches.size()) {
        return false;
      }
      VectorSchemaRoot batch = batches.get(next++);
      try (ArrowRecordBatch record = new VectorUnloader(batch).getRecordBatch()) {
        new VectorLoader(getVectorSchemaRoot()).load(record);
      }
      return true;
    }

    @Override
    public long bytesRead() {
      return 0;
    }

    @Override
    protected void closeReadSource() {
      for (VectorSchemaRoot batch : batches) {
        batch.close();
      }
      batches.clear();
    }

    @Override
    protected Schema readSchema() {
      return schema;
    }
  }
}
