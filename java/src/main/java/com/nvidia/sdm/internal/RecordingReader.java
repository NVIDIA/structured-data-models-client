// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * A source's answer, read through to the reader it answered with, keeping the first exception that
 * reader threw while the core read it: a lazy reader fails as its batches are read, after {@code
 * answer} returned, and the core sees only that the stream failed. The run's error carries the
 * exception kept here as it carries one {@code answer} threw itself. Not part of the API.
 */
final class RecordingReader extends ArrowReader {
  private final ArrowReader answer;
  private Exception failure;

  RecordingReader(BufferAllocator allocator, ArrowReader answer) {
    super(allocator);
    this.answer = answer;
  }

  /** The first exception the answer threw while it was read; null where it threw none. */
  Exception failure() {
    return failure;
  }

  private void keep(Exception thrown) {
    if (failure == null) {
      failure = thrown;
    }
  }

  @Override
  public VectorSchemaRoot getVectorSchemaRoot() throws IOException {
    try {
      return answer.getVectorSchemaRoot();
    } catch (IOException | RuntimeException e) {
      keep(e);
      throw e;
    }
  }

  @Override
  public boolean loadNextBatch() throws IOException {
    try {
      return answer.loadNextBatch();
    } catch (IOException | RuntimeException e) {
      keep(e);
      throw e;
    }
  }

  @Override
  public Map<Long, Dictionary> getDictionaryVectors() throws IOException {
    return answer.getDictionaryVectors();
  }

  @Override
  public Dictionary lookup(long id) {
    return answer.lookup(id);
  }

  @Override
  public Set<Long> getDictionaryIds() {
    return answer.getDictionaryIds();
  }

  @Override
  public long bytesRead() {
    return answer.bytesRead();
  }

  @Override
  public void close() throws IOException {
    answer.close();
  }

  @Override
  public void close(boolean closeReadSource) throws IOException {
    answer.close(closeReadSource);
  }

  @Override
  protected void closeReadSource() throws IOException {
    answer.close();
  }

  @Override
  protected Schema readSchema() throws IOException {
    return getVectorSchemaRoot().getSchema();
  }
}
