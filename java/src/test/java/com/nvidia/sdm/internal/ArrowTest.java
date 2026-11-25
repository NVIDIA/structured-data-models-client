// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Arrow across the binding, both ways, through the C Data interface: data a Java provider exports
 * reaches the core, and the core's comes back, through the test-only fetch that echoes what it
 * admitted. Every allocator closes with nothing held, so every buffer either side lent was
 * released.
 */
class ArrowTest {
  @BeforeEach
  void testOps() {
    TestOps.required();
  }

  private static final Schema SCHEMA =
      new Schema(
          List.of(
              new Field("id", FieldType.nullable(new ArrowType.Int(64, true)), null),
              new Field("name", FieldType.nullable(new ArrowType.Utf8()), null)));

  /** A reader of {@code batches} batches of {@code rows} rows each. */
  private static ArrowReader reader(BufferAllocator allocator, int batches, int rows) {
    return new ArrowReader(allocator) {
      private int read;

      @Override
      public boolean loadNextBatch() throws java.io.IOException {
        if (read == batches) {
          return false;
        }
        VectorSchemaRoot root = getVectorSchemaRoot();
        BigIntVector id = (BigIntVector) root.getVector("id");
        VarCharVector name = (VarCharVector) root.getVector("name");
        id.allocateNew(rows);
        name.allocateNew(rows);
        for (int i = 0; i < rows; i++) {
          long value = (long) read * rows + i;
          if (value % 7 == 3) {
            id.setNull(i);
            name.setNull(i);
          } else {
            id.set(i, value);
            name.setSafe(i, ("n" + value).getBytes(java.nio.charset.StandardCharsets.UTF_8));
          }
        }
        root.setRowCount(rows);
        read++;
        return true;
      }

      @Override
      public long bytesRead() {
        return 0;
      }

      @Override
      protected void closeReadSource() {}

      @Override
      protected Schema readSchema() {
        return SCHEMA;
      }
    };
  }

  /** Sends {@code batches} of {@code rows} through the echoing fetch; the rows read back. */
  private static long roundTrip(int batches, int rows) throws Exception {
    long seen = 0;
    try (BufferAllocator allocator = new RootAllocator();
        Run run =
            Run.start(
                TestOps.start("fetch", Map.of("fetch", 1)),
                Map.of(1, TestOps.json(TestOps.fetch(true))))) {
      try (NativeMessage step = run.next()) {
        assertEquals("fetch", Messages.read(step).get("step"));
      }
      try (ArrowReader sent = reader(allocator, batches, rows);
          ArrowArrayStream stream = ArrowExchange.export(allocator, sent)) {
        run.reply(TestOps.fetchReply(1), Map.of(1, stream));
      }
      Map<String, Object> done;
      try (NativeMessage step = run.next()) {
        done = Messages.read(step);
      }
      @SuppressWarnings("unchecked")
      Map<String, Object> result = (Map<String, Object>) done.get("result");
      assertEquals((long) batches * rows, ((Number) result.get("rows")).longValue());
      try (NativeArrow echoed = (NativeArrow) run.takeAttachment(1);
          ArrowReader back = echoed.reader(allocator)) {
        long expected = 0;
        while (back.loadNextBatch()) {
          VectorSchemaRoot root = back.getVectorSchemaRoot();
          BigIntVector id = (BigIntVector) root.getVector("id");
          for (int i = 0; i < root.getRowCount(); i++, expected++) {
            if (expected % 7 == 3) {
              assertEquals(true, id.isNull(i));
            } else {
              assertEquals(expected, id.get(i));
            }
          }
          seen += root.getRowCount();
        }
      }
    }
    return seen;
  }

  @Test
  void dataCrossesBothWaysAndEveryBufferIsReleased() throws Exception {
    assertEquals(3 * 1000, roundTrip(3, 1000));
  }

  @Test
  void anEmptyStreamCrosses() throws Exception {
    assertEquals(0, roundTrip(0, 0));
  }

  @Test
  void aLargeStreamCrosses() throws Exception {
    assertEquals(20 * 100_000, roundTrip(20, 100_000));
  }

  @Test
  void aClosedArrowAttachmentIsNotRead() throws Exception {
    try (BufferAllocator allocator = new RootAllocator();
        Run run =
            Run.start(
                TestOps.start("fetch", Map.of("fetch", 1)),
                Map.of(1, TestOps.json(TestOps.fetch(true))))) {
      run.next().close();
      try (ArrowReader sent = reader(allocator, 1, 10);
          ArrowArrayStream stream = ArrowExchange.export(allocator, sent)) {
        run.reply(TestOps.fetchReply(1), Map.of(1, stream));
      }
      run.next().close();
      NativeArrow echoed = (NativeArrow) run.takeAttachment(1);
      echoed.close();
      echoed.close();
      assertThrows(IllegalStateException.class, () -> echoed.reader(allocator));
    }
  }
}
