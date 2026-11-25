// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.DictionaryEncoder;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Table;

/** Collect host reader batches into one independently owned public table. */
public final class ReadTables {
  private ReadTables() {}

  /** Consumes and closes the reader, including on conversion failure. */
  public static Table collect(ArrowReader reader, BufferAllocator allocator) throws IOException {
    List<FieldVector> owned = new ArrayList<>();
    int size = 0;
    try (reader) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      for (FieldVector vector : root.getFieldVectors()) {
        var encoding = vector.getField().getDictionary();
        var field = encoding == null ? vector.getField()
            : reader.lookup(encoding.getId()).getVector().getField();
        owned.add(new org.apache.arrow.vector.types.pojo.Field(vector.getName(),
            new org.apache.arrow.vector.types.pojo.FieldType(field.isNullable(), field.getType(),
                null, field.getMetadata()), field.getChildren()).createVector(allocator));
      }
      while (reader.loadNextBatch()) {
        int next = Math.addExact(size, root.getRowCount());
        for (int column = 0; column < owned.size(); column++) {
          FieldVector source = root.getVector(column);
          var encoding = source.getField().getDictionary();
          FieldVector decoded = encoding == null ? source
              : (FieldVector) DictionaryEncoder.decode(source, reader.lookup(encoding.getId()));
          try {
            for (int row = 0; row < root.getRowCount(); row++) {
              owned.get(column).copyFromSafe(row, size + row, decoded);
            }
          } finally {
            if (decoded != source) decoded.close();
          }
        }
        size = next;
      }
      for (FieldVector vector : owned) vector.setValueCount(size);
      return new Table(owned, size);
    } finally {
      // Table transfers each vector's buffers; these wrappers are empty after successful return.
      owned.forEach(FieldVector::close);
    }
  }
}
