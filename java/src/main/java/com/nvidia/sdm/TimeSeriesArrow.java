// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowData;
import java.time.*;
import java.util.*;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.table.Table;
import org.apache.arrow.vector.types.pojo.*;

/** Timestamp-list adapter avoiding Arrow19's unsupported timezone-list copier. */
final class TimeSeriesArrow {
  private TimeSeriesArrow() {}

  static byte[] snapshot(Table table, String timestamps) {
    if (timestamps == null) return com.nvidia.sdm.internal.ArrowTabular.snapshot(table);
    List<FieldVector> vectors = new ArrayList<>();
    try {
      for (int col = 0; col < table.getVectorCount(); col++) {
        Field field = table.getSchema().getFields().get(col);
        if (!field.getName().equals(timestamps)
            || field.getChildren().size() != 1
            || !(field.getChildren().get(0).getType() instanceof ArrowType.Timestamp)) {
          vectors.add(table.getVectorCopy(col));
          continue;
        }
        ArrowType.Timestamp type = (ArrowType.Timestamp) field.getChildren().get(0).getType();
        Field normalized =
            new Field(timestamps, field.getFieldType(), List.of(field.getChildren().get(0)));
        normalized =
            new Field(
                timestamps, FieldType.nullable(ArrowType.List.INSTANCE), normalized.getChildren());
        ListVector copy = (ListVector) normalized.createVector(ArrowData.allocator());
        vectors.add(copy);
        copy.allocateNew();
        var child = (TimeStampVector) copy.getDataVector();
        var reader = table.getReader(col);
        for (int row = 0; row < table.getRowCount(); row++) {
          reader.setPosition(row);
          if (!reader.isSet()) {
            copy.setNull(row);
            continue;
          }
          Object read = reader.readObject();
          if (!(read instanceof List<?> values))
            throw new IllegalArgumentException("timestamps must be lists");
          int offset = copy.startNewValue(row);
          for (int i = 0; i < values.size(); i++) {
            Object value = values.get(i);
            while (offset + i >= child.getValueCapacity()) child.reAlloc();
            if (value == null) {
              child.setNull(offset + i);
              continue;
            }
            if (value instanceof Long encoded) {
              child.setSafe(offset + i, encoded);
              continue;
            }
            Instant instant =
                value instanceof ZonedDateTime zoned
                    ? zoned.toInstant()
                    : ((LocalDateTime) value).toInstant(ZoneOffset.UTC);
            long encoded =
                switch (type.getUnit()) {
                  case SECOND -> instant.getEpochSecond();
                  case MILLISECOND -> instant.toEpochMilli();
                  case MICROSECOND ->
                      Math.addExact(
                          Math.multiplyExact(instant.getEpochSecond(), 1_000_000L),
                          instant.getNano() / 1000);
                  case NANOSECOND ->
                      Math.addExact(
                          Math.multiplyExact(instant.getEpochSecond(), 1_000_000_000L),
                          instant.getNano());
                };
            child.setSafe(offset + i, encoded);
          }
          copy.endValue(row, values.size());
        }
        copy.setValueCount(Math.toIntExact(table.getRowCount()));
      }
      try (var root = new VectorSchemaRoot(vectors)) {
        vectors = new ArrayList<>();
        var bytes = new java.io.ByteArrayOutputStream();
        try (var writer =
            new org.apache.arrow.vector.ipc.ArrowStreamWriter(
                root,
                table.getDictionaryProvider(),
                java.nio.channels.Channels.newChannel(bytes))) {
          writer.start();
          writer.writeBatch();
          writer.end();
        }
        return bytes.toByteArray();
      }
    } catch (java.io.IOException
        | IllegalArgumentException
        | ArithmeticException
        | ClassCastException
        | UnsupportedOperationException error) {
      throw new StructuredDataException(
          "time-series input cannot be represented as Arrow", "INVALID_REQUEST", Map.of(), error);
    } finally {
      for (var vector : vectors) vector.close();
    }
  }
}
