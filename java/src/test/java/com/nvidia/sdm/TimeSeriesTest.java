// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.time.*;
import java.util.*;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.table.Table;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.*;
import org.junit.jupiter.api.Test;

class TimeSeriesTest {
  private static ListVector lists(
      RootAllocator allocator, String name, List<?> rows, boolean timestamps) {
    Field field =
        new Field(
            name,
            FieldType.nullable(ArrowType.List.INSTANCE),
            List.of(
                new Field(
                    "item",
                    FieldType.nullable(
                        timestamps
                            ? new ArrowType.Timestamp(TimeUnit.NANOSECOND, "UTC")
                            : new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                    null)));
    ListVector vector = (ListVector) field.createVector(allocator);
    vector.allocateNew();
    FieldVector child = vector.getDataVector();
    for (int row = 0; row < rows.size(); row++) {
      if (rows.get(row) == null) {
        vector.setNull(row);
        continue;
      }
      List<?> values = (List<?>) rows.get(row);
      int start = vector.startNewValue(row);
      for (int i = 0; i < values.size(); i++) {
        Object value = values.get(i);
        if (value == null) child.setNull(start + i);
        else if (timestamps)
          ((TimeStampNanoTZVector) child).setSafe(start + i, ((Number) value).longValue());
        else ((Float8Vector) child).setSafe(start + i, ((Number) value).doubleValue());
      }
      vector.endValue(row, values.size());
    }
    vector.setValueCount(rows.size());
    return vector;
  }

  @Test
  void sharedPublicHistoriesAndIndependentOwnership() throws Exception {
    List<?> fixtures =
        (List<?>)
            com.nvidia.sdm.internal.Json.read(
                java.nio.file.Files.readAllBytes(
                    java.nio.file.Path.of("../tests/fixtures/timeseries/cases.json")));
    for (Object raw : fixtures) {
      Map<?, ?> c = (Map<?, ?>) raw;
      String name = c.get("name").toString();
      ForecastInput made;
      List<?> histories = (List<?>) c.get("histories");
      try (RootAllocator allocator = new RootAllocator();
          BigIntVector ids = new BigIntVector("id", allocator);
          ListVector values = lists(allocator, "sales", histories, false)) {
        ids.allocateNew();
        for (int i = 0; i < histories.size(); i++) ids.setSafe(i, i + 1);
        ids.setValueCount(histories.size());
        List<FieldVector> columns = new ArrayList<>(List.of(ids, values));
        ListVector times =
            c.get("timestamps") == null
                ? null
                : lists(allocator, "times", (List<?>) c.get("timestamps"), true);
        if (times != null) columns.add(times);
        try (Table table = new Table(columns, histories.size())) {
          var options = TimeSeriesOptions.builder("sales").entityColumn("id").numTimeframes(2);
          if (times != null) options.timestampsColumn("times");
          if (c.get("step_minutes") != null)
            options.timeDelta(Duration.ofMinutes(((Number) c.get("step_minutes")).longValue()));
          if (c.get("anchor_ns") != null)
            options.anchorTime(Instant.ofEpochSecond(0, ((Number) c.get("anchor_ns")).longValue()));
          if (Boolean.TRUE.equals(c.get("error"))) {
            assertEquals(
                "INVALID_REQUEST",
                assertThrows(
                        StructuredDataException.class,
                        () -> TimeSeries.build(table, options.build()),
                        name)
                    .code());
            continue;
          }
          made = TimeSeries.build(table, options.build());
          assertEquals(histories.size(), table.getRowCount());
        } finally {
          if (times != null) times.close();
        }
      }
      assertEquals(List.of("id"), made.schema().table("entity").primaryKey());
      assertEquals(List.of(), made.schema().table("target").primaryKey());
      assertTrue(made.query().text().contains("FORECAST 2 TIMEFRAMES"));
      try (var allocator = new RootAllocator();
          var reader =
              com.nvidia.sdm.internal.ArrowData.reader(
                  made.schema().data().get("target"), allocator)) {
        assertTrue(reader.loadNextBatch());
        var root = reader.getVectorSchemaRoot();
        List<?> expected = (List<?>) c.get("expected_values");
        assertEquals(expected.size(), root.getRowCount(), name);
        for (int i = 0; i < expected.size(); i++) {
          assertEquals(
              ((Number) ((List<?>) c.get("expected_ids")).get(i)).longValue(),
              ((BigIntVector) root.getVector("id")).get(i),
              name);
          assertEquals(
              ((Number) ((List<?>) c.get("expected_times")).get(i)).longValue(),
              ((TimeStampNanoTZVector) root.getVector("timestamp")).get(i),
              name);
          if (expected.get(i) == null) assertTrue(root.getVector("value").isNull(i), name);
          else
            assertEquals(
                ((Number) expected.get(i)).doubleValue(),
                ((Float8Vector) root.getVector("value")).get(i),
                name);
        }
      }
    }
  }

  @Test
  void rejectsInvalidOptionsAndPreservesInput() {
    try (var allocator = new RootAllocator();
        var values = lists(allocator, "sales", List.of(List.of(1., 2.)), false);
        var table = new Table(List.of(values), 1)) {
      for (var options :
          List.of(
              TimeSeriesOptions.builder("sales").numTimeframes(0).build(),
              TimeSeriesOptions.builder("sales")
                  .timeDelta(Duration.ofSeconds(60).plusNanos(1))
                  .anchorTime(Instant.now())
                  .build(),
              TimeSeriesOptions.builder("sales")
                  .timeDelta(Duration.ofDays(1))
                  .anchorTime(Instant.MIN)
                  .build()))
        assertEquals(
            "INVALID_REQUEST",
            assertThrows(StructuredDataException.class, () -> TimeSeries.build(table, options))
                .code());
      ForecastInput made =
          TimeSeries.build(
              table,
              TimeSeriesOptions.builder("sales")
                  .timeDelta(Duration.ofDays(1))
                  .anchorTime(Instant.parse("2024-01-03T00:00:00Z"))
                  .build());
      assertEquals(List.of("entity_id"), made.schema().table("entity").primaryKey());
      assertEquals(1, table.getRowCount());
    }
  }

  @Test
  void longNullTimestampListsFailWithoutLeaking() {
    try (var allocator = new RootAllocator();
        var values = lists(allocator, "sales", List.of(Collections.nCopies(12000, 1.0)), false)) {
      Field child =
          new Field(
              "item",
              FieldType.nullable(new ArrowType.Timestamp(TimeUnit.NANOSECOND, "UTC")),
              null);
      try (var times =
          (ListVector)
              new Field("times", FieldType.nullable(ArrowType.List.INSTANCE), List.of(child))
                  .createVector(allocator)) {
        times.allocateNew();
        int offset = times.startNewValue(0);
        var data = (TimeStampNanoTZVector) times.getDataVector();
        for (int i = 0; i < 12000; i++) {
          while (i >= data.getValueCapacity()) data.reAlloc();
          data.setNull(i);
        }
        times.endValue(0, 12000);
        times.setValueCount(1);
        try (var table = new Table(List.of(values, times), 1)) {
          assertEquals(
              "INVALID_REQUEST",
              assertThrows(
                      StructuredDataException.class,
                      () ->
                          TimeSeries.build(
                              table,
                              TimeSeriesOptions.builder("sales")
                                  .timestampsColumn("times")
                                  .timeDelta(Duration.ofDays(1))
                                  .build()))
                  .code());
        }
      }
    }
  }
}
