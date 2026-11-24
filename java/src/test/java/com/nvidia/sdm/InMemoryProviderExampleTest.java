// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;

/**
 * The complete Java example of {@code docs/reference/data-providers.md}: a provider serving two
 * in-memory tables, answering {@code describe} and {@code scan} and leaving every other request to
 * the client's in-memory fallback, used to build a schema and materialize a query.
 */
class InMemoryProviderExampleTest {

  /** One in-memory table: its columns, by name, each a list of values, in order. */
  record Table(Map<String, ArrowType> types, Map<String, List<Object>> columns) {
    int rows() {
      return columns.values().iterator().next().size();
    }
  }

  static final LocalDateTime START = LocalDateTime.of(2024, 1, 1, 0, 0);
  static final ArrowType INT = new ArrowType.Int(64, true);
  static final ArrowType FLOAT = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
  static final ArrowType TIME = new ArrowType.Timestamp(TimeUnit.MICROSECOND, null);

  static final Map<String, Table> TABLES = tables();

  static Map<String, Table> tables() {
    Map<String, Table> tables = new LinkedHashMap<>();
    tables.put(
        "users",
        new Table(
            Map.of("id", INT, "age", INT),
            ordered(Map.of("id", List.of(1L, 2L, 3L), "age", List.of(25L, 40L, 35L)), "id", "age")));
    List<Object> orderId = new ArrayList<>(), userId = new ArrayList<>();
    List<Object> amount = new ArrayList<>(), placed = new ArrayList<>();
    for (int i = 0; i < 60; i++) {
      orderId.add((long) i);
      userId.add((long) (i % 3 + 1));
      amount.add((double) i);
      placed.add(START.plusDays(6L * i));
    }
    tables.put(
        "orders",
        new Table(
            Map.of("order_id", INT, "user_id", INT, "amount", FLOAT, "placed", TIME),
            ordered(
                Map.of("order_id", orderId, "user_id", userId, "amount", amount, "placed", placed),
                "order_id",
                "user_id",
                "amount",
                "placed")));
    return tables;
  }

  /** The source: {@code describe} and {@code scan}, and a fallback for everything else. */
  static final class InMemorySource implements Source {
    @Override
    public Capabilities capabilities() {
      // The client answers every request not declared here in memory, from scans.
      return Capabilities.builder().fallback(10_000_000, 1L << 31).build();
    }

    @Override
    public ArrowReader answer(
        DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
      if (request instanceof DataRequest.Describe describe)
        return describe(describe, expected.orElseThrow(), allocator);
      if (request instanceof DataRequest.Scan scan) return scan(scan, allocator);
      throw new UnsupportedOperationException(request.name());
    }

    private ArrowReader describe(
        DataRequest.Describe request, Schema expected, BufferAllocator allocator) throws Exception {
      List<String> names =
          request.tables().isEmpty()
              ? List.copyOf(TABLES.keySet())
              : request.tables().stream().map(DataRequest.Relation::table).toList();
      List<List<Object>> rows = new ArrayList<>();
      switch (request.part()) {
        case TABLES -> names.forEach(name -> rows.add(List.of(name)));
        case COLUMNS -> {
          Table table = TABLES.get(names.get(0));
          for (String column : table.columns().keySet())
            rows.add(List.of(column, table.types().get(column).toString(), false, false));
        }
        default -> {} // No keys, foreign keys or namespace: the schema names them.
      }
      return arrow(expected, rows, allocator);
    }

    private ArrowReader scan(DataRequest.Scan request, BufferAllocator allocator) throws Exception {
      Table table = TABLES.get(request.table().table());
      List<Field> fields = new ArrayList<>();
      for (String column : request.columns())
        fields.add(new Field(column, FieldType.nullable(table.types().get(column)), null));
      List<List<Object>> rows = new ArrayList<>();
      for (int at = 0; at < table.rows(); at++) {
        final int row = at;
        if (!request.filters().stream().allMatch(f -> holds(table, row, f))) continue;
        rows.add(request.columns().stream().map(c -> table.columns().get(c).get(row)).toList());
        if (request.limit().isPresent() && rows.size() == request.limit().getAsLong()) break;
      }
      return arrow(new Schema(fields), rows, allocator);
    }

    /** The conditions every provider applies; value filters are sent only where declared. */
    private boolean holds(Table table, int row, DataRequest.Filter filter) {
      if (filter instanceof DataRequest.NotNull f)
        return table.columns().get(f.column()).get(row) != null;
      if (filter instanceof DataRequest.KeyIn f) {
        Object key = table.columns().get(f.columns().get(0)).get(row);
        return keys(f.keys()).contains(key);
      }
      if (filter instanceof DataRequest.TimeRange f) {
        LocalDateTime at = (LocalDateTime) table.columns().get(f.column()).get(row);
        Predicate<DataRequest.Bound> above =
            b -> b.inclusive() ? !at.isBefore(wall(b.value().value())) : at.isAfter(wall(b.value().value()));
        Predicate<DataRequest.Bound> below =
            b -> b.inclusive() ? !at.isAfter(wall(b.value().value())) : at.isBefore(wall(b.value().value()));
        return at != null
            && f.lower().map(above::test).orElse(true)
            && f.upper().map(below::test).orElse(true);
      }
      throw new UnsupportedOperationException(filter.getClass().getSimpleName());
    }
  }

  @Test
  void theDocumentedProviderAnswersAQuery() throws Exception {
    Source source = new InMemorySource();
    var schema =
        com.nvidia.sdm.Schema.fromSource(
                source,
                FromSourceOptions.builder()
                    .table("users")
                    .table("orders")
                    .primaryKey("users", "id")
                    .primaryKey("orders", "order_id")
                    .timeColumn("orders", "placed")
                    .build())
            .withRelationships(List.of(new Relationship("orders", "user_id", "users", "id")));
    Request request =
        PredictiveQuery.parse(
                "PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.id WHERE users.age > 30")
            .materialize(schema, MaterializeOptions.builder().indices(List.of(2L, 3L)).build());
    assertEquals("available", request.contextReport().status());
    assertEquals(2, request.predictRows());
  }

  // --- Arrow helpers -------------------------------------------------------------------------

  static Map<String, List<Object>> ordered(Map<String, List<Object>> columns, String... order) {
    Map<String, List<Object>> out = new LinkedHashMap<>();
    for (String name : order) out.put(name, columns.get(name));
    return out;
  }

  static LocalDateTime wall(Object value) {
    return value instanceof java.time.Instant instant
        ? LocalDateTime.ofInstant(instant, ZoneOffset.UTC)
        : (LocalDateTime) value;
  }

  static List<Object> keys(DataRequest.KeySet keys) {
    List<Object> values = new ArrayList<>();
    try (ArrowReader reader = keys.reader()) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        for (int at = 0; at < root.getRowCount(); at++) values.add(root.getVector(0).getObject(at));
      }
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
    return values;
  }

  /** {@code rows} of {@code schema} as an Arrow stream the client reads and closes. */
  static ArrowReader arrow(Schema schema, List<List<Object>> rows, BufferAllocator allocator)
      throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
        ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bytes)) {
      root.allocateNew();
      for (int at = 0; at < rows.size(); at++)
        for (int c = 0; c < schema.getFields().size(); c++)
          set(root.getVector(c), at, rows.get(at).get(c));
      root.setRowCount(rows.size());
      writer.start();
      writer.writeBatch();
      writer.end();
    }
    return new ArrowStreamReader(new ByteArrayInputStream(bytes.toByteArray()), allocator);
  }

  static void set(FieldVector vector, int at, Object value) {
    if (value == null) vector.setNull(at);
    else if (vector instanceof BigIntVector v) v.setSafe(at, ((Number) value).longValue());
    else if (vector instanceof Float8Vector v) v.setSafe(at, ((Number) value).doubleValue());
    else if (vector instanceof BitVector v) v.setSafe(at, (Boolean) value ? 1 : 0);
    else if (vector instanceof TimeStampMicroVector v) {
      LocalDateTime time = (LocalDateTime) value;
      v.setSafe(at, time.toEpochSecond(ZoneOffset.UTC) * 1_000_000L + time.getNano() / 1000);
    } else if (vector instanceof VarCharVector v)
      v.setSafe(at, value.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
    else throw new IllegalStateException("no writer for " + vector.getField());
  }
}
