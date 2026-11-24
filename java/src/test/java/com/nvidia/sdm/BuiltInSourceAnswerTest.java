// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.HostProviders;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.ProviderFailure;
import java.sql.Connection;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;

/**
 * A built-in source is asked through the same {@code capabilities()} and {@code answer()} as any
 * source: asked directly, it answers each request of a run exactly what the core received from it
 * in the run, as Python's {@code test_a_built_in_source_answers_directly_what_a_run_receives}.
 */
class BuiltInSourceAnswerTest {

  /** One request the core asked in a run, its key sets and the answer, each as IPC bytes. */
  private record Asked(
      Map<String, Object> wire,
      Map<Integer, byte[]> keys,
      Optional<org.apache.arrow.vector.types.pojo.Schema> expected,
      byte[] answer) {}

  /**
   * A caller's source handing every request to {@code source}, keeping each request, its key sets
   * and the answer the core received.
   */
  private static final class Recording implements Source {
    private final Source source;
    private final List<Asked> asked = new ArrayList<>();

    Recording(Source source) {
      this.source = source;
    }

    @Override
    public Capabilities capabilities() {
      return source.capabilities();
    }

    @Override
    public ArrowReader answer(
        DataRequest request,
        Optional<org.apache.arrow.vector.types.pojo.Schema> expected,
        BufferAllocator allocator)
        throws Exception {
      Map<Integer, ArrowReader> readers = new LinkedHashMap<>();
      Map<String, Object> wire = HostProviders.Requests.write(request, readers);
      Map<Integer, byte[]> keys = new LinkedHashMap<>();
      readers.forEach((id, reader) -> keys.put(id, ArrowData.ipc(reader)));
      byte[] answer = answered(source, wire, keys, expected, allocator);
      asked.add(new Asked(wire, keys, expected, answer));
      return ArrowData.reader(answer, allocator);
    }

    @Override
    public ProviderFailure failure(Exception failure) {
      return source.failure(failure);
    }
  }

  @Test
  void aBuiltInSourceAnswersDirectlyWhatARunReceives() throws Exception {
    try (Connection connection = Shop.open("duckdb");
        BufferAllocator allocator = new RootAllocator()) {
      DuckDbSource source = DuckDbSource.of(connection);
      Recording recording = new Recording(source);
      Schema schema =
          Schema.fromSource(recording)
              .withRelationships(List.of(new Relationship("orders", "user_id", "users", "user_id")))
              .withTimeColumn("orders", "ts");
      PredictiveQuery.parse("PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id")
          .materialize(
              schema,
              MaterializeOptions.builder()
                  .indices(List.of(0L, 1L))
                  .anchorTime(LocalDate.of(2024, 3, 1))
                  .build());
      Set<String> kinds = new TreeSet<>();
      for (Asked asked : recording.asked) kinds.add((String) asked.wire().get("request"));
      assertTrue(kinds.containsAll(List.of("describe", "neighbors", "time_bounds")), kinds.toString());

      // The same requests, asked directly, in a fresh operation.
      assertEquals(
          HostProviders.readCapabilities(capabilitiesOf(source)), source.capabilities());
      for (Asked asked : recording.asked) {
        // The library's own allocator, as an operation answers with: the source's SQL kit keeps
        // the operation's catalog reads, its driver's Arrow among them, for the operation.
        byte[] direct =
            answered(source, asked.wire(), asked.keys(), asked.expected(), ArrowData.allocator());
        assertEquals(
            rows(asked.answer(), allocator),
            rows(direct, allocator),
            asked.wire().toString());
      }
    }
  }

  @Test
  void aBuiltInSourceRefusesByNameWhatItsSqlCannotAnswer() throws Exception {
    try (Connection connection = Shop.open("duckdb");
        BufferAllocator allocator = new RootAllocator()) {
      DuckDbSource source = DuckDbSource.of(connection);
      // A scan of no column is no statement DuckDB's SQL writes.
      DataRequest scan =
          new DataRequest.Scan(
              new DataRequest.Relation(List.of("users")),
              List.of(),
              List.of(),
              java.util.OptionalLong.empty());
      StructuredDataException refused =
          assertThrows(
              StructuredDataException.class, () -> source.answer(scan, Optional.empty(), allocator));
      assertEquals("INVALID_REQUEST", refused.code());
    }
  }

  /**
   * {@code source}'s answer to the request {@code wire} names, its key sets read from {@code keys},
   * as IPC bytes; each key set's reader closed, as the library closes those it hands a source.
   */
  private static byte[] answered(
      Source source,
      Map<String, Object> wire,
      Map<Integer, byte[]> keys,
      Optional<org.apache.arrow.vector.types.pojo.Schema> expected,
      BufferAllocator allocator)
      throws Exception {
    List<ArrowReader> opened = new ArrayList<>();
    try {
      DataRequest request =
          HostProviders.Requests.read(
              wire,
              id -> {
                ArrowReader reader = ArrowData.reader(keys.get(id), allocator);
                opened.add(reader);
                return reader;
              });
      return ArrowData.ipc(source.answer(request, expected, allocator));
    } finally {
      for (ArrowReader reader : opened) reader.close();
    }
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> capabilitiesOf(Source source) {
    return (Map<String, Object>) HostProviders.chosen(source).choice().get("capabilities");
  }

  /** The schema and the rows of {@code ipc}, the rows in no order. */
  private static List<Object> rows(byte[] ipc, BufferAllocator allocator) throws Exception {
    List<String> rows = new ArrayList<>();
    String schema;
    try (ArrowReader reader = ArrowData.reader(ipc, allocator)) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      schema = root.getSchema().toString();
      while (reader.loadNextBatch()) {
        for (int at = 0; at < root.getRowCount(); at++) {
          List<Object> row = new ArrayList<>();
          for (FieldVector vector : root.getFieldVectors()) row.add(vector.getObject(at));
          rows.add(row.toString());
        }
      }
    }
    rows.sort(null);
    return List.of(schema, rows);
  }
}
