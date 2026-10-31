// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import com.nvidia.sdm.internal.Json;
import java.io.*;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

class PersistenceTest {
  private static byte[] save(Schema schema) throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream(); schema.writeTo(out); return out.toByteArray();
  }
  private static byte[] save(Request request) throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream(); request.writeTo(out); return out.toByteArray();
  }
  private static Request restore(Request request) throws Exception {
    return Request.readFrom(new ByteArrayInputStream(save(request)));
  }

  @Test void frameSchemaRestoresDataKeysTimesRelationshipsAndCanPredict() throws Exception {
    Schema original = RelationalTest.fixture(true, true);
    Schema restored = Schema.readFrom(new ByteArrayInputStream(save(original)));
    assertEquals(original.toHeld(), restored.toHeld());
    assertEquals(original.data().keySet(), restored.data().keySet());
    original.data().forEach((name, bytes) -> assertArrayEquals(bytes, restored.data().get(name)));
    assertEquals(original.conversionNotes(), restored.conversionNotes());
    Request request = PredictiveQuery.parse(RelationalTest.SCALAR).materialize(restored, RelationalTest.options(1L, 3L).build());
    try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint(); SdmClient client = endpoint.client()) {
      assertEquals(2, client.predict(request, SdmClient.KUMO_RELATIONAL).rows());
    }
  }

  @Test void directRequestRestoresOptionsAndOwnsDataAfterStreamsAndTablesClose() throws Exception {
    Request original;
    try (RootAllocator allocator = new RootAllocator(); Table context = TabularTest.table(allocator, true, 1, 2);
        Table scoring = TabularTest.table(allocator, false, 3)) {
      original = new Request(context, scoring, "y", "regression", RequestOptions.builder()
          .inference(Map.of("wide", new java.math.BigInteger("18446744073709551617"), "temperature", .5))
          .taskOptions(Map.of("unused", List.of("a", 7))).build());
    }
    Request restored = restore(original);
    assertEquals(original.inference(), restored.inference()); assertEquals(original.taskOptions(), restored.taskOptions());
    assertArrayEquals(original.contextData(), restored.contextData());
    assertArrayEquals(original.predictData(), restored.predictData());
    assertEquals(original.schema().toHeld(), restored.schema().toHeld());
    assertThrows(UnsupportedOperationException.class, () -> restored.inference().put("new", 2));
    try (TabularTest.Endpoint endpoint = new TabularTest.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      assertEquals(1, client.predict(restored, SdmClient.KUMO_TABULAR).rows());
      try (FittedModel fitted = client.fit(restored, SdmClient.KUMO_TABULAR)) {
        assertEquals(1, fitted.predict(restored).rows());
      }
    }
  }

  @Test void materializedTabularPreservesQueryAndCanPredict() throws Exception {
    Request original;
    try (RootAllocator allocator = new RootAllocator(); Table table = TabularTest.table(allocator, true, 1, 2, 3)) {
      original = PredictiveQuery.parse("PREDICT data.y FOR EACH data.id").materialize(
          Schema.fromData(Map.of("data", table)), MaterializeOptions.builder().indices(3L).build());
    }
    Request restored = restore(original);
    assertEquals(original.query(), restored.query()); assertEquals(original.entityTable(), restored.entityTable());
    try (TabularTest.Endpoint endpoint = new TabularTest.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      assertEquals(1, client.predict(restored, SdmClient.KUMO_TABULAR).rows());
    }
  }

  @Test void gatheredRelationalRestoresExactPayloadsPartsAndPredictsWithoutSampling() throws Exception {
    Schema schema = RelationalTest.fixture(true, true);
    for (String query : List.of(RelationalTest.SCALAR, RelationalTest.FORECAST, RelationalTest.RANK)) {
      var options = RelationalTest.options(1L, 3L);
      if (!query.equals(RelationalTest.SCALAR)) options.anchorTime(LocalDate.of(2024, 5, 1));
      if (query.equals(RelationalTest.RANK)) options.candidates(List.of(1L, 2L, 3L));
      Request original = PredictiveQuery.parse(query).materialize(schema, options.build());
      Request restored = restore(original);
      assertEquals(original.payloads(), restored.payloads()); assertEquals(original.query(), restored.query());
      assertEquals(original.asked(), restored.asked());
      assertEquals(original.parts().map(List::size), restored.parts().map(List::size));
      assertEquals(original.contextRows(), restored.contextRows()); assertEquals(original.predictRows(), restored.predictRows());
      try (RelationalTest.Endpoint endpoint = new RelationalTest.Endpoint(); SdmClient client = endpoint.client()) {
        assertEquals(client.predict(original, SdmClient.KUMO_RELATIONAL).rows(), client.predict(restored, SdmClient.KUMO_RELATIONAL).rows());
        if (query.equals(RelationalTest.SCALAR)) {
          try (FittedModel fitted = client.fit(restored, SdmClient.KUMO_RELATIONAL)) { assertEquals(2, fitted.predict(restored).rows()); }
        }
      }
    }
  }

  @Test void streamsRemainOpenAndSnapshotsCanBeConcatenated() throws Exception {
    Schema schema = RelationalTest.fixture(false, false);
    class Output extends ByteArrayOutputStream { boolean closed; @Override public void close() { closed = true; } }
    class Input extends ByteArrayInputStream { boolean closed; Input(byte[] bytes) { super(bytes); } @Override public void close() { closed = true; } }
    Output output = new Output(); schema.writeTo(output); schema.writeTo(output); assertFalse(output.closed);
    Input input = new Input(output.toByteArray());
    assertEquals(schema.toHeld(), Schema.readFrom(input).toHeld());
    assertEquals(schema.toHeld(), Schema.readFrom(input).toHeld()); assertFalse(input.closed); assertEquals(-1, input.read());
    Input broken = new Input(new byte[]{1,2,3}); assertThrows(IOException.class, () -> Schema.readFrom(broken)); assertFalse(broken.closed);
  }

  @Test void malformedFramingMissingBlobsInvalidArrowAndStateVersionsFailBeforeHttp() throws Exception {
    Request direct;
    try (RootAllocator allocator = new RootAllocator(); Table context = TabularTest.table(allocator, true, 1, 2);
        Table scoring = TabularTest.table(allocator, false, 3)) { direct = new Request(context, scoring, "y", "regression"); }
    byte[] bytes = save(direct);
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(java.util.Arrays.copyOf(bytes, bytes.length - 1))));
    byte[] version = bytes.clone(); java.nio.ByteBuffer.wrap(version).putInt(8, 999);
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(version)));
    byte[] huge = bytes.clone(); java.nio.ByteBuffer.wrap(huge).putInt(12, Integer.MAX_VALUE);
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(huge)));
    byte[] disagreement = change(bytes, manifest -> manifest.put("inference", Map.of("unexpected", 1)));
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(disagreement)));
    byte[] missing = change(bytes, manifest -> manifest.put("context", 999));
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(missing)));
    int manifestLength = java.nio.ByteBuffer.wrap(bytes).getInt(12);
    int firstEntry = 16 + manifestLength + 4;
    byte[] negative = bytes.clone(); java.nio.ByteBuffer.wrap(negative).putLong(firstEntry + 4, -1L);
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(negative)));
    int secondEntry = Math.toIntExact(firstEntry + 12L + java.nio.ByteBuffer.wrap(bytes).getLong(firstEntry + 4));
    byte[] duplicate = bytes.clone(); java.nio.ByteBuffer.wrap(duplicate).putInt(secondEntry, 0);
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(duplicate)));
    byte[] invalidArrow = bytes.clone();
    int firstBlob = 16 + manifestLength + 4 + 4 + 8;
    java.util.Arrays.fill(invalidArrow, firstBlob, firstBlob + 8, (byte) 0);
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(invalidArrow)));
    Request relational = PredictiveQuery.parse(RelationalTest.SCALAR).materialize(RelationalTest.fixture(false, false), RelationalTest.options(1L).build());
    byte[] unsupported = change(save(relational), manifest -> {
      @SuppressWarnings("unchecked") Map<String,Object> state = (Map<String,Object>) manifest.get("state"); state.put("version", 999);
    });
    assertThrows(IOException.class, () -> Request.readFrom(new ByteArrayInputStream(unsupported)));
  }

  @Test void liveSourcesAreExplicitlyExcluded() throws Exception {
    try (var connection = java.sql.DriverManager.getConnection("jdbc:duckdb:"); var statement = connection.createStatement()) {
      statement.execute("CREATE TABLE data(id BIGINT PRIMARY KEY, value DOUBLE)");
      statement.execute("INSERT INTO data VALUES (1,2),(2,4)");
      Schema schema = Schema.fromSource(DuckDbSource.of(connection));
      StructuredDataException failure = assertThrows(StructuredDataException.class, () -> save(schema));
      assertEquals("UNSUPPORTED_REQUEST", failure.code()); assertFalse(connection.isClosed());
    }
  }

  private static byte[] change(byte[] original, java.util.function.Consumer<Map<String,Object>> change) throws Exception {
    DataInputStream input = new DataInputStream(new ByteArrayInputStream(original));
    byte[] prefix = input.readNBytes(12); byte[] metadata = input.readNBytes(input.readInt());
    @SuppressWarnings("unchecked") Map<String,Object> manifest = (Map<String,Object>) Json.read(metadata);
    change.accept(manifest); byte[] updated = Json.write(manifest);
    ByteArrayOutputStream bytes = new ByteArrayOutputStream(); DataOutputStream output = new DataOutputStream(bytes);
    output.write(prefix); output.writeInt(updated.length); output.write(updated); input.transferTo(output);
    return bytes.toByteArray();
  }
}
