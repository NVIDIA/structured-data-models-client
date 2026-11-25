// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.OptionalRequest;
import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The core's messages read into the public provider types as the shared vectors ({@code
 * contract/host-provider-vectors.json}) say, which the Python host's tests read too: each request,
 * with the key sets it names; each expected schema; and each declared capabilities' wire.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class HostProviderVectorsTest {
  private static final Map<String, Object> VECTORS = vectors();
  private final BufferAllocator allocator = new RootAllocator();

  @SuppressWarnings("unchecked")
  private static Map<String, Object> vectors() {
    try {
      return (Map<String, Object>)
          Json.read(Files.readAllBytes(Path.of("..", "contract", "host-provider-vectors.json")));
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  @SuppressWarnings("unchecked")
  static Stream<Map<String, Object>> cases(String part) {
    return ((List<Object>) VECTORS.get(part)).stream().map(c -> (Map<String, Object>) c);
  }

  static Stream<Map<String, Object>> requests() {
    return cases("requests");
  }

  static Stream<Map<String, Object>> expected() {
    return cases("expected");
  }

  static Stream<Map<String, Object>> capabilities() {
    return cases("capabilities");
  }

  @AfterAll
  void close() {
    allocator.close();
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("requests")
  @SuppressWarnings("unchecked")
  void eachRequestIsReadAsTheSharedVectorsSay(Map<String, Object> vector) throws Exception {
    List<ArrowReader> opened = new ArrayList<>();
    try {
      DataRequest read =
          HostProviders.Requests.read(
              (Map<String, Object>) vector.get("wire"),
              id -> {
                ArrowReader reader = keySet(id);
                opened.add(reader);
                return reader;
              });
      assertEquals(vector.get("decoded"), neutral(read), (String) vector.get("name"));
    } finally {
      for (ArrowReader reader : opened) reader.close();
    }
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("expected")
  @SuppressWarnings("unchecked")
  void eachExpectedSchemaIsReadAsTheSharedVectorsSay(Map<String, Object> vector) {
    Optional<Schema> schema = HostProviders.expected((Map<String, Object>) vector.get("wire"));
    Object decoded =
        schema
            .<Object>map(
                s ->
                    s.getFields().stream()
                        .map(f -> (Object) List.of(f.getName(), type(f), f.isNullable()))
                        .toList())
            .orElse(null);
    assertEquals(vector.get("decoded"), decoded, (String) vector.get("name"));
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("capabilities")
  @SuppressWarnings("unchecked")
  void eachCapabilitiesDeclaredIsSentAsTheSharedVectorsSay(Map<String, Object> vector)
      throws Exception {
    Map<String, Object> declared = (Map<String, Object>) vector.get("declared");
    Capabilities.Builder builder = Capabilities.builder();
    if (declared.get("optional") instanceof List<?> optional)
      for (Object name : optional)
        builder.answers(
            Arrays.stream(OptionalRequest.values())
                .filter(r -> r.wireName().equals(name))
                .findFirst()
                .orElseThrow());
    if (declared.get("filters") instanceof List<?> filters)
      for (Object name : filters)
        builder.filters(
            Arrays.stream(DataRequest.FilterKind.values())
                .filter(k -> k.wireName().equals(name))
                .findFirst()
                .orElseThrow());
    if (declared.get("max_keys_per_request") instanceof Number max)
      builder.maxKeysPerRequest(max.longValue());
    if ("unseeded".equals(declared.get("sampling")))
      builder.sampling(Capabilities.Sampling.UNSEEDED);
    if (declared.get("max_seed") instanceof Number seed) builder.maxSeed(seed.longValue());
    if (declared.containsKey("null_order"))
      builder.nullOrder(
          declared.get("null_order") == null
              ? null
              : Capabilities.NullOrder.valueOf(((String) declared.get("null_order")).toUpperCase()));
    if (declared.get("fallback") instanceof Map<?, ?> fallback)
      builder.fallback(
          ((Number) fallback.get("max_rows")).longValue(),
          ((Number) fallback.get("max_bytes")).longValue());
    Map<String, Object> wire = HostProviders.capabilities(builder.build());
    assertEquals(vector.get("wire"), Json.read(Json.write(wire)), (String) vector.get("name"));
  }

  // --- round trips ------------------------------------------------------------------------------

  @SuppressWarnings("unchecked")
  static Stream<Map<String, Object>> trips(String part) {
    Map<String, Object> trips = (Map<String, Object>) VECTORS.get("round_trips");
    return ((List<Object>) trips.get(part)).stream().map(c -> (Map<String, Object>) c);
  }

  static Stream<Map<String, Object>> everyRequest() {
    return Stream.concat(requests(), trips("requests"));
  }

  static Stream<Map<String, Object>> capabilityTrips() {
    return trips("capabilities");
  }

  static Stream<Map<String, Object>> failureTrips() {
    return trips("failures");
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("everyRequest")
  @SuppressWarnings("unchecked")
  void eachRequestReadAndWrittenAgainIsTheRequestSent(Map<String, Object> vector)
      throws Exception {
    Map<Integer, ArrowReader> given = new LinkedHashMap<>();
    try {
      Map<String, Object> wire = (Map<String, Object>) vector.get("wire");
      DataRequest read =
          HostProviders.Requests.read(wire, id -> given.computeIfAbsent(id, this::keySet));
      Map<Integer, ArrowReader> keys = new LinkedHashMap<>();
      Map<String, Object> written = HostProviders.Requests.write(read, keys);
      // A written request numbers its key sets anew: each is named back by the id the core gave
      // its rows under, the very reader read.
      Map<Long, Long> renamed = new LinkedHashMap<>();
      keys.forEach(
          (at, reader) ->
              given.forEach(
                  (id, original) -> {
                    if (original == reader) renamed.put((long) at, (long) id);
                  }));
      assertEquals(keys.size(), renamed.size(), (String) vector.get("name"));
      assertEquals(
          wire, renamed(Json.read(Json.write(written)), renamed), (String) vector.get("name"));
    } finally {
      for (ArrowReader reader : given.values()) reader.close();
    }
  }

  @SuppressWarnings("unchecked")
  private static Object renamed(Object node, Map<Long, Long> renamed) {
    if (node instanceof Map<?, ?> map) {
      Map<String, Object> out = new LinkedHashMap<>();
      map.forEach((name, item) -> out.put((String) name, renamed(item, renamed)));
      if (map.containsKey("attachment") && map.containsKey("columns")) {
        out.put("attachment", renamed.get(((Number) map.get("attachment")).longValue()));
      }
      return out;
    }
    if (node instanceof List<?> list) {
      List<Object> out = new ArrayList<>();
      for (Object item : list) out.add(renamed(item, renamed));
      return out;
    }
    return node;
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("capabilityTrips")
  @SuppressWarnings("unchecked")
  void eachCapabilitiesReadAndWrittenAgainAreTheOnesSent(Map<String, Object> vector)
      throws Exception {
    Map<String, Object> wire = (Map<String, Object>) vector.get("wire");
    Capabilities read = HostProviders.readCapabilities(wire);
    assertEquals(
        wire, Json.read(Json.write(HostProviders.capabilities(read))), (String) vector.get("name"));
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("failureTrips")
  @SuppressWarnings("unchecked")
  void eachFailureReadAndWrittenAgainTellsTheSame(Map<String, Object> vector) throws Exception {
    Map<String, Object> wire = (Map<String, Object>) vector.get("wire");
    Map<String, Object> told = new LinkedHashMap<>();
    wire.forEach(
        (name, given) -> {
          if (given != null) told.put(name, given);
        });
    assertEquals(
        told,
        Json.read(Json.write(HostProviders.failure(HostProviders.readFailure(wire)))),
        (String) vector.get("name"));
  }

  @org.junit.jupiter.api.Test
  @SuppressWarnings("unchecked")
  void aValueKeepsItsExactType() throws Exception {
    Map<String, Object> wire =
        (Map<String, Object>) trips("requests").findFirst().orElseThrow().get("wire");
    DataRequest.Scan read = (DataRequest.Scan) HostProviders.Requests.read(wire, this::keySet);
    DataRequest.Compare first = (DataRequest.Compare) read.filters().get(9);
    assertEquals(
        new DataRequest.Value(new BigInteger("18446744073709551615"), new ArrowType.Int(64, false)),
        first.value());
    DataRequest.Compare zoned = (DataRequest.Compare) read.filters().get(27);
    assertEquals(
        new ArrowType.Timestamp(
            org.apache.arrow.vector.types.TimeUnit.NANOSECOND, "America/New_York"),
        zoned.value().type());
    assertEquals(
        new DataRequest.RelationColumn("total", Optional.empty(), Optional.of("price * quantity")),
        read.table().columns().get(1));
  }

  // --- the vectors' key sets ---------------------------------------------------------------------

  @SuppressWarnings("unchecked")
  private ArrowReader keySet(int id) {
    Map<String, Object> keySet =
        (Map<String, Object>) ((Map<String, Object>) VECTORS.get("key_sets")).get(Integer.toString(id));
    List<Field> fields = new ArrayList<>();
    for (Object field : (List<Object>) keySet.get("fields")) {
      String name = (String) ((List<Object>) field).get(0);
      fields.add(new Field(name, FieldType.nullable(new ArrowType.Int(64, true)), null));
    }
    List<List<Object>> rows = (List<List<Object>>) keySet.get("rows");
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (VectorSchemaRoot root = VectorSchemaRoot.create(new Schema(fields), allocator);
        ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bytes)) {
      root.allocateNew();
      for (int at = 0; at < rows.size(); at++)
        for (int column = 0; column < fields.size(); column++)
          ((BigIntVector) root.getVector(column))
              .setSafe(at, ((Number) rows.get(at).get(column)).longValue());
      root.setRowCount(rows.size());
      writer.start();
      writer.writeBatch();
      writer.end();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
    return ArrowData.reader(bytes.toByteArray(), allocator);
  }

  // --- the neutral form --------------------------------------------------------------------------

  /** A public request, filter, part or value in the vectors' neutral form. */
  private static Object neutral(Object value) throws Exception {
    if (value == null || value instanceof String || value instanceof Boolean) return value;
    if (value instanceof Optional<?> optional) return neutral(optional.orElse(null));
    if (value instanceof java.util.OptionalLong optional)
      return optional.isPresent() ? optional.getAsLong() : null;
    if (value instanceof Integer || value instanceof Long) return ((Number) value).longValue();
    if (value instanceof List<?> list) {
      List<Object> out = new ArrayList<>();
      for (Object item : list) out.add(neutral(item));
      return out;
    }
    if (value instanceof DataRequest.Relation relation) return relation.parts();
    if (value instanceof DataRequest.KeySet keySet) return keySet(keySet);
    if (value instanceof DataRequest.Count count) return Map.of("count", count.count());
    if (value instanceof DataRequest.AllBut allBut) return Map.of("all_but", allBut.value());
    if (value instanceof Enum<?> kind) return kind.name().toLowerCase();
    if (value instanceof Record record) return record(record);
    throw new IllegalArgumentException("no neutral form of " + value.getClass());
  }

  private static Map<String, Object> record(Record record) throws Exception {
    Map<String, Object> out = new LinkedHashMap<>();
    if (record instanceof DataRequest request) out.put("request", request.name());
    String filter = FILTERS.get(record.getClass().getSimpleName());
    if (filter != null) out.put("filter", filter);
    for (var component : record.getClass().getRecordComponents()) {
      Object field = component.getAccessor().invoke(record);
      String name = snake(component.getName());
      if (LATER.contains(name) && Optional.empty().equals(field)) {
        // Fields the shared vectors' first cases predate.
        continue;
      }
      out.put(
          name,
          switch (name) {
            case "value", "seed" -> scalar(field);
            case "values" -> {
              List<Object> values = new ArrayList<>();
              for (Object each : (List<?>) field) values.add(scalar(each));
              yield values;
            }
            default -> neutral(field);
          });
    }
    return out;
  }

  /** Fields the vectors' decoded forms leave out where they say nothing; round trips hold them. */
  private static final List<String> LATER = List.of("key_dtype", "source_type");

  private static final Map<String, String> FILTERS =
      Map.of(
          "KeyIn", "key_in",
          "TimeRange", "time_range",
          "NotNull", "not_null",
          "Compare", "compare",
          "In", "in",
          "IsNull", "is_null",
          "AnyOf", "any_of",
          "AllOf", "all_of",
          "HasRows", "has_rows");

  private static String snake(String camel) {
    return camel.replaceAll("([a-z])([A-Z])", "$1_$2").toLowerCase();
  }

  private static Map<String, Object> keySet(DataRequest.KeySet keySet) throws Exception {
    List<Object> rows = new ArrayList<>();
    ArrowReader reader = keySet.reader();
    while (reader.loadNextBatch()) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      for (int at = 0; at < root.getRowCount(); at++) {
        List<Object> row = new ArrayList<>();
        for (FieldVector vector : root.getFieldVectors()) row.add(vector.getObject(at));
        rows.add(row);
      }
    }
    Map<String, Object> out = new LinkedHashMap<>();
    out.put("columns", keySet.columns());
    out.put("rows", rows);
    return out;
  }

  private static final DateTimeFormatter WALL =
      DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSSSSS");

  private static Object scalar(Object value) {
    if (value instanceof DataRequest.Value typed) return scalar(typed.value());
    if (value == null) return null;
    if (value instanceof Boolean b) return Map.of("bool", b);
    if (value instanceof Long || value instanceof Integer || value instanceof BigInteger)
      return Map.of("int", ((Number) value).longValue());
    if (value instanceof Double d) return Map.of("float", BigDecimal.valueOf(d));
    if (value instanceof BigDecimal d) return Map.of("decimal", d.toPlainString());
    if (value instanceof String s) return Map.of("text", s);
    if (value instanceof LocalDate d) return Map.of("date", d.toString());
    if (value instanceof java.time.LocalTime t)
      return Map.of("time", t.format(DateTimeFormatter.ofPattern("HH:mm:ss.SSSSSSSSS")));
    Map<String, Object> time = new LinkedHashMap<>();
    if (value instanceof LocalDateTime wall) {
      time.put("timestamp", wall.format(WALL));
      time.put("zone", null);
      return time;
    }
    if (value instanceof Instant instant) {
      // An instant keeps no zone name: the vectors' zoned values are UTC's.
      time.put("timestamp", LocalDateTime.ofInstant(instant, ZoneOffset.UTC).format(WALL));
      time.put("zone", "UTC");
      return time;
    }
    throw new IllegalArgumentException("no neutral form of " + value.getClass());
  }

  private static String type(Field field) {
    ArrowType type = field.getType();
    if (type instanceof ArrowType.List)
      return "list<" + type(field.getChildren().get(0)) + ">";
    if (type instanceof ArrowType.Utf8) return "utf8";
    if (type instanceof ArrowType.Bool) return "bool";
    if (type instanceof ArrowType.Int i) return (i.getIsSigned() ? "int" : "uint") + i.getBitWidth();
    if (type instanceof ArrowType.Timestamp t)
      return "timestamp[" + switch (t.getUnit()) {
        case SECOND -> "s";
        case MILLISECOND -> "ms";
        case MICROSECOND -> "us";
        case NANOSECOND -> "ns";
      } + "]";
    return type.toString();
  }
}
