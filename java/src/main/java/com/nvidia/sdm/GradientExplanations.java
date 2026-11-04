// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Json;
import java.nio.charset.StandardCharsets;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;

/** Validated gradients for each prediction, interpreted by the Rust engine. */
public final class GradientExplanations {
  /** Whether the engine could identify every row uniquely by its entity ID. */
  public enum Kind { ID, ROW_INDEX }
  /** Typed identity: integer IDs remain exact, including values wider than 64 bits. */
  public record Key(Kind kind, Object value, Long forecastStep) {}
  private final Map<Key, Map<String, Object>> entries;

  @SuppressWarnings("unchecked")
  GradientExplanations(Map<String, Object> collection) {
    Map<Key, Map<String, Object>> decoded = new LinkedHashMap<>();
    if (collection != null) {
      Kind kind = "id".equals(collection.get("keyed_by")) ? Kind.ID : Kind.ROW_INDEX;
      for (Object item : (List<?>) collection.get("rows")) {
        Map<String, Object> row = (Map<String, Object>) item;
        if (row.get("explanation") == null) continue;
        Object id = kind == Kind.ID ? parse((String) row.get("id")) : row.get("row_index");
        Number step = (Number) row.get("forecast_step");
        decoded.put(new Key(kind, id, step == null ? null : step.longValue()),
            (Map<String, Object>) immutable(parse((String) row.get("explanation"))));
      }
    }
    entries = Collections.unmodifiableMap(decoded);
  }

  /** Immutable entries in result-row order. */
  public Map<Key, Map<String, Object>> entries() { return entries; }
  /** The entry for this exact typed identity, or null if it was not explained. */
  public Map<String, Object> get(Key key) { return entries.get(key); }

  private static Object parse(String text) {
    try { return Json.read(text.getBytes(StandardCharsets.UTF_8)); }
    catch (Json.JsonException e) {
      throw new StructuredDataException("invalid engine gradient record", "INTERNAL_ERROR", Map.of(), null);
    }
  }

  private static Object immutable(Object value) {
    if (value instanceof Map<?, ?> map) {
      Map<String, Object> copy = new LinkedHashMap<>();
      map.forEach((key, item) -> copy.put((String) key, immutable(item)));
      return Collections.unmodifiableMap(copy);
    }
    if (value instanceof List<?> list) {
      List<Object> copy = new ArrayList<>();
      list.forEach(item -> copy.add(immutable(item)));
      return Collections.unmodifiableList(copy);
    }
    return value;
  }

  /**
   * One Arrow row per gradient. The caller closes the returned root before its allocator.
   * Ordinary IDs and keys retain string or integer Arrow types. Mixed/wide keys use JSON text,
   * identified by schema metadata; output descriptors are lossless JSON.
   */
  @SuppressWarnings("unchecked")
  public VectorSchemaRoot toArrow(BufferAllocator allocator) {
    List<Object[]> lines = new ArrayList<>();
    entries.forEach((key, entry) -> {
      List<?> outputs = (List<?>) entry.get("outputs");
      Map<String, Object> instance = (Map<String, Object>) entry.get("instance");
      if (instance != null) {
        List<?> columns = (List<?>) instance.get("columns");
        List<?> gradients = (List<?>) instance.get("gradients");
        for (int out = 0; out < outputs.size(); out++) {
          List<?> values = (List<?>) gradients.get(out);
          for (int col = 0; col < columns.size(); col++) {
            lines.add(new Object[] {key, "instance", 0L, key.kind() == Kind.ID ? key.value() : null,
                0L, outputs.get(out), columns.get(col), values.get(col)});
          }
        }
      }
      Map<String, Object> related = (Map<String, Object>) entry.getOrDefault("related", Map.of());
      Map<String, Object> hops = (Map<String, Object>) entry.getOrDefault("hops", Map.of());
      related.forEach((name, raw) -> {
        Map<String, Object> table = (Map<String, Object>) raw;
        List<?> columns = (List<?>) table.get("columns");
        List<?> gradients = (List<?>) table.get("gradients");
        List<?> keys = (List<?>) table.get("keys");
        List<?> distances = (List<?>) hops.get(name);
        for (int out = 0; out < outputs.size(); out++) {
          List<?> rows = (List<?>) gradients.get(out);
          for (int row = 0; row < rows.size(); row++) {
            List<?> values = (List<?>) rows.get(row);
            for (int col = 0; col < columns.size(); col++) {
              lines.add(new Object[] {key, name, distances == null ? null : distances.get(row),
                  keys == null ? null : keys.get(row), (long) row, outputs.get(out), columns.get(col), values.get(col)});
            }
          }
        }
      });
    });
    VarCharVector kind = new VarCharVector("identity_kind", allocator);
    List<Object> identities = lines.stream().map(line -> ((Key) line[0]).value()).toList();
    FieldVector id = keyVector("identity", identities, allocator);
    BigIntVector step = new BigIntVector("forecast_step", allocator);
    VarCharVector table = new VarCharVector("table", allocator);
    BigIntVector hop = new BigIntVector("hop", allocator);
    List<Object> keys = lines.stream().map(line -> line[3]).toList();
    FieldVector relatedKey = keyVector("key", keys, allocator);
    BigIntVector row = new BigIntVector("row", allocator);
    VarCharVector output = new VarCharVector("output_json", allocator);
    VarCharVector column = new VarCharVector("column", allocator);
    Float8Vector gradient = new Float8Vector("gradient", allocator);
    List<FieldVector> vectors = List.of(kind,id,step,table,hop,relatedKey,row,output,column,gradient);
    Map<String,String> metadata = new LinkedHashMap<>();
    boolean encodedIds = encoded(identities), encodedKeys = encoded(keys);
    if (encodedIds) metadata.put("sdm.identity_encoding", "json");
    if (encodedKeys) metadata.put("sdm.key_encoding", "json");
    VectorSchemaRoot root = new VectorSchemaRoot(new Schema(vectors.stream().map(FieldVector::getField).toList(),metadata),vectors,0);
    try {
      root.allocateNew();
      for (int i = 0; i < lines.size(); i++) {
        Object[] line = lines.get(i); Key key = (Key) line[0];
        kind.setSafe(i, (key.kind() == Kind.ID ? "id" : "row_index").getBytes(StandardCharsets.UTF_8));
        putKey(id,i,key.value(),encodedIds);
        if (key.forecastStep() != null) step.setSafe(i, key.forecastStep());
        table.setSafe(i, ((String) line[1]).getBytes(StandardCharsets.UTF_8));
        if (line[2] != null) hop.setSafe(i, ((Number) line[2]).longValue());
        putKey(relatedKey,i,line[3],encodedKeys);
        row.setSafe(i, ((Number) line[4]).longValue());
        output.setSafe(i, Json.write(line[5]));
        column.setSafe(i, ((String) line[6]).getBytes(StandardCharsets.UTF_8));
        gradient.setSafe(i, ((Number) line[7]).doubleValue());
      }
      root.setRowCount(lines.size());
      return root;
    } catch (RuntimeException e) { root.close(); throw e; }
  }
  private static boolean integral(Object value) {
    return value instanceof Byte || value instanceof Short || value instanceof Integer || value instanceof Long
        || value instanceof BigInteger n && n.bitLength() < 64;
  }
  private static boolean encoded(List<Object> values) {
    return !values.stream().allMatch(v -> v == null || v instanceof String)
        && !values.stream().allMatch(v -> v == null || integral(v));
  }
  private static FieldVector keyVector(String name,List<Object> values,BufferAllocator allocator) {
    return values.stream().anyMatch(v -> v != null) && values.stream().allMatch(v -> v == null || integral(v))
        ? new BigIntVector(name,allocator) : new VarCharVector(name,allocator);
  }
  private static void putKey(FieldVector vector,int at,Object value,boolean json) {
    if (value == null) { vector.setNull(at); return; }
    if (vector instanceof BigIntVector longs) longs.setSafe(at,((Number)value).longValue());
    else ((VarCharVector)vector).setSafe(at,json ? Json.write(value) : ((String)value).getBytes(StandardCharsets.UTF_8));
  }

}
