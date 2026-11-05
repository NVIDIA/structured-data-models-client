// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowTabular;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Json;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Descriptive context-column associations computed by Rust, not causal effects.
 * Formatting preserves engine order and never recomputes cohorts or scores.
 */
public final class ColumnExplanations {
  /** One column/cohort/class row; classLabel is null for regression and otherwise an exact scalar. */
  public record Row(String table, long hop, String column, String stype, String cohort,
      Object classLabel, double share, double mean, double score) {}

  private final List<Row> rows;
  private final String encodedAnalysis;

  /** Analyze a model request's labelled context once, through the Rust engine. */
  public ColumnExplanations(Map<String, ?> payload) {
    ColumnExplanations result = fromCore(analyze(Objects.requireNonNull(payload, "payload")));
    this.rows = result.rows;
    this.encodedAnalysis = result.encodedAnalysis;
  }

  private static Map<String, Object> analyze(Map<String, ?> payload) {
    try (Host.Finished finished = Host.run(
        Map.of("op", "analyze_columns", "payloads", List.of(1)),
        Map.of(1, Host.attachment(payload)))) {
      Map<String, Object> analysis = ArrowTabular.value(finished);
      if ("unsupported".equals(analysis.get("status")) || "failed".equals(analysis.get("status"))) {
        throw new StructuredDataException("column analysis is unavailable", "INVALID_REQUEST",
            Map.of("reason", analysis.getOrDefault("reason", "unavailable")), null);
      }
      return analysis;
    }
  }

  @SuppressWarnings("unchecked")
  private ColumnExplanations(Map<String, Object> analysis, List<?> rawRows) {
    try {
      List<Row> values = new ArrayList<>();
      for (Object raw : Objects.requireNonNull(rawRows)) {
        Map<String, Object> row = (Map<String, Object>) raw;
        Object label = Json.read(((String) row.get("class_json")).getBytes(StandardCharsets.UTF_8));
        if (label != null && !(label instanceof String || label instanceof Number || label instanceof Boolean))
          throw new IllegalArgumentException("class label is not a scalar");
        long hop = new java.math.BigDecimal(((Number) row.get("hop")).toString()).longValueExact();
        if (hop < 0 || hop > 0xffff_ffffL) throw new IllegalArgumentException("invalid hop");
        values.add(new Row(requiredText(row, "table"), hop, requiredText(row, "column"),
            (String) row.get("stype"), requiredText(row, "cohort"), label,
            finite(row, "share"), finite(row, "mean"), finite(row, "score")));
      }
      rows = List.copyOf(values);
      encodedAnalysis = new String(Json.write(analysis), StandardCharsets.UTF_8);
    } catch (Json.JsonException | ClassCastException | IllegalArgumentException | NullPointerException | ArithmeticException error) {
      throw new StructuredDataException("invalid engine column analysis", "INTERNAL_ERROR", Map.of(), null);
    }
  }

  static ColumnExplanations fromCore(Map<String, Object> analysis) {
    if (analysis == null) return null;
    Object status = analysis.get("status");
    if ("unsupported".equals(status) || "failed".equals(status)) {
      if (!(analysis.get("reason") instanceof String)) throw invalidAnalysis();
      return null;
    }
    if (!"ready".equals(status) || !(analysis.get("rows") instanceof List<?> values)) throw invalidAnalysis();
    return new ColumnExplanations(analysis, values);
  }

  private static StructuredDataException invalidAnalysis() {
    return new StructuredDataException("invalid engine column analysis", "INTERNAL_ERROR", Map.of(), null);
  }

  private static String requiredText(Map<String, Object> row, String name) {
    return Objects.requireNonNull((String) row.get(name));
  }

  private static double finite(Map<String, Object> row, String name) {
    double value = ((Number) row.get(name)).doubleValue();
    if (!Double.isFinite(value)) throw new IllegalArgumentException("nonfinite analysis value");
    return value;
  }

  String encoded() { return encodedAnalysis; }

  /** Detached immutable rows in engine score/cohort/class order. */
  public List<Row> rows() { return rows; }

  /**
   * Render the same nine columns as Python's dataframe view.
   * The caller closes the returned root before its allocator; this method never closes the allocator.
   * Mixed, decimal or wide class labels use lossless JSON text with metadata sdm.class_encoding=json.
   */
  public VectorSchemaRoot toArrow(BufferAllocator allocator) {
    Objects.requireNonNull(allocator, "allocator");
    List<Object> labels = rows.stream().map(Row::classLabel).toList();
    VarCharVector table = new VarCharVector("table", allocator);
    BigIntVector hop = new BigIntVector("hop", allocator);
    VarCharVector column = new VarCharVector("column", allocator);
    VarCharVector stype = new VarCharVector("stype", allocator);
    VarCharVector cohort = new VarCharVector("cohort", allocator);
    FieldVector classes = classVector(labels, allocator);
    Float8Vector share = new Float8Vector("share", allocator);
    Float8Vector mean = new Float8Vector("mean", allocator);
    Float8Vector score = new Float8Vector("score", allocator);
    boolean json = encodedLabels(labels);
    List<FieldVector> vectors = List.of(table, hop, column, stype, cohort, classes, share, mean, score);
    VectorSchemaRoot root = new VectorSchemaRoot(new Schema(
        vectors.stream().map(FieldVector::getField).toList(),
        json ? Map.of("sdm.class_encoding", "json") : Map.of()), vectors, 0);
    try {
      root.allocateNew();
      for (int i = 0; i < rows.size(); i++) {
        Row row = rows.get(i);
        putText(table, i, row.table()); hop.setSafe(i, row.hop());
        putText(column, i, row.column()); putText(stype, i, row.stype());
        putText(cohort, i, row.cohort()); putClass(classes, i, row.classLabel(), json);
        share.setSafe(i, row.share()); mean.setSafe(i, row.mean()); score.setSafe(i, row.score());
      }
      root.setRowCount(rows.size());
      return root;
    } catch (RuntimeException | Error error) {
      root.close();
      throw error;
    }
  }

  private static boolean integral(Object value) {
    return value instanceof Byte || value instanceof Short || value instanceof Integer || value instanceof Long
        || value instanceof BigInteger integer && integer.bitLength() < 64;
  }

  private static boolean all(List<Object> values, java.util.function.Predicate<Object> kind) {
    return values.stream().allMatch(value -> value == null || kind.test(value));
  }

  private static boolean encodedLabels(List<Object> values) {
    return !(all(values, String.class::isInstance) || all(values, ColumnExplanations::integral)
        || all(values, Boolean.class::isInstance)
        || all(values, value -> value instanceof Double || value instanceof Float));
  }

  private static FieldVector classVector(List<Object> values, BufferAllocator allocator) {
    if (values.stream().anyMatch(Objects::nonNull)) {
      if (all(values, ColumnExplanations::integral)) return new BigIntVector("class", allocator);
      if (all(values, Boolean.class::isInstance)) return new BitVector("class", allocator);
      if (all(values, value -> value instanceof Double || value instanceof Float)) return new Float8Vector("class", allocator);
    }
    return new VarCharVector("class", allocator);
  }

  private static void putText(VarCharVector vector, int at, String value) {
    if (value == null) vector.setNull(at);
    else vector.setSafe(at, value.getBytes(StandardCharsets.UTF_8));
  }

  private static void putClass(FieldVector vector, int at, Object value, boolean json) {
    if (value == null) vector.setNull(at);
    else if (vector instanceof BigIntVector longs) longs.setSafe(at, ((Number) value).longValue());
    else if (vector instanceof BitVector bits) bits.setSafe(at, Boolean.TRUE.equals(value) ? 1 : 0);
    else if (vector instanceof Float8Vector floats) floats.setSafe(at, ((Number) value).doubleValue());
    else try {
      ((VarCharVector) vector).setSafe(at, json ? Json.write(value) : ((String) value).getBytes(StandardCharsets.UTF_8));
    } catch (Json.JsonException error) {
      throw new StructuredDataException("invalid engine class label", "INTERNAL_ERROR", Map.of(), null);
    }
  }
}
