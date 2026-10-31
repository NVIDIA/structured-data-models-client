// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Indices;
import com.nvidia.sdm.internal.Providers;
import com.nvidia.sdm.internal.Results;
import java.math.BigInteger;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.function.Function;

/**
 * A question expressed in PQL, as Python's {@code PredictiveQuery} is (design choice 10): {@link
 * #parse} keeps the text, refusing only an empty one, and {@link #materialize} resolves it against
 * a schema and gathers the request that answers it, both on the core.
 *
 * <pre>{@code
 * Request request = PredictiveQuery.parse(
 *         "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id")
 *     .materialize(schema, MaterializeOptions.builder()
 *         .indices(List.of(1, 2))
 *         .anchorTime(LocalDate.of(2024, 3, 1))
 *         .build());
 * }</pre>
 */
public final class PredictiveQuery {
  private final String text;

  private PredictiveQuery(String text) {
    this.text = text;
  }

  /**
   * The query {@code text}, kept as given; nothing is resolved until it is materialized, as
   * Python's constructor keeps it.
   *
   * @throws StructuredDataException {@code INVALID_REQUEST} when the text is null or blank
   */
  public static PredictiveQuery parse(String text) {
    if (text == null || blank(text)) {
      throw new StructuredDataException(
          "query must be a non-empty string", "INVALID_REQUEST", Map.of(), null);
    }
    return new PredictiveQuery(text);
  }

  /**
   * Whether {@code text} is empty once stripped as Python's {@code str.strip()} strips: of every
   * character {@code str.isspace()} holds, Java's whitespace and space separators, the no-break
   * spaces among them, and NEL (U+0085).
   */
  private static boolean blank(String text) {
    return text.codePoints()
        .allMatch(c -> Character.isWhitespace(c) || Character.isSpaceChar(c) || c == 0x85);
  }

  /** The query's text, as given. */
  public String text() {
    return text;
  }

  /** Materialize one in-memory table using new Arrow rows to score. */
  public Request materialize(Schema schema, org.apache.arrow.vector.table.Table predict) {
    if (schema.data().size() != 1)
      throw new StructuredDataException("Arrow scoring requires one in-memory table", "INVALID_REQUEST", Map.of(), null);
    Map<String, Object> input = new LinkedHashMap<>();
    input.put("schema", schema.toHeld()); input.put("query", text);
    input.put("indices", List.of()); input.put("num_context_samples", null);
    try (Host.Finished finished = com.nvidia.sdm.internal.ArrowTabular.run("materialize_predict", input,
        Map.of("table", schema.data().values().iterator().next(), "predict",
            com.nvidia.sdm.internal.ArrowTabular.snapshot(predict)), null)) {
      Map<String, Object> result = com.nvidia.sdm.internal.ArrowTabular.value(finished);
      @SuppressWarnings("unchecked") Map<String, Object> description = (Map<String, Object>) result.get("description");
      Map<String, Object> request = com.nvidia.sdm.internal.ArrowTabular.request(
          (String) description.get("target"), (String) description.get("task"),
          (String) description.get("entity_table"), description.get("schema"));
      return new Request(finished.arrow(((Number) result.get("context")).intValue()),
          finished.arrow(((Number) result.get("predict")).intValue()), request, text);
    }
  }

  /**
   * The request that answers this query over {@code schema}'s tables, gathered on the core as
   * Python's {@code materialize(indices=...)} gathers it on the core route: the core resolves the
   * query against the schema for the relational model, samples the source and lays out the
   * request, with every option as given. A request of more entities than one call
   * scores is gathered as several calls, which {@link SdmClient#predict} answers as one.
   *
   * @throws StructuredDataException {@code INVALID_REQUEST} for a query the core does not accept
   *     or an option it refuses, naming it; {@code UNSUPPORTED_REQUEST} for a capability the core
   *     does not support, naming the rule; {@code PROVIDER_ERROR} when the database failed a read
   */
  public Request materialize(Schema schema, MaterializeOptions options) {
    Objects.requireNonNull(schema, "schema");
    Objects.requireNonNull(options, "options");
    if (options.indices() == null) {
      throw new StructuredDataException(
          "materialize() needs the rows to score: pass indices naming the entities. There is no"
              + " all-rows default, because it would silently score an entire table",
          "INVALID_REQUEST",
          Map.of(),
          null);
    }
    boolean relational = SdmClient.KUMO_RELATIONAL.equals(options.model())
        || (options.model() == null && (schema.tables().size() > 1 || !schema.relationships().isEmpty()));
    if (!schema.data().isEmpty() && relational) {
      try (RelationalMaterializer materializer = new RelationalMaterializer(schema)) {
        return materialize(materializer, options);
      }
    }
    if (schema.data().isEmpty() && options.model() != null
        && !SdmClient.KUMO_RELATIONAL.equals(options.model()))
      throw new StructuredDataException("warehouse materialization requires kumo-relational", "UNSUPPORTED_REQUEST", Map.of(), null);
    if (!schema.data().isEmpty()) {
      if (schema.data().size() != 1) throw new StructuredDataException(
          "Tabular needs one table; choose kumo-relational for related tables", "INVALID_REQUEST", Map.of(), null);
      Map<String, Object> input = new LinkedHashMap<>();
      input.put("schema", schema.toHeld()); input.put("query", text);
      input.put("indices", options.indices().stream().map(com.nvidia.sdm.internal.ArrowTabular::key).toList());
      input.put("num_context_samples", options.numContextSamples());
      if (options.candidates() != null || !options.options().isEmpty() || options.anchorTime() != null || options.contextAnchorGiven())
        throw new StructuredDataException("warehouse sampling options do not apply to Tabular",
            "INVALID_REQUEST", Map.of(), null);
      try (Host.Finished finished = com.nvidia.sdm.internal.ArrowTabular.run("materialize", input,
          Map.of("table", schema.data().values().iterator().next()), null)) {
        Map<String, Object> result = com.nvidia.sdm.internal.ArrowTabular.value(finished);
        @SuppressWarnings("unchecked") Map<String, Object> description = (Map<String, Object>) result.get("description");
        Map<String, Object> request = com.nvidia.sdm.internal.ArrowTabular.request(
            (String) description.get("target"), (String) description.get("task"),
            (String) description.get("entity_table"), description.get("schema"));
        return new Request(finished.arrow(((Number) result.get("context")).intValue()),
            finished.arrow(((Number) result.get("predict")).intValue()), request, text);
      }
    }
    Providers.Chosen chosen = Providers.of(schema.source());
    Object anchor = options.anchorTime();
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "materialize");
    operation.put("provider", chosen.choice());
    operation.put("schema", 1);
    operation.put("query", 2);
    operation.put("indices", 3);
    operation.put("anchor_time", anchor == MaterializeOptions.EACH_ENTITY ? null : timestamp(anchor));
    operation.put("seed", seed(options.options().get("random_seed")));
    operation.put("unseeded", options.options().containsKey("random_seed") && options.options().get("random_seed") == null);
    operation.put("anchor_entity", anchor == MaterializeOptions.EACH_ENTITY);
    operation.put("values", 5);
    operation.put("options", List.of());
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(schema.toHeld()));
    attachments.put(2, Host.attachment(text));
    attachments.put(3, Indices.column(options.indices(), ArrowData.allocator()));
    attachments.put(5, Host.optionAttachment(values(options)));
    if (options.candidates() != null) {
      operation.put("candidates", 4);
      attachments.put(4, Host.attachment(candidates(options.candidates())));
    }
    Map<String, Function<Map<String, Object>, Object>> held =
        new LinkedHashMap<>(Providers.held(schema.source()));
    held.put("query", fields -> text);
    try (Host.Finished finished =
        Host.run(operation, attachments, new Host.Call(chosen.provider(), null, held))) {
      Results.Gathered gathered =
          (Results.Gathered) Results.expect(finished.result(), "materialize");
      if (gathered.query() == null) {
        throw new StructuredDataException(
            "the core gathered a query and said nothing of it", "INTERNAL_ERROR", Map.of(), null);
      }
      @SuppressWarnings("unchecked") Map<String, Object> state = (Map<String, Object>) finished.result().get("state");
      com.nvidia.sdm.internal.MaterializedRequest retained =
          new com.nvidia.sdm.internal.MaterializedRequest(state, finished);
      return new Request(schema, retained, text, gathered.query(),
          inferenceAsked(options.options().get("inference_config")));
    }
  }

  /** Reuse an in-memory graph and its continuing sampler across queries. */
  public Request materialize(RelationalMaterializer materializer, MaterializeOptions options) {
    return Objects.requireNonNull(materializer, "materializer").materialize(this,
        Objects.requireNonNull(options, "options"));
  }

  /** The anchor time as the contract's {@code ts} scalar, in nanoseconds. */
  static Map<String, Object> timestamp(Object anchor) {
    if (anchor == null) {
      return null;
    }
    Instant instant;
    String zone = null;
    if (anchor instanceof LocalDateTime wall) {
      instant = wall.toInstant(ZoneOffset.UTC);
    } else {
      ZonedDateTime zoned = (ZonedDateTime) anchor;
      instant = zoned.toInstant();
      zone = zoned.getZone().equals(ZoneOffset.UTC) ? "UTC" : zoned.getZone().getId();
    }
    BigInteger nanos =
        BigInteger.valueOf(instant.getEpochSecond())
            .multiply(BigInteger.valueOf(1_000_000_000L))
            .add(BigInteger.valueOf(instant.getNano()));
    Map<String, Object> scalar = new LinkedHashMap<>();
    scalar.put("t", "ts");
    scalar.put("v", nanos.longValueExact() + "");
    scalar.put("unit", "ns");
    scalar.put("zone", zone);
    return scalar;
  }

  /**
   * A seed as the contract writes one: a whole number as signed hexadecimal text with no leading
   * zeros; none for any other value, which the call's values carry to the core as given.
   */
  static String seed(Object value) {
    BigInteger whole;
    if (value instanceof Long || value instanceof Integer || value instanceof Short
        || value instanceof Byte) {
      whole = BigInteger.valueOf(((Number) value).longValue());
    } else if (value instanceof BigInteger big) {
      whole = big;
    } else {
      return null;
    }
    return (whole.signum() < 0 ? "-" : "") + "0x" + whole.abs().toString(16);
  }

  /** The call's values the core reads: the context size, then each option, by name. */
  static Map<String, Object> values(MaterializeOptions options) {
    Map<String, Object> values = new LinkedHashMap<>();
    values.put("num_context_samples", options.numContextSamples());
    values.putAll(new TreeMap<>(options.options()));
    if (options.contextAnchorGiven()) {
      Object anchor = options.contextAnchorTime();
      values.put(
          "context_anchor_time",
          anchor == MaterializeOptions.EACH_ENTITY ? "entity" : timestamp(anchor));
    }
    return values;
  }

  /** The inference options the caller set, which the core refuses where a model reads none. */
  static List<String> inferenceAsked(Object config) {
    if (!(config instanceof Map<?, ?> map)) {
      return List.of();
    }
    List<String> names = new ArrayList<>();
    for (Object key : map.keySet()) {
      names.add(String.valueOf(key));
    }
    names.sort(null);
    return names;
  }

  static List<Object> candidates(List<Object> candidates) {
    if (candidates == null) return null;
    List<Object> written = new ArrayList<>();
    java.util.Set<Object> seen = new java.util.HashSet<>();
    for (Object value : candidates) {
      if ((value instanceof Double d && !Double.isFinite(d))
          || (value instanceof Float f && !Float.isFinite(f)))
        throw new StructuredDataException("ranking candidates must be finite", "INVALID_REQUEST", Map.of(), null);
      boolean integer = value instanceof Byte || value instanceof Short || value instanceof Integer
          || value instanceof Long || value instanceof java.math.BigInteger;
      if (value instanceof Number && !integer && !(value instanceof java.math.BigDecimal)
          && !(value instanceof Double) && !(value instanceof Float))
        throw new StructuredDataException("unsupported ranking candidate type", "INVALID_REQUEST", Map.of(), null);
      Object identity = value;
      if (value instanceof Double || value instanceof Float)
        identity = new java.math.BigDecimal(((Number) value).doubleValue()).stripTrailingZeros();
      else if (value instanceof Number number)
        identity = new java.math.BigDecimal(number.toString()).stripTrailingZeros();
      else if (value instanceof Boolean b) identity = b ? java.math.BigDecimal.ONE : java.math.BigDecimal.ZERO;
      if (!seen.add(identity)) { written.add("repeat"); continue; }
      if (value == null) written.add(Map.of("value", Map.of("t", "null")));
      else if (value instanceof Boolean b) written.add(Map.of("value", Map.of("t", "bool", "v", b)));
      else if (value instanceof String text) written.add(Map.of("value", Map.of("t", "utf8", "v", text)));
      else if (value instanceof java.math.BigDecimal decimal) written.add(Map.of("decimal", decimal.toPlainString()));
      else if (value instanceof Double || value instanceof Float)
        written.add(Map.of("value", Map.of("t", "f64", "v", Double.toString(((Number) value).doubleValue()))));
      else if (value instanceof Number number) written.add(Map.of("integer", number.toString()));
      else throw new StructuredDataException("unsupported ranking candidate type", "INVALID_REQUEST", Map.of(), null);
    }
    return written;
  }

  @Override
  public String toString() {
    return "PredictiveQuery(" + text + ")";
  }
}
