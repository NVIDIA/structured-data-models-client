// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Indices;
import com.nvidia.sdm.internal.Json;
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

  /**
   * The request that answers this query over {@code schema}'s tables, gathered on the core as
   * Python's {@code materialize(indices=...)} gathers it on the core route: the core resolves the
   * query against the schema for the relational model, samples the source and lays out the
   * request, with every option as given. A request of more entities than one call
   * scores is gathered as several calls, which {@link SdmClient#predict} answers as one.
   *
   * @throws StructuredDataException {@code INVALID_REQUEST} for a query the core does not accept
   *     or an option it refuses, naming it; {@code UNSUPPORTED_REQUEST} for what the core does not
   *     materialize yet (a ranking or a forecast, which the Java library does not ask yet), naming
   *     the rule; {@code PROVIDER_ERROR} when the database failed a read
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
    operation.put("anchor_entity", anchor == MaterializeOptions.EACH_ENTITY);
    operation.put("values", 5);
    operation.put("options", List.of());
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(schema.toHeld()));
    attachments.put(2, Host.attachment(text));
    attachments.put(3, Indices.column(options.indices(), ArrowData.allocator()));
    attachments.put(5, Host.attachment(values(options)));
    Map<String, Function<Map<String, Object>, Object>> held = Map.of("query", fields -> text);
    try (Host.Finished finished =
        Host.run(operation, attachments, new Host.Call(chosen.provider(), null, held))) {
      Results.Gathered gathered =
          (Results.Gathered) Results.expect(finished.result(), "materialize");
      Results.State state = gathered.state();
      if (gathered.query() == null) {
        throw new StructuredDataException(
            "the core gathered a query and said nothing of it", "INTERNAL_ERROR", Map.of(), null);
      }
      if (state.inParts()) {
        throw new StructuredDataException(
            "the core gathered a question in parts, which the Java library does not ask yet",
            "UNSUPPORTED_REQUEST",
            Map.of("rule", "in_parts"),
            null);
      }
      List<byte[]> payloads = new ArrayList<>();
      payloads.add(finished.bytes(state.payload()));
      String query = (String) read(finished.bytes(state.query()));
      List<Long> rows = null;
      if (state.batches() != null) {
        rows = new ArrayList<>();
        for (Results.Batch batch : state.batches()) {
          payloads.add(finished.bytes(batch.payload()));
          rows.add(batch.rows());
        }
      }
      return new Request(
          schema,
          payloads,
          text,
          state.version(),
          rows,
          gathered.query(),
          inferenceAsked(options.options().get("inference_config")),
          query);
    }
  }

  /** The anchor time as the contract's {@code ts} scalar, in nanoseconds. */
  private static Map<String, Object> timestamp(Object anchor) {
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
  private static String seed(Object value) {
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
  private static Map<String, Object> values(MaterializeOptions options) {
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
  private static List<String> inferenceAsked(Object config) {
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

  private static Object read(byte[] json) {
    try {
      return Json.read(json);
    } catch (Json.JsonException e) {
      throw new StructuredDataException(
          "the core answered JSON this host cannot read", "INTERNAL_ERROR", Map.of(), null);
    }
  }

  @Override
  public String toString() {
    return "PredictiveQuery(" + text + ")";
  }
}
