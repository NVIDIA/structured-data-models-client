// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.Source;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.DataRequest.Bound;
import com.nvidia.sdm.provider.DataRequest.Filter;
import com.nvidia.sdm.provider.DataRequest.KeySet;
import com.nvidia.sdm.provider.DataRequest.Relation;
import com.nvidia.sdm.provider.DataRequest.Value;
import com.nvidia.sdm.provider.FailureFacts;
import com.nvidia.sdm.provider.OptionalRequest;
import com.nvidia.sdm.provider.ProviderFailure;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Base64;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.function.Function;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * The one translation between the core's wire format and the public source types ({@code
 * com.nvidia.sdm.provider}), both ways, and the run loop's provider over every {@link Source}, as
 * Python's {@code host_providers}.
 *
 * <p>Read: a fetch's request (the contract's {@code DataRequest}) into the public {@link
 * DataRequest} it is, each key set it names one of the core's Arrow attachments; the schema its
 * answer must have; a source's capabilities and failures. Written: a public request back into the
 * contract's JSON, its key sets by attachment id, as a built-in source hands it to its SQL kit; a
 * source's {@link Capabilities} and {@link ProviderFailure} as the core reads them. Each way is the
 * other's inverse: a request read and written again is the request the core sent.
 *
 * <p>It decides nothing: which requests a source is sent, which filters, and a key set too large
 * for one request split into several, are the core's, alike for every host and every source. Not
 * part of the API.
 */
public final class HostProviders {
  private HostProviders() {}

  /**
   * The core's provider choice for {@code source}, and the run loop's provider answering its
   * fetches. Its capabilities are read once, at the start of the operation.
   */
  public static Providers.Chosen chosen(Source source) {
    Capabilities declared = source.capabilities();
    if (declared == null) {
      throw new IllegalStateException("Source.capabilities() returned null");
    }
    Map<String, Object> choice = new LinkedHashMap<>();
    choice.put("provider", "host");
    choice.put("capabilities", capabilities(declared));
    return new Providers.Chosen(choice, new Adapter(source));
  }

  /**
   * A request a source refused by name, with the error the core's table holds (a built-in source's
   * SQL kit refusing one its engine cannot answer): the run hands it back to the core as the
   * source's refusal ({@code FetchOutcome::Refused}), which the operation then fails with.
   */
  static final class Refused extends Exception {
    private static final long serialVersionUID = 1L;

    private final transient Map<String, Object> error;

    Refused(Map<String, Object> error) {
      super("the source refused the request", null, false, false);
      this.error = error;
    }

    Map<String, Object> error() {
      return error;
    }
  }

  /** The run loop's provider over a source: each fetch read into a request and asked of it. */
  static final class Adapter implements Host.Provider {
    private final Source source;

    Adapter(Source source) {
      this.source = source;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ArrowReader answer(
        Map<String, Object> step, Map<Integer, ArrowReader> keys, BufferAllocator allocator)
        throws Exception {
      Map<String, Object> plan = (Map<String, Object>) step.get("plan");
      Optional<Schema> expected = expected((Map<String, Object>) plan.get("expect"));
      Map<String, Object> wire = (Map<String, Object>) step.get("request");
      DataRequest request = Requests.read(wire, id -> keySet(keys, id));
      try {
        return source.answer(request, expected, allocator);
      } catch (NativeError refused) {
        // The core's own error, raised by the source: its refusal by name.
        Map<String, Object> control = (Map<String, Object>) Json.read(refused.message().control());
        throw new Refused((Map<String, Object>) control.get("error"));
      }
    }

    @Override
    public Map<String, Object> failure(Exception failure) {
      return HostProviders.failure(source.failure(failure));
    }

    private static ArrowReader keySet(Map<Integer, ArrowReader> keys, int id) {
      ArrowReader reader = keys.get(id);
      if (reader == null) {
        throw Messages.internal("a request names a key set the core did not give");
      }
      return reader;
    }
  }

  // --- capabilities ------------------------------------------------------------------------

  /**
   * The contract's {@code ProviderCapabilities} of {@code declared}, written as the core writes
   * them: each field left out where the core leaves it out at its default.
   */
  public static Map<String, Object> capabilities(Capabilities declared) {
    Map<String, Object> wire = new LinkedHashMap<>();
    wire.put("optional", declared.optional().stream().map(OptionalRequest::wireName).sorted().toList());
    wire.put(
        "max_keys_per_request",
        declared.maxKeysPerRequest().isPresent() ? declared.maxKeysPerRequest().getAsLong() : null);
    wire.put("sampling", named(declared.sampling()));
    wire.put("max_seed", declared.maxSeed());
    wire.put("text_times", declared.textTimes().map(HostProviders::named).orElse(null));
    wire.put("null_order", declared.nullOrder().map(HostProviders::named).orElse(null));
    wire.put("index_advice", declared.indexAdvice().map(HostProviders::named).orElse(null));
    Map<String, Object> optional = new LinkedHashMap<>();
    optional.put(
        "filters", declared.filters().stream().map(DataRequest.FilterKind::wireName).sorted().toList());
    optional.put(
        "fallback",
        declared
            .fallback()
            .map(
                limits -> {
                  Map<String, Object> fallback = new LinkedHashMap<>();
                  fallback.put("max_rows", limits.maxRows());
                  fallback.put("max_bytes", limits.maxBytes());
                  return fallback;
                })
            .orElse(null));
    optional.put(
        "fallback_text_order", declared.fallbackTextOrder().map(HostProviders::named).orElse(null));
    optional.put("derived_relations", declared.derivedRelations());
    optional.put("sample_rows", declared.sampleByRowCount());
    optional.put("neighbor_lookback", declared.neighborLookback());
    optional.put("keys_optional", declared.keysOptional());
    optional.put("key_enforcement", declared.keyEnforcement());
    optional.put("opaque_vendor", declared.opaqueVendor().orElse(null));
    optional.put("compares_stored_times", declared.comparesStoredTimes());
    optional.put("samples_by_key_dtype", declared.sampleByKeyType());
    optional.put("reads_nan_as_missing", declared.nanIsMissing());
    optional.put("database", declared.engine().orElse(null));
    optional.put("column_affinities", declared.columnTypesAreAffinities());
    optional.put("reads_tables_as_they_stand", declared.rereadTablesEveryRun());
    optional.put("namespace", declared.namespace());
    optional.put("tries_upper_case_names", declared.retryUpperCaseNames());
    optional.put("namespace_first", declared.askNamespaceFirst());
    optional.put(
        "view",
        declared
            .semanticModel()
            .map(
                model -> {
                  Map<String, Object> view = new LinkedHashMap<>();
                  view.put("name", model.name());
                  view.put("trust_expressions", model.trustExpressions());
                  return view;
                })
            .orElse(null));
    optional.forEach(
        (name, given) -> {
          boolean left =
              given == null
                  || Boolean.FALSE.equals(given)
                  || (given instanceof List<?> list && list.isEmpty());
          if (!left) {
            wire.put(name, given);
          }
        });
    return wire;
  }

  /** The contract's {@code ProviderCapabilities} as the public {@link Capabilities} it declares. */
  @SuppressWarnings("unchecked")
  public static Capabilities readCapabilities(Map<String, Object> wire) {
    Capabilities.Builder builder = Capabilities.builder();
    for (Object name : (List<Object>) wire.get("optional")) {
      builder.answers(member(OptionalRequest.class, (String) name));
    }
    if (wire.get("filters") instanceof List<?> filters) {
      for (Object name : filters) {
        builder.filters(member(DataRequest.FilterKind.class, (String) name));
      }
    }
    if (wire.get("max_keys_per_request") instanceof Number max) {
      builder.maxKeysPerRequest(max.longValue());
    }
    builder.sampling(member(Capabilities.Sampling.class, (String) wire.get("sampling")));
    builder.maxSeed(new BigInteger(wire.get("max_seed").toString()));
    builder.nullOrder(member(Capabilities.NullOrder.class, (String) wire.get("null_order")));
    if (wire.get("fallback") instanceof Map<?, ?> fallback) {
      builder.fallback(
          ((Number) fallback.get("max_rows")).longValue(),
          ((Number) fallback.get("max_bytes")).longValue());
    }
    builder.fallbackTextOrder(
        member(Capabilities.TextOrder.class, (String) wire.get("fallback_text_order")));
    builder.textTimes(member(Capabilities.TextTimes.class, (String) wire.get("text_times")));
    builder.indexAdvice(member(Capabilities.IndexAdvice.class, (String) wire.get("index_advice")));
    builder.engine((String) wire.get("database"));
    builder.namespace(Requests.strings(wire.get("namespace")));
    builder.askNamespaceFirst(Boolean.TRUE.equals(wire.get("namespace_first")));
    builder.retryUpperCaseNames(Boolean.TRUE.equals(wire.get("tries_upper_case_names")));
    builder.columnTypesAreAffinities(Boolean.TRUE.equals(wire.get("column_affinities")));
    builder.rereadTablesEveryRun(Boolean.TRUE.equals(wire.get("reads_tables_as_they_stand")));
    if (wire.get("view") instanceof Map<?, ?> view) {
      builder.semanticModel(
          new Capabilities.SemanticModel(
              (String) view.get("name"), Boolean.TRUE.equals(view.get("trust_expressions"))));
    }
    builder.derivedRelations(Boolean.TRUE.equals(wire.get("derived_relations")));
    builder.sampleByRowCount(Boolean.TRUE.equals(wire.get("sample_rows")));
    builder.sampleByKeyType(Boolean.TRUE.equals(wire.get("samples_by_key_dtype")));
    builder.neighborLookback(Boolean.TRUE.equals(wire.get("neighbor_lookback")));
    builder.keysOptional(Boolean.TRUE.equals(wire.get("keys_optional")));
    builder.keyEnforcement(Boolean.TRUE.equals(wire.get("key_enforcement")));
    builder.comparesStoredTimes(Boolean.TRUE.equals(wire.get("compares_stored_times")));
    builder.nanIsMissing(Boolean.TRUE.equals(wire.get("reads_nan_as_missing")));
    builder.opaqueVendor((String) wire.get("opaque_vendor"));
    return builder.build();
  }

  /** An enumeration member by its contract name ({@code seeded_stable}); null for none. */
  private static <E extends Enum<E>> E member(Class<E> kind, String name) {
    return name == null ? null : Enum.valueOf(kind, name.toUpperCase(Locale.ROOT));
  }

  /** An enumeration member's contract name. */
  private static String named(Enum<?> member) {
    return member.name().toLowerCase(Locale.ROOT);
  }

  // --- failures ----------------------------------------------------------------------------

  /**
   * The fields of the contract's {@code ProviderError} {@code told} says, each where it says one:
   * the facts the core reads the failure by among them.
   */
  public static Map<String, Object> failure(ProviderFailure told) {
    Map<String, Object> fields = new LinkedHashMap<>();
    told.sqlState().ifPresent(text -> fields.put("sqlstate", text));
    told.vendorCode().ifPresent(text -> fields.put("vendor_code", text));
    told.errorClass().ifPresent(text -> fields.put("error_class", text));
    told.message().ifPresent(text -> fields.put("driver_message", text));
    told.facts().ifPresent(facts -> fields.put("facts", facts(facts)));
    return fields;
  }

  /** The contract's {@code ProviderError} as the public failure it tells. */
  @SuppressWarnings("unchecked")
  public static ProviderFailure readFailure(Map<String, Object> wire) {
    return new ProviderFailure(
        Optional.ofNullable((String) wire.get("sqlstate")),
        Optional.ofNullable((String) wire.get("vendor_code")),
        Optional.ofNullable((String) wire.get("error_class")),
        Optional.ofNullable((String) wire.get("driver_message")),
        Optional.ofNullable((Map<String, Object>) wire.get("facts")).map(HostProviders::readFacts));
  }

  private static Map<String, Object> facts(FailureFacts facts) {
    Map<String, Object> wire = new LinkedHashMap<>();
    wire.put("not_found", facts.notFound());
    wire.put("refuses_expression", facts.refusesExpression());
    wire.put("unusable", facts.unusable().map(HostProviders::named).orElse(null));
    wire.put("denied", facts.denied());
    wire.put("reason", facts.reason().map(HostProviders::named).orElse(null));
    return wire;
  }

  private static FailureFacts readFacts(Map<String, Object> wire) {
    return new FailureFacts(
        Boolean.TRUE.equals(wire.get("not_found")),
        Boolean.TRUE.equals(wire.get("refuses_expression")),
        Optional.ofNullable(member(FailureFacts.Unusable.class, (String) wire.get("unusable"))),
        Boolean.TRUE.equals(wire.get("denied")),
        Optional.ofNullable(member(FailureFacts.Reason.class, (String) wire.get("reason"))));
  }

  // --- the expected schema -----------------------------------------------------------------

  /**
   * The schema an answer must have, where its plan fixes one; empty where the answer holds the
   * source's own columns.
   */
  @SuppressWarnings("unchecked")
  public static Optional<Schema> expected(Map<String, Object> expect) {
    if (expect == null || !"schema".equals(expect.get("expect"))) {
      return Optional.empty();
    }
    List<Field> fields = new ArrayList<>();
    for (Object field : (List<Object>) expect.get("fields")) {
      fields.add(Fields.read((Map<String, Object>) field));
    }
    return Optional.of(new Schema(fields));
  }

  // --- requests ----------------------------------------------------------------------------

  /** The contract's {@code DataRequest} JSON, read into the public records and written back. */
  public static final class Requests {
    private Requests() {}

    /** The request {@code wire} is, each key set it names the reader {@code keys} gives by id. */
    @SuppressWarnings("unchecked")
    public static DataRequest read(Map<String, Object> wire, Function<Integer, ArrowReader> keys) {
      String kind = (String) wire.get("request");
      return switch (kind) {
        case "describe" -> new DataRequest.Describe(
            relations((List<Object>) wire.get("tables")),
            member(DataRequest.DescribePart.class, (String) wire.get("part")));
        case "scan" -> new DataRequest.Scan(
            relation(wire.get("table")),
            strings(wire.get("columns")),
            filters(wire.get("filters"), keys),
            optionalLong(wire.get("limit")));
        case "preview" -> new DataRequest.Preview(
            relation(wire.get("table")), strings(wire.get("columns")), longValue(wire.get("limit")));
        case "count_rows" -> new DataRequest.CountRows(
            relation(wire.get("table")), filters(wire.get("filters"), keys));
        case "key_counts" -> {
          List<DataRequest.TableKey> list = new ArrayList<>();
          int position = 0;
          for (Object item : (List<Object>) wire.get("keys")) {
            Map<String, Object> key = (Map<String, Object>) item;
            list.add(
                new DataRequest.TableKey(
                    relation(key.get("table")),
                    strings(key.get("columns")),
                    key.get("label") == null ? position : longValue(key.get("label")),
                    Boolean.TRUE.equals(key.get("unique"))));
            position++;
          }
          yield new DataRequest.KeyCounts(list);
        }
        case "time_bounds" -> {
          List<DataRequest.TableColumn> list = new ArrayList<>();
          for (Object item : (List<Object>) wire.get("columns")) {
            Map<String, Object> column = (Map<String, Object>) item;
            list.add(
                new DataRequest.TableColumn(
                    relation(column.get("table")), (String) column.get("column")));
          }
          yield new DataRequest.TimeBounds(list);
        }
        case "column_ranges" -> new DataRequest.ColumnRanges(
            relation(wire.get("table")), strings(wire.get("columns")));
        case "repeated_key" -> new DataRequest.RepeatedKey(
            relation(wire.get("table")), strings(wire.get("key")));
        case "sample_entities" -> new DataRequest.SampleEntities(
            relation(wire.get("table")),
            strings(wire.get("key")),
            longValue(wire.get("n")),
            Values.read((Map<String, Object>) wire.get("seed")),
            strings(wire.get("columns")),
            filters(wire.get("filters"), keys),
            optionalLong(wire.get("rows")),
            Optional.ofNullable((String) wire.get("key_dtype")));
        case "neighbors" -> new DataRequest.Neighbors(
            relation(wire.get("table")),
            strings(wire.get("key")),
            strings(wire.get("columns")),
            keySet(wire.get("seeds"), keys),
            Optional.ofNullable((Map<String, Object>) wire.get("window")).map(Requests::window),
            Optional.ofNullable(wire.get("per_seed")).map(Requests::limit),
            order(wire.get("order")));
        case "lookup" -> new DataRequest.Lookup(
            relation(wire.get("table")),
            strings(wire.get("key")),
            strings(wire.get("columns")),
            keySet(wire.get("seeds"), keys));
        default -> throw Messages.internal("the core sent a data request this host does not know");
      };
    }

    /**
     * {@code request} as the contract's {@code DataRequest} JSON, each key set it names put in
     * {@code keys} by the attachment id it names it by: what {@link #read} reads back as {@code
     * request}.
     */
    public static Map<String, Object> write(
        DataRequest request, Map<Integer, ArrowReader> keys) {
      KeySets sets = new KeySets(keys);
      Map<String, Object> wire = new LinkedHashMap<>();
      wire.put("request", request.name());
      if (request instanceof DataRequest.Describe r) {
        wire.put("tables", r.tables().stream().map(Requests::write).toList());
        wire.put("part", named(r.part()));
      } else if (request instanceof DataRequest.Scan r) {
        wire.put("table", write(r.table()));
        wire.put("columns", r.columns());
        wire.put("filters", writeFilters(r.filters(), sets));
        wire.put("limit", r.limit().isPresent() ? r.limit().getAsLong() : null);
      } else if (request instanceof DataRequest.Preview r) {
        wire.put("table", write(r.table()));
        wire.put("columns", r.columns());
        wire.put("limit", r.limit());
      } else if (request instanceof DataRequest.CountRows r) {
        wire.put("table", write(r.table()));
        wire.put("filters", writeFilters(r.filters(), sets));
      } else if (request instanceof DataRequest.KeyCounts r) {
        List<Object> list = new ArrayList<>();
        for (int at = 0; at < r.keys().size(); at++) {
          DataRequest.TableKey key = r.keys().get(at);
          Map<String, Object> written = new LinkedHashMap<>();
          written.put("table", write(key.table()));
          written.put("columns", key.columns());
          // A key's label is its position unless the request names another.
          if (key.label() != at) {
            written.put("label", key.label());
          }
          if (key.unique()) {
            written.put("unique", true);
          }
          list.add(written);
        }
        wire.put("keys", list);
      } else if (request instanceof DataRequest.TimeBounds r) {
        List<Object> list = new ArrayList<>();
        for (DataRequest.TableColumn column : r.columns()) {
          Map<String, Object> written = new LinkedHashMap<>();
          written.put("table", write(column.table()));
          written.put("column", column.column());
          list.add(written);
        }
        wire.put("columns", list);
      } else if (request instanceof DataRequest.ColumnRanges r) {
        wire.put("table", write(r.table()));
        wire.put("columns", r.columns());
      } else if (request instanceof DataRequest.RepeatedKey r) {
        wire.put("table", write(r.table()));
        wire.put("key", r.key());
      } else if (request instanceof DataRequest.SampleEntities r) {
        wire.put("table", write(r.table()));
        wire.put("key", r.key());
        wire.put("n", r.n());
        wire.put("seed", Values.write(r.seed()));
        wire.put("columns", r.columns());
        wire.put("filters", writeFilters(r.filters(), sets));
        r.keyDtype().ifPresent(dtype -> wire.put("key_dtype", dtype));
        r.rows().ifPresent(rows -> wire.put("rows", rows));
      } else if (request instanceof DataRequest.Neighbors r) {
        wire.put("table", write(r.table()));
        wire.put("key", r.key());
        wire.put("columns", r.columns());
        wire.put("seeds", sets.write(r.seeds()));
        wire.put("window", r.window().map(Requests::write).orElse(null));
        wire.put("per_seed", r.perSeed().map(Requests::write).orElse(null));
        wire.put(
            "order", r.order().isEmpty() ? null : r.order().stream().map(Requests::write).toList());
      } else if (request instanceof DataRequest.Lookup r) {
        wire.put("table", write(r.table()));
        wire.put("key", r.key());
        wire.put("columns", r.columns());
        wire.put("seeds", sets.write(r.seeds()));
      }
      return wire;
    }

    /** The key sets a written request names, each by the id it is given in order, once each. */
    private static final class KeySets {
      private final Map<Integer, ArrowReader> given;
      private final Map<ArrowReader, Integer> ids = new IdentityHashMap<>();

      KeySets(Map<Integer, ArrowReader> given) {
        this.given = given;
      }

      Map<String, Object> write(KeySet keys) {
        Integer at = ids.get(keys.reader());
        if (at == null) {
          at = ids.size() + 1;
          ids.put(keys.reader(), at);
          given.put(at, keys.reader());
        }
        Map<String, Object> wire = new LinkedHashMap<>();
        wire.put("attachment", (long) at);
        wire.put("columns", keys.schema().getFields().stream().map(Fields::write).toList());
        return wire;
      }
    }

    // --- read ------------------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private static Relation relation(Object wire) {
      Map<String, Object> relation = (Map<String, Object>) wire;
      List<DataRequest.RelationColumn> columns = new ArrayList<>();
      if (relation.get("columns") instanceof List<?> given) {
        for (Object item : given) {
          Map<String, Object> column = (Map<String, Object>) item;
          columns.add(
              new DataRequest.RelationColumn(
                  (String) column.get("name"),
                  Optional.ofNullable((String) column.get("column")),
                  Optional.ofNullable((String) column.get("expression"))));
        }
      }
      return new Relation(
          strings(relation.get("source")),
          columns,
          Optional.ofNullable((String) relation.get("filter")),
          Boolean.TRUE.equals(relation.get("found")));
    }

    private static List<Relation> relations(List<Object> wire) {
      List<Relation> list = new ArrayList<>();
      if (wire != null) {
        for (Object item : wire) {
          list.add(relation(item));
        }
      }
      return list;
    }

    @SuppressWarnings("unchecked")
    static List<String> strings(Object wire) {
      if (wire == null) {
        return List.of();
      }
      List<String> list = new ArrayList<>();
      for (Object item : (List<Object>) wire) {
        list.add((String) item);
      }
      return list;
    }

    @SuppressWarnings("unchecked")
    private static List<Filter> filters(Object wire, Function<Integer, ArrowReader> keys) {
      List<Filter> list = new ArrayList<>();
      if (wire == null) {
        return list;
      }
      for (Object item : (List<Object>) wire) {
        list.add(filter((Map<String, Object>) item, keys));
      }
      return list;
    }

    @SuppressWarnings("unchecked")
    private static Filter filter(Map<String, Object> filter, Function<Integer, ArrowReader> keys) {
      return switch ((String) filter.get("filter")) {
        case "key_in" -> new DataRequest.KeyIn(
            strings(filter.get("columns")), keySet(filter.get("keys"), keys));
        case "time_range" -> new DataRequest.TimeRange(
            (String) filter.get("column"),
            Optional.ofNullable((Map<String, Object>) filter.get("lower")).map(Requests::bound),
            Optional.ofNullable((Map<String, Object>) filter.get("upper")).map(Requests::bound));
        case "not_null" -> new DataRequest.NotNull((String) filter.get("column"));
        case "compare" -> new DataRequest.Compare(
            (String) filter.get("column"),
            member(DataRequest.CompareOp.class, (String) filter.get("op")),
            Values.read((Map<String, Object>) filter.get("value")));
        case "in" -> {
          List<Value> values = new ArrayList<>();
          for (Object value : (List<Object>) filter.get("values")) {
            values.add(Values.read((Map<String, Object>) value));
          }
          yield new DataRequest.In((String) filter.get("column"), values);
        }
        case "is_null" -> new DataRequest.IsNull((String) filter.get("column"));
        case "any_of" -> new DataRequest.AnyOf(filters(filter.get("filters"), keys));
        case "all_of" -> new DataRequest.AllOf(filters(filter.get("filters"), keys));
        case "has_rows" -> new DataRequest.HasRows(
            (String) filter.get("column"),
            relation(filter.get("table")),
            (String) filter.get("key"),
            filters(filter.get("filters"), keys),
            longValue(filter.get("at_least")));
        default -> throw Messages.internal("the core sent a filter this host does not know");
      };
    }

    @SuppressWarnings("unchecked")
    private static Bound bound(Map<String, Object> wire) {
      return new Bound(
          Values.read((Map<String, Object>) wire.get("value")),
          Boolean.TRUE.equals(wire.get("inclusive")));
    }

    @SuppressWarnings("unchecked")
    private static KeySet keySet(Object wire, Function<Integer, ArrowReader> keys) {
      Map<String, Object> keySet = (Map<String, Object>) wire;
      List<Field> columns = new ArrayList<>();
      for (Object column : (List<Object>) keySet.get("columns")) {
        columns.add(Fields.read((Map<String, Object>) column));
      }
      return new KeySet(
          new Schema(columns), keys.apply(((Number) keySet.get("attachment")).intValue()));
    }

    @SuppressWarnings("unchecked")
    private static DataRequest.Window window(Map<String, Object> wire) {
      return new DataRequest.Window(
          (String) wire.get("column"),
          Optional.ofNullable((Map<String, Object>) wire.get("lower")).map(Requests::seedBound),
          Optional.ofNullable((Map<String, Object>) wire.get("upper")).map(Requests::seedBound),
          Optional.ofNullable((String) wire.get("source_type")));
    }

    private static DataRequest.SeedBound seedBound(Map<String, Object> wire) {
      return new DataRequest.SeedBound(
          (String) wire.get("seed_column"), Boolean.TRUE.equals(wire.get("inclusive")));
    }

    @SuppressWarnings("unchecked")
    private static DataRequest.NeighborLimit limit(Object wire) {
      if (wire instanceof Number number) {
        return new DataRequest.Count(number.longValue());
      }
      Map<String, Object> map = (Map<String, Object>) wire;
      if (map.get("negative") != null) {
        return new DataRequest.AllBut((String) map.get("negative"));
      }
      if (map.get("sql_expression") != null) {
        return new DataRequest.SqlLimit((String) map.get("sql_expression"));
      }
      throw Messages.internal("the core sent a per-seed limit this host does not know");
    }

    @SuppressWarnings("unchecked")
    private static List<DataRequest.TieTerm> order(Object wire) {
      List<DataRequest.TieTerm> list = new ArrayList<>();
      if (wire == null) {
        return list;
      }
      for (Object item : (List<Object>) wire) {
        Map<String, Object> term = (Map<String, Object>) item;
        list.add(
            new DataRequest.TieTerm(
                (String) term.get("column"),
                Boolean.TRUE.equals(term.get("as_timestamp")),
                Boolean.TRUE.equals(term.get("descending")),
                Boolean.TRUE.equals(term.get("nulls_last")),
                Boolean.TRUE.equals(term.get("string_like"))));
      }
      return list;
    }

    private static long longValue(Object wire) {
      return ((Number) wire).longValue();
    }

    private static OptionalLong optionalLong(Object wire) {
      return wire == null ? OptionalLong.empty() : OptionalLong.of(longValue(wire));
    }

    // --- write -----------------------------------------------------------------------------

    private static Map<String, Object> write(Relation relation) {
      Map<String, Object> wire = new LinkedHashMap<>();
      wire.put("source", relation.parts());
      if (!relation.columns().isEmpty()) {
        List<Object> columns = new ArrayList<>();
        for (DataRequest.RelationColumn column : relation.columns()) {
          Map<String, Object> written = new LinkedHashMap<>();
          written.put("name", column.name());
          column.column().ifPresent(name -> written.put("column", name));
          column.expression().ifPresent(text -> written.put("expression", text));
          columns.add(written);
        }
        wire.put("columns", columns);
      }
      relation.filter().ifPresent(text -> wire.put("filter", text));
      if (relation.found()) {
        wire.put("found", true);
      }
      return wire;
    }

    private static List<Object> writeFilters(List<Filter> filters, KeySets sets) {
      List<Object> list = new ArrayList<>();
      for (Filter filter : filters) {
        list.add(writeFilter(filter, sets));
      }
      return list;
    }

    private static Map<String, Object> writeFilter(Filter filter, KeySets sets) {
      Map<String, Object> wire = new LinkedHashMap<>();
      if (filter instanceof DataRequest.KeyIn f) {
        wire.put("filter", "key_in");
        wire.put("columns", f.columns());
        wire.put("keys", sets.write(f.keys()));
      } else if (filter instanceof DataRequest.TimeRange f) {
        wire.put("filter", "time_range");
        wire.put("column", f.column());
        wire.put("lower", f.lower().map(Requests::write).orElse(null));
        wire.put("upper", f.upper().map(Requests::write).orElse(null));
      } else if (filter instanceof DataRequest.NotNull f) {
        wire.put("filter", "not_null");
        wire.put("column", f.column());
      } else if (filter instanceof DataRequest.Compare f) {
        wire.put("filter", "compare");
        wire.put("column", f.column());
        wire.put("op", named(f.op()));
        wire.put("value", Values.write(f.value()));
      } else if (filter instanceof DataRequest.In f) {
        wire.put("filter", "in");
        wire.put("column", f.column());
        wire.put("values", f.values().stream().map(Values::write).toList());
      } else if (filter instanceof DataRequest.IsNull f) {
        wire.put("filter", "is_null");
        wire.put("column", f.column());
      } else if (filter instanceof DataRequest.AnyOf f) {
        wire.put("filter", "any_of");
        wire.put("filters", writeFilters(f.filters(), sets));
      } else if (filter instanceof DataRequest.AllOf f) {
        wire.put("filter", "all_of");
        wire.put("filters", writeFilters(f.filters(), sets));
      } else if (filter instanceof DataRequest.HasRows f) {
        wire.put("filter", "has_rows");
        wire.put("column", f.column());
        wire.put("table", write(f.table()));
        wire.put("key", f.key());
        wire.put("filters", writeFilters(f.filters(), sets));
        wire.put("at_least", f.atLeast());
      }
      return wire;
    }

    private static Map<String, Object> write(Bound bound) {
      Map<String, Object> wire = new LinkedHashMap<>();
      wire.put("value", Values.write(bound.value()));
      wire.put("inclusive", bound.inclusive());
      return wire;
    }

    private static Map<String, Object> write(DataRequest.Window window) {
      Map<String, Object> wire = new LinkedHashMap<>();
      wire.put("column", window.column());
      window.sourceType().ifPresent(type -> wire.put("source_type", type));
      wire.put("lower", window.lower().map(Requests::write).orElse(null));
      wire.put("upper", window.upper().map(Requests::write).orElse(null));
      return wire;
    }

    private static Map<String, Object> write(DataRequest.SeedBound bound) {
      Map<String, Object> wire = new LinkedHashMap<>();
      wire.put("seed_column", bound.seedColumn());
      wire.put("inclusive", bound.inclusive());
      return wire;
    }

    private static Object write(DataRequest.NeighborLimit limit) {
      if (limit instanceof DataRequest.Count count) {
        return count.count();
      }
      if (limit instanceof DataRequest.AllBut allBut) {
        return Map.of("negative", allBut.value());
      }
      return Map.of("sql_expression", ((DataRequest.SqlLimit) limit).expression());
    }

    private static Map<String, Object> write(DataRequest.TieTerm term) {
      Map<String, Object> wire = new LinkedHashMap<>();
      if (term.stringLike()) {
        wire.put("string_like", true);
      }
      wire.put("column", term.column());
      wire.put("as_timestamp", term.asTimestamp());
      wire.put("descending", term.descending());
      wire.put("nulls_last", term.nullsLast());
      return wire;
    }

    /** A contract scalar ({@code {"t": ..., "v": ...}}) as the plain Java value it holds. */
    static Object value(Map<String, Object> wire) {
      return Values.read(wire).value();
    }
  }

  // --- values ------------------------------------------------------------------------------

  /** The contract's scalars as {@link Value}s, exactly, and back. */
  static final class Values {
    private Values() {}

    private static final long MICROS = 1_000_000L;
    private static final long NANOS = 1_000_000_000L;

    /** A contract scalar as the plain Java value it holds, and the Arrow type it is held as. */
    static Value read(Map<String, Object> wire) {
      String text = wire.get("v") == null ? null : wire.get("v").toString();
      return switch ((String) wire.get("t")) {
        case "null" -> new Value(null, ArrowType.Null.INSTANCE);
        case "bool" -> new Value(wire.get("v"), ArrowType.Bool.INSTANCE);
        case "i8" -> new Value(Byte.valueOf(text), new ArrowType.Int(8, true));
        case "i16" -> new Value(Short.valueOf(text), new ArrowType.Int(16, true));
        case "i32" -> new Value(Integer.valueOf(text), new ArrowType.Int(32, true));
        case "i64" -> new Value(Long.valueOf(text), new ArrowType.Int(64, true));
        case "u8" -> new Value(Short.valueOf(text), new ArrowType.Int(8, false));
        case "u16" -> new Value(Integer.valueOf(text), new ArrowType.Int(16, false));
        case "u32" -> new Value(Long.valueOf(text), new ArrowType.Int(32, false));
        case "u64" -> new Value(new BigInteger(text), new ArrowType.Int(64, false));
        case "f32" -> new Value(
            (float) floating(text), new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE));
        case "f64" -> new Value(
            floating(text), new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE));
        case "d128", "d256" -> {
          int scale = ((Number) wire.get("s")).intValue();
          int precision = ((Number) wire.get("p")).intValue();
          int width = "d128".equals(wire.get("t")) ? 128 : 256;
          yield new Value(
              new BigDecimal(new BigInteger(text), scale),
              new ArrowType.Decimal(precision, scale, width));
        }
        case "utf8" -> new Value(text, ArrowType.Utf8.INSTANCE);
        case "binary" -> new Value(Base64.getDecoder().decode(text), ArrowType.Binary.INSTANCE);
        case "date32" -> new Value(
            LocalDate.ofEpochDay(Long.parseLong(text)), new ArrowType.Date(DateUnit.DAY));
        case "ts" -> {
          long raw = Long.parseLong(text);
          boolean nanos = "ns".equals(wire.get("unit"));
          long per = nanos ? NANOS : MICROS;
          Instant instant =
              Instant.ofEpochSecond(
                  Math.floorDiv(raw, per), Math.floorMod(raw, per) * (NANOS / per));
          String zone = (String) wire.get("zone");
          yield new Value(
              zone == null ? LocalDateTime.ofInstant(instant, ZoneOffset.UTC) : instant,
              new ArrowType.Timestamp(unit(wire), zone));
        }
        case "duration" -> new Value(
            "ns".equals(wire.get("unit"))
                ? Duration.ofNanos(Long.parseLong(text))
                : Duration.of(Long.parseLong(text), java.time.temporal.ChronoUnit.MICROS),
            new ArrowType.Duration(unit(wire)));
        case "time64" -> new Value(
            LocalTime.ofNanoOfDay(
                "ns".equals(wire.get("unit")) ? Long.parseLong(text) : Long.parseLong(text) * 1000L),
            new ArrowType.Time(unit(wire), 64));
        default -> throw Messages.internal(
            "the core sent a value of a type this host does not know");
      };
    }

    private static TimeUnit unit(Map<String, Object> wire) {
      return "ns".equals(wire.get("unit")) ? TimeUnit.NANOSECOND : TimeUnit.MICROSECOND;
    }

    private static double floating(String text) {
      return switch (text) {
        case "NaN" -> Double.NaN;
        case "inf" -> Double.POSITIVE_INFINITY;
        case "-inf" -> Double.NEGATIVE_INFINITY;
        default -> Double.parseDouble(text);
      };
    }

    /** {@code given} as the contract scalar of its type, exactly: the inverse of {@link #read}. */
    static Map<String, Object> write(Value given) {
      Object value = given.value();
      Map<String, Object> wire = new LinkedHashMap<>();
      if (given.type() instanceof ArrowType.Null n) {
        wire.put("t", "null");
      } else if (given.type() instanceof ArrowType.Bool b) {
        wire.put("t", "bool");
        wire.put("v", value);
      } else if (given.type() instanceof ArrowType.Int i) {
        wire.put("t", (i.getIsSigned() ? "i" : "u") + i.getBitWidth());
        wire.put("v", value.toString());
      } else if (given.type() instanceof ArrowType.FloatingPoint f) {
        boolean single = f.getPrecision() == FloatingPointPrecision.SINGLE;
        wire.put("t", single ? "f32" : "f64");
        wire.put("v", floatText(((Number) value).doubleValue(), single));
      } else if (given.type() instanceof ArrowType.Decimal d) {
        wire.put("t", d.getBitWidth() == 128 ? "d128" : "d256");
        wire.put("p", (long) d.getPrecision());
        wire.put("s", (long) d.getScale());
        wire.put(
            "v",
            ((BigDecimal) value).setScale(d.getScale(), RoundingMode.UNNECESSARY)
                .unscaledValue()
                .toString());
      } else if (given.type() instanceof ArrowType.Utf8 u) {
        wire.put("t", "utf8");
        wire.put("v", value);
      } else if (given.type() instanceof ArrowType.Binary b) {
        wire.put("t", "binary");
        wire.put("v", Base64.getEncoder().encodeToString((byte[]) value));
      } else if (given.type() instanceof ArrowType.Date d) {
        wire.put("t", "date32");
        wire.put("v", Long.toString(((LocalDate) value).toEpochDay()));
      } else if (given.type() instanceof ArrowType.Timestamp t) {
        Instant instant =
            value instanceof LocalDateTime wall ? wall.toInstant(ZoneOffset.UTC) : (Instant) value;
        wire.put("t", "ts");
        wire.put("unit", unitName(t.getUnit()));
        wire.put("zone", t.getTimezone());
        wire.put("v", count(instant.getEpochSecond(), instant.getNano(), t.getUnit()));
      } else if (given.type() instanceof ArrowType.Duration d) {
        Duration span = (Duration) value;
        wire.put("t", "duration");
        wire.put("unit", unitName(d.getUnit()));
        wire.put("v", count(span.getSeconds(), span.getNano(), d.getUnit()));
      } else if (given.type() instanceof ArrowType.Time t) {
        long nanos = ((LocalTime) value).toNanoOfDay();
        wire.put("t", "time64");
        wire.put("unit", unitName(t.getUnit()));
        wire.put("v", Long.toString(t.getUnit() == TimeUnit.NANOSECOND ? nanos : nanos / 1000L));
      } else {
        throw new IllegalArgumentException( "a value of a type the contract does not hold: " + given.type());
      }
      return wire;
    }

    private static String unitName(TimeUnit unit) {
      return unit == TimeUnit.NANOSECOND ? "ns" : "us";
    }

    /** {@code seconds} and {@code nanos} past them, counted in {@code unit}, as digits. */
    private static String count(long seconds, int nanos, TimeUnit unit) {
      BigInteger whole = BigInteger.valueOf(seconds);
      return unit == TimeUnit.NANOSECOND
          ? whole.multiply(BigInteger.valueOf(NANOS)).add(BigInteger.valueOf(nanos)).toString()
          : whole.multiply(BigInteger.valueOf(MICROS)).add(BigInteger.valueOf(nanos / 1000)).toString();
    }

    /**
     * {@code number} as the core writes a float: the shortest digits that read back as it (as a
     * float where {@code single}), plain from 1e-4 up to 1e16 and in exponent form outside, as
     * Rust's {@code {:?}} writes them.
     */
    static String floatText(double number, boolean single) {
      if (Double.isNaN(number)) {
        return "NaN";
      }
      if (Double.isInfinite(number)) {
        return number > 0 ? "inf" : "-inf";
      }
      if (number == 0) {
        return 1 / number < 0 ? "-0.0" : "0.0";
      }
      BigDecimal exact = new BigDecimal(number);
      BigDecimal shortest = exact;
      for (int digits = 1; digits <= 17; digits++) {
        BigDecimal candidate = exact.round(new MathContext(digits, RoundingMode.HALF_EVEN));
        boolean back =
            single
                ? candidate.floatValue() == (float) number
                : candidate.doubleValue() == number;
        if (back) {
          shortest = candidate;
          break;
        }
      }
      shortest = shortest.stripTrailingZeros();
      String minus = shortest.signum() < 0 ? "-" : "";
      String shown = shortest.unscaledValue().abs().toString();
      // The power of ten of the leading digit.
      int point = shown.length() - shortest.scale() - 1;
      double magnitude = Math.abs(number);
      if (magnitude >= 1e-4 && magnitude < 1e16) {
        int whole = point + 1;
        if (whole <= 0) {
          return minus + "0." + "0".repeat(-whole) + shown;
        }
        if (whole >= shown.length()) {
          return minus + shown + "0".repeat(whole - shown.length()) + ".0";
        }
        return minus + shown.substring(0, whole) + "." + shown.substring(whole);
      }
      String mantissa = shown.length() == 1 ? shown : shown.charAt(0) + "." + shown.substring(1);
      return minus + mantissa + "e" + point;
    }
  }

  // --- fields ------------------------------------------------------------------------------

  /** The contract's fields as Arrow fields, and back. */
  static final class Fields {
    private Fields() {}

    @SuppressWarnings("unchecked")
    static Field read(Map<String, Object> field) {
      String name = (String) field.get("name");
      boolean nullable = Boolean.TRUE.equals(field.get("nullable"));
      Map<String, String> metadata =
          field.get("metadata") instanceof Map<?, ?> given ? (Map<String, String>) given : null;
      Map<String, Object> type = (Map<String, Object>) field.get("type");
      List<Field> children = new ArrayList<>();
      ArrowType arrow;
      switch ((String) type.get("type")) {
        case "null" -> arrow = ArrowType.Null.INSTANCE;
        case "bool" -> arrow = ArrowType.Bool.INSTANCE;
        case "int8" -> arrow = new ArrowType.Int(8, true);
        case "int16" -> arrow = new ArrowType.Int(16, true);
        case "int32" -> arrow = new ArrowType.Int(32, true);
        case "int64" -> arrow = new ArrowType.Int(64, true);
        case "uint8" -> arrow = new ArrowType.Int(8, false);
        case "uint16" -> arrow = new ArrowType.Int(16, false);
        case "uint32" -> arrow = new ArrowType.Int(32, false);
        case "uint64" -> arrow = new ArrowType.Int(64, false);
        case "float32" -> arrow = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
        case "float64" -> arrow = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
        case "decimal128" -> arrow =
            new ArrowType.Decimal(number(type, "precision"), number(type, "scale"), 128);
        case "decimal256" -> arrow =
            new ArrowType.Decimal(number(type, "precision"), number(type, "scale"), 256);
        case "utf8" -> arrow = ArrowType.Utf8.INSTANCE;
        case "large_utf8" -> arrow = ArrowType.LargeUtf8.INSTANCE;
        case "binary" -> arrow = ArrowType.Binary.INSTANCE;
        case "large_binary" -> arrow = ArrowType.LargeBinary.INSTANCE;
        case "date32" -> arrow = new ArrowType.Date(DateUnit.DAY);
        case "timestamp" -> arrow =
            new ArrowType.Timestamp(Values.unit(type), (String) type.get("zone"));
        case "duration" -> arrow = new ArrowType.Duration(Values.unit(type));
        case "time64" -> arrow = new ArrowType.Time(Values.unit(type), 64);
        case "list" -> {
          arrow = ArrowType.List.INSTANCE;
          children.add(read((Map<String, Object>) type.get("item")));
        }
        case "struct" -> {
          arrow = ArrowType.Struct.INSTANCE;
          for (Object child : (List<Object>) type.get("fields")) {
            children.add(read((Map<String, Object>) child));
          }
        }
        default -> throw Messages.internal("a column of a type a host answer cannot hold");
      }
      return new Field(name, new FieldType(nullable, arrow, null, metadata), children);
    }

    private static int number(Map<String, Object> map, String name) {
      return ((Number) map.get(name)).intValue();
    }

    /** {@code field} as the contract writes a field: the inverse of {@link #read}. */
    static Map<String, Object> write(Field field) {
      Map<String, Object> wire = new LinkedHashMap<>();
      wire.put("name", field.getName());
      wire.put("type", type(field));
      wire.put("nullable", field.isNullable());
      if (field.getMetadata() != null && !field.getMetadata().isEmpty()) {
        wire.put("metadata", new LinkedHashMap<>(field.getMetadata()));
      }
      return wire;
    }

    private static Map<String, Object> type(Field field) {
      Map<String, Object> wire = new LinkedHashMap<>();
      if (field.getType() instanceof ArrowType.Null n) {
        wire.put("type", "null");
      } else if (field.getType() instanceof ArrowType.Bool b) {
        wire.put("type", "bool");
      } else if (field.getType() instanceof ArrowType.Int i) {
        wire.put("type", (i.getIsSigned() ? "int" : "uint") + i.getBitWidth());
      } else if (field.getType() instanceof ArrowType.FloatingPoint f) {
        wire.put( "type", f.getPrecision() == FloatingPointPrecision.SINGLE ? "float32" : "float64");
      } else if (field.getType() instanceof ArrowType.Decimal d) {
        wire.put("type", d.getBitWidth() == 128 ? "decimal128" : "decimal256");
        wire.put("precision", (long) d.getPrecision());
        wire.put("scale", (long) d.getScale());
      } else if (field.getType() instanceof ArrowType.Utf8 u) {
        wire.put("type", "utf8");
      } else if (field.getType() instanceof ArrowType.LargeUtf8 u) {
        wire.put("type", "large_utf8");
      } else if (field.getType() instanceof ArrowType.Binary b) {
        wire.put("type", "binary");
      } else if (field.getType() instanceof ArrowType.LargeBinary b) {
        wire.put("type", "large_binary");
      } else if (field.getType() instanceof ArrowType.Date d) {
        wire.put("type", "date32");
      } else if (field.getType() instanceof ArrowType.Timestamp t) {
        wire.put("type", "timestamp");
        wire.put("unit", Values.unitName(t.getUnit()));
        wire.put("zone", t.getTimezone());
      } else if (field.getType() instanceof ArrowType.Duration d) {
        wire.put("type", "duration");
        wire.put("unit", Values.unitName(d.getUnit()));
      } else if (field.getType() instanceof ArrowType.Time t) {
        wire.put("type", "time64");
        wire.put("unit", Values.unitName(t.getUnit()));
      } else if (field.getType() instanceof ArrowType.List l) {
        wire.put("type", "list");
        wire.put("item", write(field.getChildren().get(0)));
      } else if (field.getType() instanceof ArrowType.Struct s) {
        wire.put("type", "struct");
        wire.put("fields", field.getChildren().stream().map(Fields::write).toList());
      } else {
        throw new IllegalArgumentException( "a column of a type the contract does not hold: " + field.getType());
      }
      return wire;
    }
  }
}
