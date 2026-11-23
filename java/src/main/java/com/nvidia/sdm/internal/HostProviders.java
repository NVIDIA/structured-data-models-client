// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.ProviderSource;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataProvider;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.DataRequest.Bound;
import com.nvidia.sdm.provider.DataRequest.Filter;
import com.nvidia.sdm.provider.DataRequest.KeySet;
import com.nvidia.sdm.provider.DataRequest.Relation;
import com.nvidia.sdm.provider.OptionalRequest;
import com.nvidia.sdm.provider.ProviderFailure;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.channels.Channels;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.function.Function;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * The bridge between the core's host-provider messages and the public {@link DataProvider}: the
 * provider choice a {@link ProviderSource} names, each fetch's request read into a {@link
 * DataRequest}, the key sets it carries, and a key set larger than the provider's limit split into
 * several calls. Not part of the API.
 */
public final class HostProviders {
  private HostProviders() {}

  /** The core's provider choice and the run loop's provider for {@code source}. */
  public static Providers.Chosen chosen(ProviderSource source) {
    DataProvider provider = source.provider();
    Capabilities capabilities = provider.capabilities();
    Map<String, Object> choice = new LinkedHashMap<>();
    choice.put("provider", "host");
    choice.put("capabilities", capabilities(capabilities));
    return new Providers.Chosen(choice, new Adapter(provider, capabilities));
  }

  /** The contract's {@code ProviderCapabilities} of {@code capabilities}. */
  static Map<String, Object> capabilities(Capabilities capabilities) {
    Map<String, Object> wire = new LinkedHashMap<>();
    wire.put(
        "optional",
        capabilities.optional().stream().map(OptionalRequest::wireName).sorted().toList());
    wire.put(
        "max_keys_per_request",
        capabilities.maxKeysPerRequest().isPresent()
            ? capabilities.maxKeysPerRequest().getAsLong()
            : null);
    wire.put(
        "sampling",
        capabilities.sampling() == Capabilities.Sampling.SEEDED_STABLE ? "seeded_stable" : "unseeded");
    wire.put("max_seed", capabilities.maxSeed());
    if (!capabilities.filters().isEmpty()) {
      wire.put(
          "filters",
          capabilities.filters().stream().map(DataRequest.FilterKind::wireName).sorted().toList());
    }
    wire.put("text_times", null);
    wire.put(
        "null_order",
        capabilities.nullOrder().map(order -> order == Capabilities.NullOrder.FIRST ? "first" : "last").orElse(null));
    wire.put("index_advice", null);
    capabilities
        .fallback()
        .ifPresent(
            limits -> {
              Map<String, Object> fallback = new LinkedHashMap<>();
              fallback.put("max_rows", limits.maxRows());
              fallback.put("max_bytes", limits.maxBytes());
              wire.put("fallback", fallback);
            });
    return wire;
  }

  /**
   * The attachments of the key sets a host fetch's request carries ({@code seeds}, and each {@code
   * key_in} filter's {@code keys}), in order, each once.
   */
  @SuppressWarnings("unchecked")
  static List<Integer> keySetAttachments(Map<String, Object> request) {
    List<Integer> ids = new ArrayList<>();
    if (request == null) {
      return ids;
    }
    if (request.get("seeds") instanceof Map<?, ?> seeds) {
      ids.add(((Number) seeds.get("attachment")).intValue());
    }
    if (request.get("filters") instanceof List<?> filters) {
      for (Object item : filters) {
        Map<String, Object> filter = (Map<String, Object>) item;
        if ("key_in".equals(filter.get("filter"))) {
          int id = ((Number) ((Map<String, Object>) filter.get("keys")).get("attachment")).intValue();
          if (!ids.contains(id)) {
            ids.add(id);
          }
        }
      }
    }
    return ids;
  }

  /** The run loop's provider: reads each fetch into a request and asks the public provider. */
  static final class Adapter implements Host.Provider {
    private final DataProvider provider;
    private final OptionalLong maxKeys;

    Adapter(DataProvider provider, Capabilities capabilities) {
      this.provider = provider;
      this.maxKeys = capabilities.maxKeysPerRequest();
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
      if (maxKeys.isEmpty()) {
        return provider.answer(request, expected, allocator);
      }
      return Split.answer(provider, request, expected, maxKeys.getAsLong(), allocator);
    }

    @Override
    public Map<String, Object> failure(Exception failure) {
      ProviderFailure facts = provider.failure(failure);
      Map<String, Object> told = new LinkedHashMap<>();
      facts.message().ifPresent(text -> told.put("driver_message", text));
      facts.sqlState().ifPresent(text -> told.put("sqlstate", text));
      facts.vendorCode().ifPresent(text -> told.put("vendor_code", text));
      facts.errorClass().ifPresent(text -> told.put("error_class", text));
      return told;
    }

    private static ArrowReader keySet(Map<Integer, ArrowReader> keys, int id) {
      ArrowReader reader = keys.get(id);
      if (reader == null) {
        throw Messages.internal("a request names a key set the core did not give");
      }
      return reader;
    }
  }

  // --- the expected schema -----------------------------------------------------------------

  @SuppressWarnings("unchecked")
  static Optional<Schema> expected(Map<String, Object> expect) {
    if (expect == null || !"schema".equals(expect.get("expect"))) {
      return Optional.empty();
    }
    List<Field> fields = new ArrayList<>();
    for (Object field : (List<Object>) expect.get("fields")) {
      fields.add(field((Map<String, Object>) field));
    }
    return Optional.of(new Schema(fields));
  }

  @SuppressWarnings("unchecked")
  private static Field field(Map<String, Object> field) {
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
      case "decimal128" -> arrow = new ArrowType.Decimal(number(type, "precision"), number(type, "scale"), 128);
      case "decimal256" -> arrow = new ArrowType.Decimal(number(type, "precision"), number(type, "scale"), 256);
      case "utf8" -> arrow = ArrowType.Utf8.INSTANCE;
      case "large_utf8" -> arrow = ArrowType.LargeUtf8.INSTANCE;
      case "binary" -> arrow = ArrowType.Binary.INSTANCE;
      case "large_binary" -> arrow = ArrowType.LargeBinary.INSTANCE;
      case "date32" -> arrow = new ArrowType.Date(DateUnit.DAY);
      case "timestamp" -> arrow = new ArrowType.Timestamp(unit(type), (String) type.get("zone"));
      case "duration" -> arrow = new ArrowType.Duration(unit(type));
      case "time64" -> arrow = new ArrowType.Time(unit(type), 64);
      case "list" -> {
        arrow = ArrowType.List.INSTANCE;
        children.add(field((Map<String, Object>) type.get("item")));
      }
      case "struct" -> {
        arrow = ArrowType.Struct.INSTANCE;
        for (Object child : (List<Object>) type.get("fields")) {
          children.add(field((Map<String, Object>) child));
        }
      }
      default -> throw Messages.internal("an expected column of a type a host answer cannot hold");
    }
    return new Field(name, new FieldType(nullable, arrow, null, metadata), children);
  }

  private static int number(Map<String, Object> map, String name) {
    return ((Number) map.get(name)).intValue();
  }

  private static TimeUnit unit(Map<String, Object> type) {
    return "ns".equals(type.get("unit")) ? TimeUnit.NANOSECOND : TimeUnit.MICROSECOND;
  }

  // --- requests ----------------------------------------------------------------------------

  /** The contract's {@code DataRequest} JSON, read into the public records. */
  static final class Requests {
    private Requests() {}

    @SuppressWarnings("unchecked")
    static DataRequest read(Map<String, Object> wire, Function<Integer, ArrowReader> keys) {
      String kind = (String) wire.get("request");
      return switch (kind) {
        case "describe" -> new DataRequest.Describe(
            relations((List<Object>) wire.get("tables")),
            DataRequest.DescribePart.valueOf(((String) wire.get("part")).toUpperCase(java.util.Locale.ROOT)));
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
            list.add(new DataRequest.TableColumn(relation(column.get("table")), (String) column.get("column")));
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
            value((Map<String, Object>) wire.get("seed")),
            strings(wire.get("columns")),
            filters(wire.get("filters"), keys),
            optionalLong(wire.get("rows")));
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

    @SuppressWarnings("unchecked")
    private static Relation relation(Object wire) {
      Map<String, Object> relation = (Map<String, Object>) wire;
      if (relation.get("columns") instanceof List<?> columns && !columns.isEmpty()
          || relation.get("filter") != null) {
        throw Messages.internal("the core sent a derived relation to a provider that declares none");
      }
      return new Relation(strings(relation.get("source")));
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
    private static List<String> strings(Object wire) {
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
        Map<String, Object> filter = (Map<String, Object>) item;
        switch ((String) filter.get("filter")) {
          case "key_in" -> list.add(
              new DataRequest.KeyIn(strings(filter.get("columns")), keySet(filter.get("keys"), keys)));
          case "time_range" -> list.add(
              new DataRequest.TimeRange(
                  (String) filter.get("column"),
                  Optional.ofNullable((Map<String, Object>) filter.get("lower")).map(Requests::bound),
                  Optional.ofNullable((Map<String, Object>) filter.get("upper")).map(Requests::bound)));
          case "not_null" -> list.add(new DataRequest.NotNull((String) filter.get("column")));
          case "compare" -> list.add(
              new DataRequest.Compare(
                  (String) filter.get("column"),
                  DataRequest.CompareOp.valueOf(((String) filter.get("op")).toUpperCase(java.util.Locale.ROOT)),
                  value((Map<String, Object>) filter.get("value"))));
          case "in" -> {
            List<Object> values = new ArrayList<>();
            for (Object value : (List<Object>) filter.get("values")) {
              values.add(value((Map<String, Object>) value));
            }
            list.add(new DataRequest.In((String) filter.get("column"), values));
          }
          case "is_null" -> list.add(new DataRequest.IsNull((String) filter.get("column")));
          case "any_of" -> list.add(new DataRequest.AnyOf(filters(filter.get("filters"), keys)));
          case "all_of" -> list.add(new DataRequest.AllOf(filters(filter.get("filters"), keys)));
          default -> throw Messages.internal("the core sent a filter this host does not know");
        }
      }
      return list;
    }

    @SuppressWarnings("unchecked")
    private static Bound bound(Map<String, Object> wire) {
      return new Bound(value((Map<String, Object>) wire.get("value")), Boolean.TRUE.equals(wire.get("inclusive")));
    }

    @SuppressWarnings("unchecked")
    private static KeySet keySet(Object wire, Function<Integer, ArrowReader> keys) {
      Map<String, Object> keySet = (Map<String, Object>) wire;
      List<String> columns = new ArrayList<>();
      for (Object column : (List<Object>) keySet.get("columns")) {
        columns.add((String) ((Map<String, Object>) column).get("name"));
      }
      return new KeySet(columns, keys.apply(((Number) keySet.get("attachment")).intValue()));
    }

    @SuppressWarnings("unchecked")
    private static DataRequest.Window window(Map<String, Object> wire) {
      return new DataRequest.Window(
          (String) wire.get("column"),
          Optional.ofNullable((Map<String, Object>) wire.get("lower")).map(Requests::seedBound),
          Optional.ofNullable((Map<String, Object>) wire.get("upper")).map(Requests::seedBound));
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
      throw Messages.internal("the core sent a per-seed limit a host provider cannot answer");
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

    /** A contract scalar ({@code {"t": ..., "v": ...}}) as the plain Java value it holds. */
    static Object value(Map<String, Object> wire) {
      if (wire == null) {
        return null;
      }
      String text = wire.get("v") == null ? null : wire.get("v").toString();
      return switch ((String) wire.get("t")) {
        case "null" -> null;
        case "bool" -> wire.get("v");
        case "i8" -> Byte.valueOf(text);
        case "i16" -> Short.valueOf(text);
        case "i32" -> Integer.valueOf(text);
        case "i64" -> Long.valueOf(text);
        case "u8" -> Short.valueOf(text);
        case "u16" -> Integer.valueOf(text);
        case "u32" -> Long.valueOf(text);
        case "u64" -> new BigInteger(text);
        case "f32" -> Float.valueOf(text);
        case "f64" -> Double.valueOf(text);
        case "d128", "d256" -> new BigDecimal(new BigInteger(text), ((Number) wire.get("s")).intValue());
        case "utf8" -> text;
        case "binary" -> Base64.getDecoder().decode(text);
        case "date32" -> LocalDate.ofEpochDay(Long.parseLong(text));
        case "ts" -> {
          long raw = Long.parseLong(text);
          Instant instant =
              "ns".equals(wire.get("unit"))
                  ? Instant.ofEpochSecond(Math.floorDiv(raw, 1_000_000_000L), Math.floorMod(raw, 1_000_000_000L))
                  : Instant.ofEpochSecond(Math.floorDiv(raw, 1_000_000L), Math.floorMod(raw, 1_000_000L) * 1000);
          yield wire.get("zone") == null ? LocalDateTime.ofInstant(instant, ZoneOffset.UTC) : instant;
        }
        case "duration" -> "ns".equals(wire.get("unit"))
            ? Duration.ofNanos(Long.parseLong(text))
            : Duration.ofNanos(Math.multiplyExact(Long.parseLong(text), 1000L));
        case "time64" -> LocalTime.ofNanoOfDay(
            "ns".equals(wire.get("unit")) ? Long.parseLong(text) : Long.parseLong(text) * 1000L);
        default -> throw Messages.internal("the core sent a value of a type this host does not know");
      };
    }
  }

  // --- splitting by the provider's key limit -------------------------------------------------

  /**
   * A {@code neighbors} or {@code lookup} whose seeds, or a {@code scan} without a limit whose one
   * key filter, hold more keys than the provider takes, asked in parts of at most {@code max} keys
   * each, and the parts' rows given as one answer in order. Any other request is asked whole.
   */
  static final class Split {
    private Split() {}

    static ArrowReader answer(
        DataProvider provider,
        DataRequest request,
        Optional<Schema> expected,
        long max,
        BufferAllocator allocator)
        throws Exception {
      KeySet keys = splittable(request);
      if (keys == null) {
        return provider.answer(request, expected, allocator);
      }
      List<byte[]> parts = parts(keys, max, allocator);
      if (parts.size() <= 1) {
        KeySet whole = new KeySet(keys.columns(), ArrowData.reader(parts.isEmpty() ? empty(keys, allocator) : parts.get(0), allocator));
        return provider.answer(with(request, whole), expected, allocator);
      }
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      ArrowStreamWriter writer = null;
      VectorSchemaRoot written = null;
      try {
        for (byte[] part : parts) {
          KeySet partKeys = new KeySet(keys.columns(), ArrowData.reader(part, allocator));
          try (ArrowReader answer = provider.answer(with(request, partKeys), expected, allocator)) {
            VectorSchemaRoot root = answer.getVectorSchemaRoot();
            while (answer.loadNextBatch()) {
              if (writer == null) {
                written = VectorSchemaRoot.create(root.getSchema(), allocator);
                writer = new ArrowStreamWriter(written, null, Channels.newChannel(out));
                writer.start();
              }
              copy(root, written);
              writer.writeBatch();
            }
            if (writer == null && root.getSchema() != null) {
              written = VectorSchemaRoot.create(root.getSchema(), allocator);
              writer = new ArrowStreamWriter(written, null, Channels.newChannel(out));
              writer.start();
            }
          } finally {
            partKeys.reader().close();
          }
        }
        if (writer != null) {
          writer.end();
        }
      } finally {
        if (writer != null) {
          writer.close();
        }
        if (written != null) {
          written.close();
        }
        keys.reader().close();
      }
      return ArrowData.reader(out.toByteArray(), allocator);
    }

    private static KeySet splittable(DataRequest request) {
      if (request instanceof DataRequest.Neighbors neighbors) {
        return neighbors.seeds();
      }
      if (request instanceof DataRequest.Lookup lookup) {
        return lookup.seeds();
      }
      if (request instanceof DataRequest.Scan scan && scan.limit().isEmpty()) {
        List<DataRequest.KeyIn> keyed =
            scan.filters().stream()
                .filter(DataRequest.KeyIn.class::isInstance)
                .map(DataRequest.KeyIn.class::cast)
                .toList();
        return keyed.size() == 1 ? keyed.get(0).keys() : null;
      }
      return null;
    }

    private static DataRequest with(DataRequest request, KeySet keys) {
      if (request instanceof DataRequest.Neighbors n) {
        return new DataRequest.Neighbors(n.table(), n.key(), n.columns(), keys, n.window(), n.perSeed(), n.order());
      }
      if (request instanceof DataRequest.Lookup l) {
        return new DataRequest.Lookup(l.table(), l.key(), l.columns(), keys);
      }
      DataRequest.Scan scan = (DataRequest.Scan) request;
      List<Filter> filters = new ArrayList<>();
      for (Filter filter : scan.filters()) {
        filters.add(filter instanceof DataRequest.KeyIn in ? new DataRequest.KeyIn(in.columns(), keys) : filter);
      }
      return new DataRequest.Scan(scan.table(), scan.columns(), filters, scan.limit());
    }

    /** The key set's rows, read whole and written again in parts of at most {@code max} rows. */
    private static List<byte[]> parts(KeySet keys, long max, BufferAllocator allocator) throws IOException {
      List<byte[]> parts = new ArrayList<>();
      ArrowReader reader = keys.reader();
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      Schema schema = root.getSchema();
      try (VectorSchemaRoot part = VectorSchemaRoot.create(schema, allocator)) {
        ByteArrayOutputStream out = null;
        ArrowStreamWriter writer = null;
        long inPart = 0;
        while (reader.loadNextBatch()) {
          int rows = root.getRowCount();
          int at = 0;
          while (at < rows) {
            if (writer == null) {
              out = new ByteArrayOutputStream();
              writer = new ArrowStreamWriter(part, null, Channels.newChannel(out));
              writer.start();
              inPart = 0;
            }
            int take = (int) Math.min(rows - at, max - inPart);
            try (VectorSchemaRoot slice = root.slice(at, take)) {
              copy(slice, part);
            }
            writer.writeBatch();
            at += take;
            inPart += take;
            if (inPart == max) {
              writer.end();
              writer.close();
              parts.add(out.toByteArray());
              writer = null;
            }
          }
        }
        if (writer != null) {
          writer.end();
          writer.close();
          parts.add(out.toByteArray());
        }
      }
      return parts;
    }

    private static byte[] empty(KeySet keys, BufferAllocator allocator) throws IOException {
      Schema schema = keys.reader().getVectorSchemaRoot().getSchema();
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
          ArrowStreamWriter writer = new ArrowStreamWriter(root, null, Channels.newChannel(out))) {
        writer.start();
        writer.end();
      }
      return out.toByteArray();
    }

    /** Copies {@code from}'s rows into {@code to}, replacing what it held. */
    private static void copy(VectorSchemaRoot from, VectorSchemaRoot to) {
      to.allocateNew();
      int rows = from.getRowCount();
      for (int column = 0; column < from.getFieldVectors().size(); column++) {
        var source = from.getVector(column);
        var target = to.getVector(column);
        for (int row = 0; row < rows; row++) {
          target.copyFromSafe(row, row, source);
        }
        target.setValueCount(rows);
      }
      to.setRowCount(rows);
    }
  }
}
