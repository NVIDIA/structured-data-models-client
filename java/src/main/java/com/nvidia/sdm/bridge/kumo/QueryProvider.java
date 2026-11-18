// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.bridge.kumo;

import ai.kumo.data.DataAccess;
import ai.kumo.data.Expr;
import ai.kumo.data.Query;
import ai.kumo.data.Rows;
import ai.kumo.graph.Column;
import ai.kumo.graph.Graph;
import ai.kumo.graph.Table;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataProvider;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.OptionalRequest;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Answers the core's requests through a prototype {@link DataAccess}: one {@link Query} per
 * request (one per column for {@code time_bounds}), and per-seed windows, order and limits in
 * Java, as the core's own in-memory provider applies them.
 */
public final class QueryProvider implements DataProvider {
  private final Map<String, Table> tables = new LinkedHashMap<>();
  private final DataAccess access;
  private final BridgeOptions options;

  public QueryProvider(Graph graph, DataAccess access, BridgeOptions options) {
    for (Table table : graph.tables()) {
      tables.put(table.name(), table);
    }
    this.access = access;
    this.options = options;
  }

  @Override
  public Capabilities capabilities() {
    return Capabilities.builder()
        .answers(
            OptionalRequest.PREVIEW,
            OptionalRequest.COUNT_ROWS,
            OptionalRequest.TIME_BOUNDS,
            OptionalRequest.NEIGHBORS,
            OptionalRequest.LOOKUP)
        .maxKeysPerRequest(options.maxKeysPerQuery())
        .fallback(options.fallbackMaxRows(), options.fallbackMaxBytes())
        .build();
  }

  @Override
  public ArrowReader answer(
      DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
    return switch (request) {
      case DataRequest.Describe describe -> describe(describe, expected.orElseThrow(), allocator);
      case DataRequest.Scan scan -> rows(
          scan.table(), scan.columns(), scan.filters(), scan.limit().isPresent() ? (int) Math.min(Integer.MAX_VALUE, scan.limit().getAsLong()) : null, "scan", allocator);
      case DataRequest.Preview preview -> rows(
          preview.table(), preview.columns(), List.of(), (int) Math.min(Integer.MAX_VALUE, preview.limit()), "preview", allocator);
      case DataRequest.CountRows count -> countRows(count, allocator);
      case DataRequest.TimeBounds bounds -> timeBounds(bounds, expected.orElseThrow(), allocator);
      case DataRequest.Neighbors neighbors -> neighbors(neighbors, allocator);
      case DataRequest.Lookup lookup -> lookup(lookup, allocator);
      default -> throw new UnsupportedOperationException(
          "the bridge does not answer " + request.name() + "; the core answers it from scan");
    };
  }

  // --- describe ------------------------------------------------------------------------------

  private ArrowReader describe(DataRequest.Describe describe, Schema schema, BufferAllocator allocator)
      throws Exception {
    List<Object[]> rows = new ArrayList<>();
    List<Table> asked = new ArrayList<>();
    if (describe.tables().isEmpty()) {
      asked.addAll(tables.values());
    } else {
      for (DataRequest.Relation relation : describe.tables()) {
        Table table = tables.get(relation.table());
        if (table != null) {
          asked.add(table);
        }
      }
    }
    switch (describe.part()) {
      case TABLES -> asked.forEach(table -> rows.add(new Object[] {table.name()}));
      case COLUMNS -> {
        for (Table table : asked) {
          for (Column column : table.columns()) {
            rows.add(
                new Object[] {
                  column.name(),
                  ai.kumo.graph.Values.dtype(column.type()),
                  false,
                  table.primaryKey().contains(column.name())
                });
          }
        }
      }
      case KEYS -> {
        for (Table table : asked) {
          if (!table.primaryKey().isEmpty()) {
            rows.add(new Object[] {"primary", table.primaryKey()});
          }
        }
      }
      case FOREIGN_KEYS, INDEXES, INDEX_COLUMNS, NAMESPACE, KEY_ENFORCEMENT -> {
        // None: relationships are declared through KumoBridge.options, and the source has no
        // namespace, indexes or enforced keys the core reads.
      }
      case VIEW_DEFINITION -> throw new UnsupportedOperationException("a prototype graph has no views");
    }
    return Answers.reader(schema, rows, allocator);
  }

  // --- scan, preview, count ---------------------------------------------------------------------

  private ArrowReader rows(
      DataRequest.Relation relation,
      List<String> columns,
      List<DataRequest.Filter> filters,
      Integer limit,
      String purpose,
      BufferAllocator allocator)
      throws Exception {
    Table table = table(relation);
    Schema schema = schema(table, columns);
    Optional<Expr> predicate = predicate(table, filters);
    if (predicate == null) {
      return Answers.reader(schema, List.of(), allocator);
    }
    Rows read = execute(query(purpose, table, columns, predicate.orElse(null), limit));
    List<Object[]> rows = new ArrayList<>();
    for (Map<String, Object> row : read.values()) {
      Object[] values = new Object[columns.size()];
      for (int i = 0; i < columns.size(); i++) {
        values[i] = row.get(columns.get(i));
      }
      rows.add(values);
    }
    return Answers.reader(schema, rows, allocator);
  }

  private ArrowReader countRows(DataRequest.CountRows count, BufferAllocator allocator)
      throws Exception {
    Table table = table(count.table());
    Schema schema = new Schema(List.of(new Field("rows", FieldType.notNullable(new ArrowType.Int(64, true)), null)));
    Optional<Expr> predicate = predicate(table, count.filters());
    long counted = 0;
    if (predicate != null) {
      Map<String, Expr> projection = new LinkedHashMap<>();
      projection.put("rows", new Expr.Call(Expr.Function.COUNT, List.of()));
      Rows read =
          execute(new Query(purpose("count_rows"), table.name(), null, projection, List.of(), predicate.orElse(null), List.of(), null));
      if (!read.values().isEmpty() && read.values().get(0).get("rows") != null) {
        counted = ((Number) read.values().get(0).get("rows")).longValue();
      }
    }
    List<Object[]> rows = new ArrayList<>();
    rows.add(new Object[] {counted});
    return Answers.reader(schema, rows, allocator);
  }

  private ArrowReader timeBounds(DataRequest.TimeBounds bounds, Schema schema, BufferAllocator allocator)
      throws Exception {
    List<Object[]> rows = new ArrayList<>();
    for (int position = 0; position < bounds.columns().size(); position++) {
      DataRequest.TableColumn column = bounds.columns().get(position);
      Table table = table(column.table());
      Expr micros = new Expr.Call(Expr.Function.TIME_MICROS, List.of(new Expr.Field(column.column())));
      Map<String, Expr> projection = new LinkedHashMap<>();
      projection.put("min", new Expr.Call(Expr.Function.MIN, List.of(micros)));
      projection.put("max", new Expr.Call(Expr.Function.MAX, List.of(micros)));
      Rows read =
          execute(new Query(purpose("time_bounds"), table.name(), null, projection, List.of(), null, List.of(), null));
      Map<String, Object> row = read.values().isEmpty() ? Map.of() : read.values().get(0);
      rows.add(new Object[] {position, row.get("min"), row.get("max")});
    }
    return Answers.reader(schema, rows, allocator);
  }

  // --- neighbors and lookup ------------------------------------------------------------------

  /** One seed: its row in the key set, its key, its batch, and its window bounds (micros). */
  private record Seed(int index, List<Object> key, Object batch, Long lower, Long upper) {}

  private ArrowReader neighbors(DataRequest.Neighbors request, BufferAllocator allocator)
      throws Exception {
    Table table = table(request.table());
    Optional<DataRequest.Window> window = request.window();
    List<Seed> seeds =
        seeds(
            request.seeds(),
            request.key().size(),
            window.flatMap(DataRequest.Window::lower).map(DataRequest.SeedBound::seedColumn),
            window.flatMap(DataRequest.Window::upper).map(DataRequest.SeedBound::seedColumn),
            allocator);
    Field batchField = batchField(request.seeds(), allocator, seeds);
    List<Field> fields = new ArrayList<>();
    fields.add(batchField);
    fields.addAll(schema(table, request.columns()).getFields());
    Schema schema = new Schema(fields);
    if (seeds.isEmpty()) {
      return Answers.reader(schema, List.of(), allocator);
    }

    // One query: the seeds' keys, within the widest window any seed asks.
    Set<String> asked = new LinkedHashSet<>(request.key());
    asked.addAll(request.columns());
    window.ifPresent(w -> asked.add(w.column()));
    request.order().forEach(term -> asked.add(term.column()));
    List<Expr> terms = new ArrayList<>();
    terms.add(keysIn(table, request.key(), seeds.stream().map(Seed::key).toList()));
    if (window.isPresent()) {
      Expr time = micros(window.get().column());
      window.get().lower().ifPresent(bound -> widest(seeds, Seed::lower, true)
          .ifPresent(at -> terms.add(new Expr.Compare(time, bound.inclusive() ? Expr.Op.GE : Expr.Op.GT, new Expr.Literal(at)))));
      window.get().upper().ifPresent(bound -> widest(seeds, Seed::upper, false)
          .ifPresent(at -> terms.add(new Expr.Compare(time, bound.inclusive() ? Expr.Op.LE : Expr.Op.LT, new Expr.Literal(at)))));
    }
    Rows read = execute(query("neighbors", table, new ArrayList<>(asked), new Expr.And(terms), null));
    Map<List<String>, List<Map<String, Object>>> byKey = byKey(read, request.key());

    Long limit = null;
    if (request.perSeed().isPresent()) {
      limit = request.perSeed().get() instanceof DataRequest.Count count ? count.count() : 0L;
    }
    Comparator<Map<String, Object>> order = order(request.order());
    // Each seed's rows within its window; with a limit, the first rows of each batch in order.
    TreeMap<Object, List<Map<String, Object>>> perBatch = new TreeMap<>(QueryProvider::compareBatch);
    List<Object[]> pairs = new ArrayList<>();
    for (Seed seed : seeds) {
      for (Map<String, Object> row : byKey.getOrDefault(identity(seed.key()), List.of())) {
        if (window.isPresent() && !within(row, window.get(), seed)) {
          continue;
        }
        if (limit == null) {
          pairs.add(answerRow(seed.batch(), row, request.columns()));
        } else {
          perBatch.computeIfAbsent(seed.batch(), b -> new ArrayList<>()).add(row);
        }
      }
    }
    if (limit != null) {
      for (Map.Entry<Object, List<Map<String, Object>>> batch : perBatch.entrySet()) {
        List<Map<String, Object>> rows = new ArrayList<>(batch.getValue());
        rows.sort(order);
        for (int i = 0; i < Math.min(limit, rows.size()); i++) {
          pairs.add(answerRow(batch.getKey(), rows.get(i), request.columns()));
        }
      }
    }
    return Answers.reader(schema, pairs, allocator);
  }

  private ArrowReader lookup(DataRequest.Lookup request, BufferAllocator allocator) throws Exception {
    Table table = table(request.table());
    List<Seed> seeds = seeds(request.seeds(), request.key().size(), Optional.empty(), Optional.empty(), allocator);
    List<Field> fields = new ArrayList<>();
    fields.add(batchField(request.seeds(), allocator, seeds));
    fields.add(new Field(DataRequest.MATCHES_COLUMN, FieldType.notNullable(new ArrowType.Int(64, true)), null));
    fields.addAll(schema(table, request.columns()).getFields());
    Schema schema = new Schema(fields);
    if (seeds.isEmpty()) {
      return Answers.reader(schema, List.of(), allocator);
    }
    Set<String> asked = new LinkedHashSet<>(request.key());
    asked.addAll(request.columns());
    Rows read =
        execute(query("lookup", table, new ArrayList<>(asked), keysIn(table, request.key(), seeds.stream().map(Seed::key).toList()), null));
    Map<List<String>, List<Map<String, Object>>> byKey = byKey(read, request.key());
    // One row per batch holding a match, in the order batches first match, with its count.
    Map<String, Object[]> perBatch = new LinkedHashMap<>();
    Map<String, Long> counts = new HashMap<>();
    for (Seed seed : seeds) {
      List<Map<String, Object>> matched = byKey.getOrDefault(identity(seed.key()), List.of());
      if (matched.isEmpty()) {
        continue;
      }
      String batch = String.valueOf(seed.batch());
      counts.merge(batch, (long) matched.size(), Long::sum);
      perBatch.computeIfAbsent(batch, b -> {
        Object[] row = new Object[2 + request.columns().size()];
        row[0] = seed.batch();
        for (int i = 0; i < request.columns().size(); i++) {
          row[2 + i] = matched.get(0).get(request.columns().get(i));
        }
        return row;
      });
    }
    List<Object[]> rows = new ArrayList<>();
    for (Map.Entry<String, Object[]> entry : perBatch.entrySet()) {
      entry.getValue()[1] = counts.get(entry.getKey());
      rows.add(entry.getValue());
    }
    return Answers.reader(schema, rows, allocator);
  }

  /** Every seed of the key set, in order: its key columns, batch, and window bound columns. */
  private static List<Seed> seeds(
      DataRequest.KeySet keys,
      int keyColumns,
      Optional<String> lowerColumn,
      Optional<String> upperColumn,
      BufferAllocator allocator)
      throws Exception {
    List<String> keyNames = new ArrayList<>();
    if (keyColumns == 1) {
      keyNames.add(DataRequest.SEED_KEY);
    } else {
      for (int i = 1; i <= keyColumns; i++) {
        keyNames.add(DataRequest.SEED_KEY + "_" + i);
      }
    }
    List<Seed> seeds = new ArrayList<>();
    ArrowReader reader = keys.reader();
    VectorSchemaRoot root = reader.getVectorSchemaRoot();
    while (reader.loadNextBatch()) {
      List<FieldVector> key = keyNames.stream().map(root::getVector).toList();
      FieldVector batch = root.getVector(DataRequest.SEED_BATCH);
      FieldVector lower = lowerColumn.map(root::getVector).orElse(null);
      FieldVector upper = upperColumn.map(root::getVector).orElse(null);
      for (int row = 0; row < root.getRowCount(); row++) {
        List<Object> values = new ArrayList<>();
        for (FieldVector vector : key) {
          values.add(Answers.plain(vector, row));
        }
        seeds.add(
            new Seed(
                seeds.size(),
                values,
                Answers.plain(batch, row),
                lower == null ? null : (Long) Answers.plain(lower, row),
                upper == null ? null : (Long) Answers.plain(upper, row)));
      }
    }
    return seeds;
  }

  /** The answer's batch column: the seed key set's own {@code batch} type. */
  private static Field batchField(DataRequest.KeySet keys, BufferAllocator allocator, List<Seed> seeds)
      throws java.io.IOException {
    Field batch = keys.reader().getVectorSchemaRoot().getSchema().findField(DataRequest.SEED_BATCH);
    return new Field(DataRequest.BATCH_COLUMN, new FieldType(false, batch.getType(), null), null);
  }

  private static Optional<Long> widest(List<Seed> seeds, java.util.function.Function<Seed, Long> bound, boolean lower) {
    Long widest = null;
    for (Seed seed : seeds) {
      Long at = bound.apply(seed);
      if (at == null) {
        continue;
      }
      widest = widest == null ? at : lower ? Math.min(widest, at) : Math.max(widest, at);
    }
    return Optional.ofNullable(widest);
  }

  private static boolean within(Map<String, Object> row, DataRequest.Window window, Seed seed) {
    Object value = row.get(window.column());
    if (value == null) {
      return false;
    }
    long time = Answers.micros(value);
    if (window.lower().isPresent()) {
      if (seed.lower() == null) {
        return false;
      }
      if (window.lower().get().inclusive() ? time < seed.lower() : time <= seed.lower()) {
        return false;
      }
    }
    if (window.upper().isPresent()) {
      if (seed.upper() == null) {
        return false;
      }
      return window.upper().get().inclusive() ? time <= seed.upper() : time < seed.upper();
    }
    return true;
  }

  private static Comparator<Map<String, Object>> order(List<DataRequest.TieTerm> terms) {
    Comparator<Map<String, Object>> comparator = (a, b) -> 0;
    for (DataRequest.TieTerm term : terms) {
      Comparator<Map<String, Object>> one =
          (a, b) -> {
            Object x = a.get(term.column());
            Object y = b.get(term.column());
            if (x == null || y == null) {
              if (x == y) {
                return 0;
              }
              // Nulls go last or first whatever the direction, as the term says.
              return (x == null) == term.nullsLast() ? 1 : -1;
            }
            int compared = compareValues(x, y, term.asTimestamp());
            return term.descending() ? -compared : compared;
          };
      comparator = comparator.thenComparing(one);
    }
    return comparator;
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static int compareValues(Object x, Object y, boolean asTimestamp) {
    if (asTimestamp) {
      return Long.compare(Answers.micros(x), Answers.micros(y));
    }
    if (x instanceof Number a && y instanceof Number b) {
      return new BigDecimal(a.toString()).compareTo(new BigDecimal(b.toString()));
    }
    if (x instanceof Comparable a && x.getClass().isInstance(y)) {
      return a.compareTo(y);
    }
    return x.toString().compareTo(y.toString());
  }

  private static int compareBatch(Object a, Object b) {
    if (a instanceof Number x && b instanceof Number y) {
      return Long.compare(x.longValue(), y.longValue());
    }
    return String.valueOf(a).compareTo(String.valueOf(b));
  }

  private static Object[] answerRow(Object batch, Map<String, Object> row, List<String> columns) {
    Object[] values = new Object[1 + columns.size()];
    values[0] = batch;
    for (int i = 0; i < columns.size(); i++) {
      values[1 + i] = row.get(columns.get(i));
    }
    return values;
  }

  private static Map<List<String>, List<Map<String, Object>>> byKey(Rows read, List<String> key) {
    Map<List<String>, List<Map<String, Object>>> byKey = new HashMap<>();
    for (Map<String, Object> row : read.values()) {
      List<Object> values = new ArrayList<>();
      for (String column : key) {
        values.add(row.get(column));
      }
      if (values.contains(null)) {
        continue;
      }
      byKey.computeIfAbsent(identity(values), k -> new ArrayList<>()).add(row);
    }
    return byKey;
  }

  /** A key's values compared by meaning: integral numbers of any width equal, text as text. */
  private static List<String> identity(List<Object> key) {
    List<String> out = new ArrayList<>(key.size());
    for (Object value : key) {
      if (value instanceof Number number) {
        BigDecimal decimal = new BigDecimal(number.toString()).stripTrailingZeros();
        out.add("n:" + decimal.toPlainString());
      } else {
        out.add("s:" + value);
      }
    }
    return out;
  }

  // --- predicates and queries ----------------------------------------------------------------

  /** The filters as one predicate; empty for none; {@code null} when no row can match. */
  private Optional<Expr> predicate(Table table, List<DataRequest.Filter> filters) throws Exception {
    List<Expr> terms = new ArrayList<>();
    for (DataRequest.Filter filter : filters) {
      switch (filter) {
        case DataRequest.KeyIn in -> {
          List<List<Object>> keys = keyRows(in.keys());
          if (keys.isEmpty()) {
            return null;
          }
          terms.add(keysIn(table, in.columns(), keys));
        }
        case DataRequest.TimeRange range -> {
          Expr time = micros(range.column());
          range.lower().ifPresent(bound -> terms.add(new Expr.Compare(time, bound.inclusive() ? Expr.Op.GE : Expr.Op.GT, new Expr.Literal(Answers.micros(bound.value())))));
          range.upper().ifPresent(bound -> terms.add(new Expr.Compare(time, bound.inclusive() ? Expr.Op.LE : Expr.Op.LT, new Expr.Literal(Answers.micros(bound.value())))));
        }
        case DataRequest.NotNull notNull -> terms.add(
            new Expr.Compare(new Expr.Field(notNull.column()), Expr.Op.IS_NOT, new Expr.Literal(null)));
      }
    }
    if (terms.isEmpty()) {
      return Optional.empty();
    }
    return Optional.of(terms.size() == 1 ? terms.get(0) : new Expr.And(terms));
  }

  private static List<List<Object>> keyRows(DataRequest.KeySet keys) throws Exception {
    List<List<Object>> rows = new ArrayList<>();
    ArrowReader reader = keys.reader();
    VectorSchemaRoot root = reader.getVectorSchemaRoot();
    while (reader.loadNextBatch()) {
      for (int row = 0; row < root.getRowCount(); row++) {
        List<Object> values = new ArrayList<>();
        for (String column : keys.columns()) {
          values.add(Answers.plain(root.getVector(column), row));
        }
        rows.add(values);
      }
    }
    return rows;
  }

  /** {@code columns IN keys}: one {@code IN} for one column, else an {@code OR} of {@code AND}s. */
  private static Expr keysIn(Table table, List<String> columns, List<List<Object>> keys) {
    Set<List<String>> seen = new LinkedHashSet<>();
    List<List<Object>> distinct = new ArrayList<>();
    for (List<Object> key : keys) {
      if (!key.contains(null) && seen.add(identity(key))) {
        distinct.add(key);
      }
    }
    if (columns.size() == 1) {
      Column.Type type = type(table, columns.get(0));
      List<Expr> values = new ArrayList<>();
      for (List<Object> key : distinct) {
        values.add(new Expr.Literal(literal(key.get(0), type)));
      }
      return new Expr.In(new Expr.Field(columns.get(0)), values);
    }
    List<Expr> any = new ArrayList<>();
    for (List<Object> key : distinct) {
      List<Expr> all = new ArrayList<>();
      for (int i = 0; i < columns.size(); i++) {
        all.add(new Expr.Compare(new Expr.Field(columns.get(i)), Expr.Op.EQ, new Expr.Literal(literal(key.get(i), type(table, columns.get(i))))));
      }
      any.add(new Expr.And(all));
    }
    return new Expr.Or(any);
  }

  /** A key value as the prototype's literals carry it for the column's type. */
  private static Object literal(Object value, Column.Type type) {
    if (value == null) {
      return null;
    }
    return switch (type) {
      case INT64 -> value instanceof Number number ? (Object) number.longValue() : (Object) Long.parseLong(value.toString());
      case FLOAT64 -> value instanceof Number number ? number.doubleValue() : Double.parseDouble(value.toString());
      case STRING -> value instanceof Number number && !(value instanceof Double || value instanceof Float)
          ? new BigDecimal(number.toString()).toPlainString() : value.toString();
      case TIMESTAMP -> Answers.micros(value);
      default -> value;
    };
  }

  private static Expr micros(String column) {
    return new Expr.Call(Expr.Function.TIME_MICROS, List.of(new Expr.Field(column)));
  }

  private Query query(String purpose, Table table, List<String> columns, Expr predicate, Integer limit) {
    Map<String, Expr> projection = new LinkedHashMap<>();
    for (String column : columns) {
      projection.put(column, new Expr.Field(column));
    }
    return new Query(purpose(purpose), table.name(), null, projection, List.of(), predicate, List.of(), limit);
  }

  private String purpose(String request) {
    return options.purposePrefix() + request;
  }

  private Rows execute(Query query) {
    return access.execute(query);
  }

  private Table table(DataRequest.Relation relation) {
    Table table = tables.get(relation.table());
    if (table == null) {
      throw new IllegalArgumentException("no table " + relation.table() + " in the graph");
    }
    return table;
  }

  private static Column.Type type(Table table, String column) {
    for (Column found : table.columns()) {
      if (found.name().equals(column)) {
        return found.type();
      }
    }
    throw new IllegalArgumentException("no column " + table.name() + "." + column + " in the graph");
  }

  private static Schema schema(Table table, List<String> columns) {
    List<Field> fields = new ArrayList<>();
    for (String column : columns) {
      fields.add(Answers.field(column, type(table, column)));
    }
    return new Schema(fields);
  }
}
