// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest.FilterKind;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.OptionalRequest;
import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.VarCharVector;
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
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * A caller's own {@link Source} over Trino, a query engine that declares no primary or
 * foreign keys, answering the core's requests through Trino's HTTP protocol.
 *
 * <p>The provider answers {@code describe}, {@code scan} and {@code sample_entities} itself, and
 * leaves the rest to the core's in-memory fallback. It is run twice: declaring every value filter
 * kind, and declaring only comparisons, {@code IN}, {@code IS NULL} and {@code AND}, as an engine
 * without {@code OR} or subqueries would. Either way each query keeps exactly the examples the
 * context fixture holds for it: the core checks every row with the query's own rules, and a filter
 * the provider does not declare is never sent to it.
 *
 * <p>Runs where {@code SDM_TRINO_TEST_URL} names a Trino server with the memory connector, such as
 * {@code docker run -p 8080:8080 trinodb/trino}; the test makes and drops a schema of its own.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@EnabledIfEnvironmentVariable(named = "SDM_TRINO_TEST_URL", matches = ".+")
class TrinoProviderTest {
  private static final String CATALOG = "memory";
  private final String schemaName = "sdm_" + UUID.randomUUID().toString().replace("-", "");
  private final Trino trino = new Trino(System.getenv("SDM_TRINO_TEST_URL"));

  private static final Set<FilterKind> EVERY = EnumSet.allOf(FilterKind.class);
  private static final Set<FilterKind> NO_OR_NO_SUBQUERY =
      EnumSet.of(FilterKind.COMPARE, FilterKind.IN, FilterKind.IS_NULL, FilterKind.ALL_OF);

  @BeforeAll
  void load() throws Exception {
    trino.run("CREATE SCHEMA " + CATALOG + "." + schemaName);
    Map<String, Object> fixture =
        ShopFixtures.load(Path.of("..", "tests", "compatibility", "fixtures", "context.json"));
    Map<String, String> types =
        Map.of(
            "users", "id BIGINT, value DOUBLE, class VARCHAR, flag BOOLEAN",
            "events", "event BIGINT, id BIGINT, feature DOUBLE, event_time TIMESTAMP(6), item BIGINT",
            "items", "item BIGINT, price DOUBLE");
    @SuppressWarnings("unchecked")
    Map<String, Map<String, Object>> tables = (Map<String, Map<String, Object>>) fixture.get("tables");
    for (var table : tables.entrySet()) {
      String name = qualified(table.getKey());
      trino.run("CREATE TABLE " + name + " (" + types.get(table.getKey()) + ")");
      @SuppressWarnings("unchecked")
      List<List<Object>> rows = (List<List<Object>>) table.getValue().get("rows");
      for (int start = 0; start < rows.size(); start += 500) {
        String values =
            rows.subList(start, Math.min(rows.size(), start + 500)).stream()
                .map(
                    row ->
                        row.stream()
                            .map(
                                v ->
                                    v instanceof String text && text.matches("\\d{4}-\\d\\d-\\d\\dT.*")
                                        ? literal(LocalDateTime.parse(text))
                                        : literal(v))
                            .collect(Collectors.joining(", ", "(", ")")))
                .collect(Collectors.joining(", "));
        trino.run("INSERT INTO " + name + " VALUES " + values);
      }
    }
  }

  @AfterAll
  void drop() throws Exception {
    for (String table : List.of("users", "events", "items"))
      trino.run("DROP TABLE IF EXISTS " + qualified(table));
    trino.run("DROP SCHEMA IF EXISTS " + CATALOG + "." + schemaName);
  }

  /** The compatibility manifest's context scenarios, each with both declarations. */
  @SuppressWarnings("unchecked")
  static Stream<Arguments> queries() throws Exception {
    Map<String, Object> manifest =
        (Map<String, Object>)
            Json.read(Files.readAllBytes(Path.of("..", "tests", "compatibility", "manifest.json")));
    return ((List<Object>) manifest.get("scenarios"))
        .stream()
        .map(s -> (Map<String, Object>) s)
        .filter(s -> ((String) s.get("id")).startsWith("context."))
        .flatMap(
            s -> Stream.of(Arguments.of("every kind", s), Arguments.of("no OR or subquery", s)));
  }

  @ParameterizedTest(name = "{0}: {1}")
  @MethodSource("queries")
  @SuppressWarnings("unchecked")
  void eachQueryKeepsExactlyTheExamplesTheSourceHolds(String declared, Map<String, Object> scenario)
      throws Exception {
    Set<FilterKind> kinds = declared.equals("every kind") ? EVERY : NO_OR_NO_SUBQUERY;
    var source = new TrinoSource(kinds);
    var schema =
        com.nvidia.sdm.Schema.fromSource(
                source,
                // Trino declares no keys: the caller names them.
                FromSourceOptions.builder()
                    .table("users")
                    .table("events")
                    .table("items")
                    .primaryKey("users", "id")
                    .primaryKey("events", "event")
                    .primaryKey("items", "item")
                    .timeColumn("events", "event_time")
                    .build())
            .withRelationships(
                List.of(
                    new Relationship("events", "id", "users", "id"),
                    new Relationship("events", "item", "items", "item")))
            .withStype("users", "value", "numerical");
    var options = MaterializeOptions.builder();
    ((Map<String, Object>) scenario.get("options"))
        .forEach(
            (key, value) -> {
              switch (key) {
                case "indices" -> options.indices((List<Object>) value);
                case "random_seed" -> options.randomSeed(((Number) value).longValue());
                case "num_context_samples" -> options.numContextSamples(((Number) value).intValue());
                case "candidates" -> options.candidates((List<Object>) value);
                case "anchor_time" ->
                    options.anchorTime(java.time.LocalDate.parse((String) value).atStartOfDay());
                default -> throw new IllegalArgumentException("unknown option " + key);
              }
            });
    Request request =
        PredictiveQuery.parse((String) scenario.get("pql")).materialize(schema, options.build());

    // A filter reaches the provider only where it declares the kinds the scenario's filter is
    // written with (sent_with).
    Map<String, Object> context = (Map<String, Object>) scenario.get("context");
    Set<String> declaredNames =
        kinds.stream().map(FilterKind::wireName).collect(Collectors.toSet());
    boolean sent =
        (Boolean) context.get("filtered_at_source")
            && declaredNames.containsAll((List<Object>) context.get("sent_with"));
    long asked = ((Number) context.get("asked")).longValue();
    var wanted =
        ((List<Object>) context.get("kept"))
            .stream()
                .map(k -> List.<Object>of(((Number) k).longValue(), asked, context.get("stopped"), sent))
                .toList();
    var report = request.contextReport();
    assertEquals("available", report.status());
    var got =
        report.parts().stream()
            .map(d -> List.<Object>of(d.kept(), d.asked(), d.stopped(), d.filteredAtSource()))
            .toList();
    assertEquals(wanted, got);
  }

  private String qualified(String table) {
    return CATALOG + "." + schemaName + "." + quoted(table);
  }

  static String quoted(String name) {
    return "\"" + name.replace("\"", "\"\"") + "\"";
  }

  /** A caller's provider over Trino: what a caller writes once for an engine of its own. */
  final class TrinoSource extends Source {
    private final Set<FilterKind> kinds;

    TrinoSource(Set<FilterKind> kinds) {
      this.kinds = kinds;
    }

    @Override
    public Capabilities capabilities() {
      return Capabilities.builder()
          .answers(OptionalRequest.SAMPLE_ENTITIES)
          .filters(kinds.toArray(FilterKind[]::new))
          .fallback(5_000_000, 1L << 30)
          .build();
    }

    @Override
    public ArrowReader answer(
        DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
      if (request instanceof DataRequest.Describe describe)
        return describe(describe, expected.orElseThrow(), allocator);
      if (request instanceof DataRequest.Scan scan)
        return select(
            "SELECT " + columns(scan.columns()) + " FROM " + table(scan.table())
                + where(scan.filters())
                + (scan.limit().isPresent() ? " LIMIT " + scan.limit().getAsLong() : ""),
            allocator);
      if (request instanceof DataRequest.SampleEntities sample)
        // A seeded sample: the rows ordered by a hash of their key and the seed.
        return select(
            "SELECT " + columns(sample.columns()) + " FROM " + table(sample.table())
                + where(sample.filters())
                + " ORDER BY xxhash64(to_utf8(concat("
                + sample.key().stream()
                    .map(k -> "CAST(" + quoted(k) + " AS VARCHAR), '|'")
                    .collect(Collectors.joining(", "))
                + ", '" + sample.seed() + "'))) LIMIT " + sample.n(),
            allocator);
      throw new UnsupportedOperationException(request.name());
    }

    private ArrowReader describe(
        DataRequest.Describe describe, Schema expected, BufferAllocator allocator) throws Exception {
      List<List<Object>> rows =
          switch (describe.part()) {
            case TABLES ->
                trino.rows(
                    "SELECT table_name FROM " + CATALOG + ".information_schema.tables"
                        + " WHERE table_schema = " + literal(schemaName) + " ORDER BY table_name");
            case COLUMNS ->
                trino
                    .rows(
                        "SELECT column_name, data_type, is_nullable = 'NO' FROM "
                            + CATALOG + ".information_schema.columns WHERE table_schema = "
                            + literal(schemaName) + " AND table_name = "
                            + literal(last(describe.tables().get(0)))
                            + " ORDER BY ordinal_position")
                    .stream()
                    .map(row -> List.<Object>of(row.get(0), row.get(1), row.get(2), false))
                    .toList();
            case NAMESPACE -> List.of(List.of(CATALOG), List.of(schemaName));
            // No keys, foreign keys or indexes: Trino declares none.
            default -> List.of();
          };
      return arrow(expected, rows, allocator);
    }

    private String table(DataRequest.Relation relation) {
      return relation.parts().size() == 1
          ? qualified(relation.parts().get(0))
          : relation.parts().stream().map(TrinoProviderTest::quoted).collect(Collectors.joining("."));
    }

    private String columns(List<String> names) {
      return names.stream().map(TrinoProviderTest::quoted).collect(Collectors.joining(", "));
    }

    private String where(List<DataRequest.Filter> filters) throws Exception {
      var conditions = new ArrayList<String>();
      for (var filter : filters) {
        String condition = condition(filter);
        if (condition != null) conditions.add(condition);
      }
      return conditions.isEmpty() ? "" : " WHERE " + String.join(" AND ", conditions);
    }

    /** The filter as Trino SQL; none for one this provider leaves to the core, which widens the read. */
    private String condition(DataRequest.Filter filter) throws Exception {
      if (filter instanceof DataRequest.NotNull f) return quoted(f.column()) + " IS NOT NULL";
      if (filter instanceof DataRequest.IsNull f) return quoted(f.column()) + " IS NULL";
      if (filter instanceof DataRequest.Compare f)
        return quoted(f.column()) + " " + operator(f.op()) + " " + literal(f.value());
      if (filter instanceof DataRequest.In f)
        return quoted(f.column()) + " IN ("
            + f.values().stream().map(TrinoProviderTest::literal).collect(Collectors.joining(", "))
            + ")";
      if (filter instanceof DataRequest.TimeRange f) {
        var bounds = new ArrayList<String>();
        f.lower().ifPresent(b -> bounds.add(quoted(f.column()) + (b.inclusive() ? " >= " : " > ") + literal(b.value())));
        f.upper().ifPresent(b -> bounds.add(quoted(f.column()) + (b.inclusive() ? " <= " : " < ") + literal(b.value())));
        return bounds.isEmpty() ? null : String.join(" AND ", bounds);
      }
      if (filter instanceof DataRequest.AllOf f) return joined(f.filters(), " AND ", false);
      if (filter instanceof DataRequest.AnyOf f) return joined(f.filters(), " OR ", true);
      if (filter instanceof DataRequest.HasRows f) {
        String key = quoted(f.key());
        var within = new ArrayList<String>(List.of(key + " IS NOT NULL"));
        for (var nested : f.filters()) {
          String c = condition(nested);
          if (c != null) within.add(c);
        }
        return quoted(f.column()) + " IN (SELECT " + key + " FROM " + table(f.table())
            + " WHERE " + String.join(" AND ", within)
            + (f.atLeast() > 1 ? " GROUP BY " + key + " HAVING COUNT(*) >= " + f.atLeast() : "")
            + ")";
      }
      if (filter instanceof DataRequest.KeyIn f) return keyIn(f);
      return null;
    }

    /** Each part written; an OR any part of which is not written is not written at all. */
    private String joined(List<DataRequest.Filter> filters, String with, boolean whole)
        throws Exception {
      var parts = new ArrayList<String>();
      for (var nested : filters) {
        String c = condition(nested);
        if (c == null && whole) return null;
        if (c != null) parts.add(c);
      }
      return parts.isEmpty() ? null : "(" + String.join(with, parts) + ")";
    }

    private String keyIn(DataRequest.KeyIn filter) throws Exception {
      var tuples = new ArrayList<String>();
      try (ArrowReader reader = filter.keys().reader()) {
        while (reader.loadNextBatch()) {
          var root = reader.getVectorSchemaRoot();
          for (int at = 0; at < root.getRowCount(); at++) {
            var values = new ArrayList<String>();
            for (String column : filter.keys().columns())
              values.add(literal(root.getVector(column).getObject(at)));
            tuples.add(values.size() == 1 ? values.get(0) : "(" + String.join(", ", values) + ")");
          }
        }
      }
      if (tuples.isEmpty()) return "FALSE";
      String columns =
          filter.columns().size() == 1
              ? quoted(filter.columns().get(0))
              : filter.columns().stream().map(TrinoProviderTest::quoted).collect(Collectors.joining(", ", "(", ")"));
      return columns + " IN (" + String.join(", ", tuples) + ")";
    }

    private static String operator(DataRequest.CompareOp op) {
      return switch (op) {
        case EQ -> "=";
        case NE -> "<>";
        case LT -> "<";
        case LE -> "<=";
        case GT -> ">";
        case GE -> ">=";
      };
    }

    /** A query's rows as Arrow, typed by the columns Trino says it returns. */
    private ArrowReader select(String sql, BufferAllocator allocator) throws Exception {
      Trino.Result result = trino.query(sql);
      var fields = new ArrayList<Field>();
      for (var column : result.columns()) fields.add(field(column.get(0), column.get(1)));
      return arrow(new Schema(fields), result.rows(), allocator);
    }
  }

  static String last(DataRequest.Relation relation) {
    return relation.parts().get(relation.parts().size() - 1);
  }

  static Field field(String name, String trinoType) {
    ArrowType type;
    if (trinoType.equals("bigint")) type = new ArrowType.Int(64, true);
    else if (trinoType.equals("integer")) type = new ArrowType.Int(32, true);
    else if (trinoType.equals("double")) type = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
    else if (trinoType.equals("boolean")) type = ArrowType.Bool.INSTANCE;
    else if (trinoType.equals("date")) type = new ArrowType.Date(DateUnit.DAY);
    else if (trinoType.startsWith("timestamp")) type = new ArrowType.Timestamp(TimeUnit.MICROSECOND, null);
    else type = ArrowType.Utf8.INSTANCE;
    return new Field(name, FieldType.nullable(type), null);
  }

  /** {@code rows} as one batch of {@code schema}, read back as the library reads an answer. */
  static ArrowReader arrow(Schema schema, List<List<Object>> rows, BufferAllocator allocator)
      throws Exception {
    var bytes = new ByteArrayOutputStream();
    try (var root = VectorSchemaRoot.create(schema, allocator);
        var writer = new ArrowStreamWriter(root, null, bytes)) {
      root.allocateNew();
      for (int at = 0; at < rows.size(); at++) {
        for (int c = 0; c < schema.getFields().size(); c++)
          set(root.getVector(c), at, rows.get(at).get(c));
      }
      root.setRowCount(rows.size());
      writer.start();
      writer.writeBatch();
      writer.end();
    }
    return ArrowData.reader(bytes.toByteArray(), allocator);
  }

  static void set(FieldVector vector, int at, Object value) {
    if (value == null) {
      vector.setNull(at);
      return;
    }
    if (vector instanceof BigIntVector v) v.setSafe(at, ((Number) value).longValue());
    else if (vector instanceof IntVector v) v.setSafe(at, ((Number) value).intValue());
    else if (vector instanceof Float8Vector v) v.setSafe(at, ((Number) value).doubleValue());
    else if (vector instanceof BitVector v) v.setSafe(at, (Boolean) value ? 1 : 0);
    else if (vector instanceof DateDayVector v)
      v.setSafe(at, (int) LocalDate.parse(value.toString()).toEpochDay());
    else if (vector instanceof TimeStampMicroVector v) {
      var time = LocalDateTime.parse(value.toString().replace(' ', 'T')).toInstant(ZoneOffset.UTC);
      v.setSafe(at, time.getEpochSecond() * 1_000_000L + time.getNano() / 1000);
    } else if (vector instanceof VarCharVector v)
      v.setSafe(at, value.toString().getBytes(StandardCharsets.UTF_8));
    else throw new IllegalStateException("no Arrow writer for " + vector.getField());
  }

  static String literal(Object value) {
    if (value == null) return "NULL";
    if (value instanceof Boolean b) return b ? "TRUE" : "FALSE";
    if (value instanceof Double d) return "DOUBLE '" + d + "'";
    if (value instanceof Float f) return "REAL '" + f + "'";
    if (value instanceof BigDecimal d) return "DECIMAL '" + d.toPlainString() + "'";
    if (value instanceof Number n) return n.toString();
    if (value instanceof LocalDateTime t)
      return "TIMESTAMP '" + t.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSSSSS")) + "'";
    if (value instanceof Instant t) return literal(LocalDateTime.ofInstant(t, ZoneOffset.UTC));
    if (value instanceof LocalDate d) return "DATE '" + d + "'";
    return "'" + value.toString().replace("'", "''") + "'";
  }

  /** Trino's client protocol: a statement posted, its results followed page by page. */
  record Trino(String url) {
    record Result(List<List<String>> columns, List<List<Object>> rows) {}

    void run(String sql) throws Exception {
      query(sql);
    }

    List<List<Object>> rows(String sql) throws Exception {
      return query(sql).rows();
    }

    @SuppressWarnings("unchecked")
    Result query(String sql) throws Exception {
      HttpClient client = HttpClient.newHttpClient();
      HttpRequest request =
          HttpRequest.newBuilder(URI.create(url + "/v1/statement"))
              .header("X-Trino-User", "sdm")
              .POST(HttpRequest.BodyPublishers.ofString(sql))
              .build();
      List<List<String>> columns = List.of();
      var rows = new ArrayList<List<Object>>();
      while (true) {
        var body =
            (Map<String, Object>)
                Json.read(client.send(request, HttpResponse.BodyHandlers.ofByteArray()).body());
        if (body.get("error") instanceof Map<?, ?> error)
          throw new IllegalStateException("Trino: " + error.get("message") + "\n" + sql);
        if (body.get("columns") instanceof List<?> named)
          columns =
              ((List<Map<String, Object>>) named)
                  .stream().map(c -> List.of((String) c.get("name"), (String) c.get("type"))).toList();
        if (body.get("data") instanceof List<?> data)
          for (Object row : data) rows.add(new ArrayList<>((List<Object>) row));
        if (!(body.get("nextUri") instanceof String next)) return new Result(columns, rows);
        request = HttpRequest.newBuilder(URI.create(next)).header("X-Trino-User", "sdm").GET().build();
      }
    }
  }
}
