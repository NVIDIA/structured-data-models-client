// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.examples.TrinoSource;
import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.DataRequest.FilterKind;
import com.nvidia.sdm.provider.FailureFacts;
import com.nvidia.sdm.provider.ProviderFailure;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The Trino example of your own source ({@link TrinoSource}), as Python's {@code
 * test_trino_source.py}: it builds a schema from Trino's catalog, materializes the compatibility
 * manifest's context scenarios keeping exactly the examples the fixture holds, splits a large key
 * set into lookups of its key limit, and tells the client what Trino's errors mean.
 *
 * <p>Each context scenario runs twice: declaring every value filter kind, and declaring only
 * comparisons, {@code IN}, {@code IS NULL} and {@code AND}, as an engine without {@code OR} or
 * subqueries would. Either way each query keeps exactly the examples the fixture holds for it: the
 * core checks every row with the query's own rules, and a filter the source does not declare is
 * never sent to it.
 *
 * <p>Runs where {@code SDM_TRINO_TEST_URL} names a Trino server with the memory connector, such as
 * {@code docker run -p 8080:8080 trinodb/trino}; CI starts one. The test makes and drops a schema of
 * its own.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@EnabledIfEnvironmentVariable(named = "SDM_TRINO_TEST_URL", matches = ".+")
class TrinoProviderTest {
  private static final String CATALOG = "memory";
  private static final Set<FilterKind> EVERY = EnumSet.allOf(FilterKind.class);
  private static final Set<FilterKind> NO_OR_NO_SUBQUERY =
      EnumSet.of(FilterKind.COMPARE, FilterKind.IN, FilterKind.IS_NULL, FilterKind.ALL_OF);

  private final String schemaName = "sdm_" + UUID.randomUUID().toString().replace("-", "");
  private Connection trino;

  @BeforeAll
  void load() throws Exception {
    String url = System.getenv("SDM_TRINO_TEST_URL").replaceFirst("^https?://", "jdbc:trino://");
    trino = DriverManager.getConnection(url, "sdm", null);
    run("CREATE SCHEMA " + CATALOG + "." + schemaName);
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
      run("CREATE TABLE " + name + " (" + types.get(table.getKey()) + ")");
      @SuppressWarnings("unchecked")
      List<List<Object>> rows = (List<List<Object>>) table.getValue().get("rows");
      for (int start = 0; start < rows.size(); start += 500) {
        String values =
            rows.subList(start, Math.min(rows.size(), start + 500)).stream()
                .map(row -> row.stream().map(TrinoProviderTest::fixtureLiteral).collect(Collectors.joining(", ", "(", ")")))
                .collect(Collectors.joining(", "));
        run("INSERT INTO " + name + " VALUES " + values);
      }
    }
  }

  @AfterAll
  void drop() throws Exception {
    for (String table : List.of("users", "events", "items", "gone"))
      run("DROP TABLE IF EXISTS " + qualified(table));
    run("DROP SCHEMA IF EXISTS " + CATALOG + "." + schemaName);
    trino.close();
  }

  @Test
  void theSchemaIsDescribedFromTrinosCatalog() throws Exception {
    Recording source = new Recording(new TrinoSource(trino, CATALOG, schemaName));
    var schema = shopSchema(source);
    assertEquals(
        Set.of("users", "events", "items"),
        schema.tables().stream().map(Table::name).collect(Collectors.toSet()));
    var events = schema.table("events");
    assertEquals(List.of("event"), events.primaryKey());
    assertEquals(Optional.of("event_time"), events.timeColumn());
    assertEquals(
        List.of("event", "id", "feature", "event_time", "item"),
        events.columns().stream().map(Column::name).toList());
    assertSame(source, schema.source());
    var parts =
        source.asked.stream()
            .filter(r -> r instanceof DataRequest.Describe)
            .map(r -> ((DataRequest.Describe) r).part())
            .collect(Collectors.toSet());
    assertTrue(
        // The tables named, their columns read, and their keys asked of a catalog declaring none.
        parts.containsAll(Set.of(DataRequest.DescribePart.COLUMNS, DataRequest.DescribePart.KEYS)),
        parts.toString());
  }

  /** The compatibility manifest's context scenarios, each with both declarations. */
  @SuppressWarnings("unchecked")
  static Stream<Arguments> queries() throws Exception {
    return scenarios().stream()
        .flatMap(
            s -> Stream.of(Arguments.of("every kind", s), Arguments.of("no OR or subquery", s)));
  }

  @ParameterizedTest(name = "{0}: {1}")
  @MethodSource("queries")
  @SuppressWarnings("unchecked")
  void eachQueryKeepsExactlyTheExamplesTheSourceHolds(String declared, Map<String, Object> scenario)
      throws Exception {
    Set<FilterKind> kinds = declared.equals("every kind") ? EVERY : NO_OR_NO_SUBQUERY;
    Request request = materialize(new TrinoSource(trino, CATALOG, schemaName, kinds, 1_000), scenario);

    // A filter reaches the source only where it declares the kinds the scenario's filter is
    // written with (sent_with).
    Map<String, Object> context = (Map<String, Object>) scenario.get("context");
    Set<String> declaredNames = kinds.stream().map(FilterKind::wireName).collect(Collectors.toSet());
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

  @Test
  void aKeySetPastTheKeyLimitReachesTheSourceInLookupsOfIt() throws Exception {
    Map<String, Object> scenario =
        scenarios().stream()
            .filter(s -> s.get("id").equals("context.static-exhausted"))
            .findFirst()
            .orElseThrow();
    Recording parts = new Recording(new TrinoSource(trino, CATALOG, schemaName, EVERY, 2));
    Recording whole = new Recording(new TrinoSource(trino, CATALOG, schemaName));
    Request split = materialize(parts, scenario);
    Request single = materialize(whole, scenario);
    assertFalse(parts.lookupKeys.isEmpty(), "the source declares lookup, so seeds reach it");
    assertTrue(parts.lookupKeys.stream().allMatch(keys -> keys <= 2), parts.lookupKeys.toString());
    assertTrue(parts.lookupKeys.size() > whole.lookupKeys.size());
    // Answered in parts, a lookup is answered as one.
    assertEquals(single.predictRows(), split.predictRows());
    assertEquals(single.contextReport().parts(), split.contextReport().parts());
  }

  @Test
  void aCountAndAPreviewAnswerAsAScanDoes() throws Exception {
    TrinoSource source = new TrinoSource(trino, CATALOG, schemaName);
    var users = new DataRequest.Relation(List.of("users"));
    List<DataRequest.Filter> over =
        List.of(
            new DataRequest.Compare(
                "value",
                DataRequest.CompareOp.GT,
                new DataRequest.Value(
                    40.0,
                    new org.apache.arrow.vector.types.pojo.ArrowType.FloatingPoint(
                        org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE))));
    try (BufferAllocator allocator = new RootAllocator()) {
      long scanned =
          rows(source.answer(new DataRequest.Scan(users, List.of("id"), over, OptionalLong.empty()), Optional.empty(), allocator)).size();
      List<List<Object>> counted =
          rows(source.answer(new DataRequest.CountRows(users, over), Optional.empty(), allocator));
      assertEquals(List.of(List.<Object>of(scanned)), counted);
      List<List<Object>> preview =
          rows(source.answer(new DataRequest.Preview(users, List.of("id", "class"), 3), Optional.empty(), allocator));
      assertEquals(3, preview.size());
      assertEquals(2, preview.get(0).size());
    }
  }

  /** The rows of an answer, each a list of its values; the reader closed. */
  private static List<List<Object>> rows(ArrowReader answer) throws Exception {
    List<List<Object>> rows = new ArrayList<>();
    try (answer) {
      while (answer.loadNextBatch()) {
        VectorSchemaRoot root = answer.getVectorSchemaRoot();
        for (int at = 0; at < root.getRowCount(); at++) {
          List<Object> row = new ArrayList<>();
          for (var vector : root.getFieldVectors()) row.add(vector.getObject(at));
          rows.add(row);
        }
      }
    }
    return rows;
  }

  @Test
  void trinosErrorsTellTheClientWhatTheyMean() throws Exception {
    TrinoSource source = new TrinoSource(trino, CATALOG, schemaName);
    var scan =
        new DataRequest.Scan(
            new DataRequest.Relation(List.of("gone")), List.of("id"), List.of(), OptionalLong.empty());
    try (BufferAllocator allocator = new RootAllocator()) {
      SQLException raised =
          assertThrows(SQLException.class, () -> source.answer(scan, Optional.empty(), allocator));
      ProviderFailure told = source.failure(raised);
      assertEquals(Optional.of("46"), told.vendorCode()); // TABLE_NOT_FOUND
      assertEquals(
          Optional.of(
              FailureFacts.builder()
                  .notFound()
                  .unusable(FailureFacts.Unusable.MISSING)
                  .reason(FailureFacts.Reason.TABLE_UNRESOLVED)
                  .build()),
          told.facts());
    }
    // Any other failure is read by its message alone.
    assertEquals(Optional.empty(), source.failure(new IllegalStateException("down")).facts());
  }

  @Test
  void aTableDroppedAfterTheSchemaFailsTheOperationByName() throws Exception {
    TrinoSource source = new TrinoSource(trino, CATALOG, schemaName);
    run("CREATE TABLE " + qualified("gone") + " AS SELECT * FROM " + qualified("users"));
    var schema =
        com.nvidia.sdm.Schema.fromSource(
            source, FromSourceOptions.builder().table("gone").primaryKey("gone", "id").build());
    run("DROP TABLE " + qualified("gone"));
    StructuredDataException failed =
        assertThrows(
            StructuredDataException.class,
            () ->
                PredictiveQuery.parse("PREDICT gone.value FOR EACH gone.id")
                    .materialize(schema, MaterializeOptions.builder().indices(List.of(1L)).build()));
    assertEquals("PROVIDER_ERROR", failed.code());
  }

  // --- helpers ---------------------------------------------------------------------------------

  /** The example source, keeping each request it answers, and the seeds of each lookup. */
  static final class Recording implements Source {
    private final TrinoSource source;
    final List<DataRequest> asked = new ArrayList<>();
    final List<Long> lookupKeys = new ArrayList<>();

    Recording(TrinoSource source) {
      this.source = source;
    }

    @Override
    public Capabilities capabilities() {
      return source.capabilities();
    }

    @Override
    public ArrowReader answer(
        DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
      asked.add(request);
      if (request instanceof DataRequest.Lookup lookup) {
        // A key set is read once: read it here, and hand on its rows.
        byte[] seeds;
        long rows = 0;
        var bytes = new java.io.ByteArrayOutputStream();
        try (ArrowReader reader = lookup.seeds().reader();
            var writer =
                new org.apache.arrow.vector.ipc.ArrowStreamWriter(
                    reader.getVectorSchemaRoot(), null, bytes)) {
          writer.start();
          while (reader.loadNextBatch()) {
            rows += reader.getVectorSchemaRoot().getRowCount();
            writer.writeBatch();
          }
          writer.end();
        }
        seeds = bytes.toByteArray();
        lookupKeys.add(rows);
        request =
            new DataRequest.Lookup(
                lookup.table(),
                lookup.key(),
                lookup.columns(),
                new DataRequest.KeySet(
                    lookup.seeds().schema(),
                    new org.apache.arrow.vector.ipc.ArrowStreamReader(
                        new java.io.ByteArrayInputStream(seeds), allocator)));
      }
      return source.answer(request, expected, allocator);
    }

    @Override
    public ProviderFailure failure(Exception failure) {
      return source.failure(failure);
    }
  }

  /** The schema over {@code source}: Trino declares no keys, so the caller names them. */
  private static com.nvidia.sdm.Schema shopSchema(Source source) {
    return com.nvidia.sdm.Schema.fromSource(
            source,
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
  }

  @SuppressWarnings("unchecked")
  private static Request materialize(Source source, Map<String, Object> scenario) {
    var options = MaterializeOptions.builder();
    ((Map<String, Object>) scenario.get("options"))
        .forEach(
            (key, value) -> {
              switch (key) {
                case "indices" -> options.indices((List<Object>) value);
                case "random_seed" -> options.randomSeed(((Number) value).longValue());
                case "num_context_samples" -> options.numContextSamples(((Number) value).intValue());
                case "candidates" -> options.candidates((List<Object>) value);
                case "anchor_time" -> options.anchorTime(LocalDate.parse((String) value).atStartOfDay());
                default -> throw new IllegalArgumentException("unknown option " + key);
              }
            });
    return PredictiveQuery.parse((String) scenario.get("pql"))
        .materialize(shopSchema(source), options.build());
  }

  @SuppressWarnings("unchecked")
  private static List<Map<String, Object>> scenarios() throws Exception {
    Map<String, Object> manifest =
        (Map<String, Object>)
            Json.read(Files.readAllBytes(Path.of("..", "tests", "compatibility", "manifest.json")));
    return ((List<Object>) manifest.get("scenarios"))
        .stream()
            .map(s -> (Map<String, Object>) s)
            .filter(s -> ((String) s.get("id")).startsWith("context."))
            .toList();
  }

  private void run(String sql) throws SQLException {
    try (Statement statement = trino.createStatement()) {
      statement.execute(sql);
    }
  }

  private String qualified(String table) {
    return CATALOG + "." + schemaName + ".\"" + table.replace("\"", "\"\"") + "\"";
  }

  private static String fixtureLiteral(Object value) {
    if (value == null) return "NULL";
    if (value instanceof Boolean b) return b ? "TRUE" : "FALSE";
    if (value instanceof Double d) return "DOUBLE '" + d + "'";
    if (value instanceof Number n) return n.toString();
    if (value instanceof String text && text.matches("\\d{4}-\\d\\d-\\d\\dT.*"))
      return "TIMESTAMP '"
          + LocalDateTime.parse(text).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSSSSS"))
          + "'";
    return "'" + value.toString().replace("'", "''") + "'";
  }
}
