// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import com.nvidia.sdm.*;
import java.sql.*;
import java.util.*;
import java.util.function.Supplier;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Table;

/** Public API timings only; no internal Trace listener or attachment capture. */
public final class ClientCost {
  static double elapsed(long start) { return (System.nanoTime() - start) / 1e9; }
  static <T> T timed(Map<String, Double> times, String name, Supplier<T> call) {
    long start = System.nanoTime(); T value = call.get(); times.put(name, elapsed(start)); return value;
  }
  static Table users(RootAllocator allocator, int rows, int width, int offset, boolean target) {
    List<FieldVector> vectors = new ArrayList<>();
    BigIntVector id = new BigIntVector("id", allocator); id.allocateNew(rows);
    for (int row = 0; row < rows; row++) id.setSafe(row, row + offset);
    id.setValueCount(rows); vectors.add(id);
    for (int column = 0; column < width; column++) {
      Float8Vector x = new Float8Vector("x" + column, allocator); x.allocateNew(rows);
      for (int row = 0; row < rows; row++) x.setSafe(row, (double)((row + offset + column) % 97));
      x.setValueCount(rows); vectors.add(x);
    }
    if (target) {
      Float8Vector y = new Float8Vector("y", allocator); y.allocateNew(rows);
      for (int row = 0; row < rows; row++) y.setSafe(row, (double)((row + offset) % 31));
      y.setValueCount(rows); vectors.add(y);
    }
    return new Table(vectors);
  }
  static Table events(RootAllocator allocator, int rows, int degree) {
    int count = Math.multiplyExact(rows, degree);
    BigIntVector key = new BigIntVector("event_id", allocator);
    BigIntVector user = new BigIntVector("id", allocator);
    Float8Vector value = new Float8Vector("value", allocator);
    key.allocateNew(count); user.allocateNew(count); value.allocateNew(count);
    for (int row = 0; row < count; row++) {
      key.setSafe(row, row); user.setSafe(row, row / degree); value.setSafe(row, (double)(row % 13));
    }
    key.setValueCount(count); user.setValueCount(count); value.setValueCount(count);
    return new Table(List.of(key, user, value));
  }
  static void populate(Connection connection, int rows, int width, int degree) throws SQLException {
    StringBuilder sql = new StringBuilder("CREATE TABLE users (id INTEGER");
    for (int column = 0; column < width; column++) sql.append(",x").append(column).append(" REAL");
    sql.append(",y REAL)");
    try (Statement statement = connection.createStatement()) {
      statement.execute(sql.toString());
      statement.execute("CREATE TABLE events (event_id INTEGER,id INTEGER,value REAL)");
    }
    String placeholders = String.join(",", Collections.nCopies(width + 2, "?"));
    try (PreparedStatement statement = connection.prepareStatement("INSERT INTO users VALUES (" + placeholders + ")")) {
      for (int row = 0; row < rows; row++) {
        statement.setLong(1, row);
        for (int column = 0; column < width; column++) statement.setDouble(column + 2, (row + column) % 97);
        statement.setDouble(width + 2, row % 31); statement.addBatch();
      }
      statement.executeBatch();
    }
    try (PreparedStatement statement = connection.prepareStatement("INSERT INTO events VALUES (?,?,?)")) {
      for (int row = 0; row < rows * degree; row++) {
        statement.setLong(1, row); statement.setLong(2, row / degree); statement.setDouble(3, row % 13); statement.addBatch();
      }
      statement.executeBatch();
    }
  }
  static int view(PredictionResult result, RootAllocator allocator, boolean fixed) {
    int rows = 0;
    try (ArrowReader reader = result.read(allocator)) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        for (int row = 0; row < root.getRowCount(); row++) {
          if (fixed && ((Number)root.getVector("prediction").getObject(row)).doubleValue() != 1.0)
            throw new AssertionError("prediction contract differs");
        }
        rows += root.getRowCount();
      }
    } catch (java.io.IOException error) { throw new RuntimeException(error); }
    return rows;
  }
  static String json(Object value) {
    if (value == null) return "null";
    if (value instanceof Number || value instanceof Boolean) return value.toString();
    if (value instanceof Map<?, ?> map) {
      List<String> parts = new ArrayList<>();
      map.forEach((key, item) -> parts.add(json(key.toString()) + ":" + json(item)));
      return "{" + String.join(",", parts) + "}";
    }
    if (value instanceof Collection<?> list) return "[" + String.join(",", list.stream().map(ClientCost::json).toList()) + "]";
    return "\"" + value.toString().replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
  }
  public static void main(String[] argv) throws Exception {
    double entryEpoch = System.currentTimeMillis() / 1000.0;
    Map<String, String> args = new HashMap<>();
    for (int at = 0; at < argv.length; at += 2) args.put(argv[at].substring(2), argv[at + 1]);
    int rows = Integer.parseInt(args.get("rows")), scores = Integer.parseInt(args.get("scores"));
    int width = Integer.parseInt(args.get("width")), degree = Integer.parseInt(args.get("degree"));
    int warmups = Integer.parseInt(args.get("warmups")), loops = Integer.parseInt(args.get("loops"));
    boolean gradients = args.get("gradients").equals("1"), fixed = System.getenv("SDM_BENCH_FIXED").equals("1");
    boolean tabular = args.get("workload").equals("tabular-session");
    boolean sqlite = args.get("workload").equals("relational-sqlite");
    String model = tabular ? SdmClient.KUMO_TABULAR : SdmClient.KUMO_RELATIONAL;
    List<String> outputs = gradients ? List.of("prediction", "explanation") : List.of("prediction");
    List<Map<String, Object>> iterations = new ArrayList<>();
    long setup = System.nanoTime();
    try (RootAllocator allocator = new RootAllocator(); Table users = users(allocator, rows, width, 0, true);
         Table scoring = users(allocator, scores, width, rows - scores, false); Table events = events(allocator, rows, degree);
         Connection connection = sqlite ? DriverManager.getConnection("jdbc:sqlite::memory:") : null;
         SdmClient client = SdmClient.connect(System.getenv("SDM_BENCH_URL"))) {
      if (sqlite) populate(connection, rows, width, degree);
      double setupSeconds = elapsed(setup);
      for (int iteration = 0; iteration < 1 + warmups + loops; iteration++) {
        Map<String, Double> times = new LinkedHashMap<>();
        PredictionResult result;
        if (tabular) {
          Request request = timed(times, "request_prepare", () -> new Request(users, scoring, "y", "regression"));
          FittedModel fitted = timed(times, "fit_http", () -> client.fit(request, model, List.of(), outputs));
          try { result = timed(times, "score_http_decode", () -> fitted.predict(request)); }
          finally { timed(times, "session_close_http", () -> { fitted.close(); return null; }); }
        } else {
          Schema schema = timed(times, "schema", () -> (sqlite ? Schema.fromSource(SqliteSource.of(connection)) : Schema.fromData(Map.of("users", users, "events", events)))
              .withPrimaryKey("users", "id").withPrimaryKey("events", "event_id").withStype("users", "y", "numerical")
              .withRelationships(List.of(new Relationship("events", "id", "users", "id"))));
          PredictiveQuery query = timed(times, "pql", () -> PredictiveQuery.parse("PREDICT users.y FOR EACH users.id"));
          List<Long> indices = new ArrayList<>();
          for (long id = rows - scores; id < rows; id++) indices.add(id);
          Request request = timed(times, "materialize_sample_encode", () -> query.materialize(schema,
              MaterializeOptions.builder().model(model).indices(indices).numContextSamples(Math.min(rows - scores, 32))
              .randomSeed(7).option("num_neighbors", List.of(degree)).option("num_hops", 1).build()));
          result = timed(times, "predict_http_decode", () -> client.predict(request, model, outputs));
        }
        if (timed(times, "result_view", () -> view(result, allocator, fixed)) != scores) throw new AssertionError("row count differs");
        if (gradients) {
          timed(times, "gradient_view", () -> {
            try (VectorSchemaRoot table = result.explanations().toArrow(allocator)) {
              if (fixed) for (int row = 0; row < table.getRowCount(); row++) {
                if (((Number)table.getVector("gradient").getObject(row)).doubleValue() != 0.25) throw new AssertionError("gradient differs");
              }
              return table.getRowCount();
            }
          });
          if (result.explanations().entries().size() != scores) throw new AssertionError("explanation count differs");
        }
        PredictionResult merged = timed(times, "result_merge", () -> PredictionResult.merge(List.of(result, result)));
        if (merged.rows() != 2 * scores) throw new AssertionError("merge row count differs");
        if (gradients && merged.explanations().entries().size() != 2 * scores) throw new AssertionError("merge lost explanations");
        iterations.add(Map.of("phase", iteration == 0 ? "cold-first" : iteration <= warmups ? "warmup" : "warm", "seconds", times));
      }
      Map<String, Object> result = new LinkedHashMap<>();
      result.put("entry_epoch_seconds", entryEpoch); result.put("status", "passed"); result.put("inputs", args); result.put("iterations", iterations);
      result.put("setup_seconds", setupSeconds); result.put("native_load", "included in setup or first public operation; not isolated");
      result.put("sql_query_count", sqlite ? null : 0); result.put("native_call_count", null);
      result.put("process_highwater_rss_bytes", null); result.put("gc", "JVM default, no forced collection");
      result.put("runtime_jar", SdmClient.class.getProtectionDomain().getCodeSource().getLocation().toString());
      result.put("trace", "disabled"); System.out.println(json(result));
    }
  }
}
