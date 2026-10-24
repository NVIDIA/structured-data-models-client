// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import com.nvidia.sdm.DuckDbSource;
import com.nvidia.sdm.FromSourceOptions;
import com.nvidia.sdm.MaterializeOptions;
import com.nvidia.sdm.ModelEndpointException;
import com.nvidia.sdm.PostgresSource;
import com.nvidia.sdm.PredictionResult;
import com.nvidia.sdm.PredictiveQuery;
import com.nvidia.sdm.Relationship;
import com.nvidia.sdm.Request;
import com.nvidia.sdm.Schema;
import com.nvidia.sdm.SdmClient;
import com.nvidia.sdm.Source;
import com.nvidia.sdm.SqliteSource;
import com.nvidia.sdm.StructuredDataException;
import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Contract;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.NativeCore;
import com.nvidia.sdm.internal.Profile;
import com.nvidia.sdm.internal.QuerySchema;
import com.nvidia.sdm.internal.QueryValidation;
import com.nvidia.sdm.internal.Trace;
import com.nvidia.sdm.internal.Validation;
import com.nvidia.sdm.internal.jdbc.ConversionTable;
import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import org.apache.arrow.vector.ipc.ArrowReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The Java runner program of the contract cases (design choice 10, "Tests"), run by
 * contract/runners/run_java.py in source-file mode against the JAR under test, with the JDBC
 * drivers beside it:
 *
 * <pre>java FLAGS -cp JAR:DEPENDENCIES:DRIVERS ContractRunner.java</pre>
 *
 * It reads one call a line on stdin, each one case's, holding only what to call and with what (no
 * case file is read here), makes it through the public API, and writes one JSON line on stdout:
 * the case, the outcome (a success, or the exception's code), and what the call did as the
 * listener ({@link Trace}) was told it: each core run's start with its attachments, each step,
 * each attachment taken, each result or error and the warnings, and each statement the provider
 * ran with its result (an Arrow result as its IPC stream, in base64), by the public call it ran
 * in. run_java.py turns them into observations as the Python runner's capture turns the Python
 * host's into its own. A runner failure (anything but an answer the API gave) exits non-zero.
 *
 * <p>The calls: S0's {"op": "validate_query", "case", "query", "schema", "profile"}, made through
 * the core's validation (internal: the public API has none, as Python's has none); S1's {"op":
 * "case", "case", "database", "connect", "setup", "until", "pql", "from_source", "edits",
 * "materialize", "outputs", "endpoint"}, through Schema.fromSource, the schema's edits,
 * PredictiveQuery.materialize and SdmClient.predict, on the database run_java.py loaded.
 */
public final class ContractRunner {
  private ContractRunner() {}

  public static void main(String[] arguments) throws Exception {
    // Every line this writes is an answer: anything else the JVM's libraries print goes to stderr.
    PrintStream answers = new PrintStream(System.out, true, StandardCharsets.UTF_8);
    System.setOut(System.err);
    BufferedReader calls =
        new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
    String line;
    while ((line = calls.readLine()) != null) {
      if (line.isBlank()) {
        continue;
      }
      Map<?, ?> call = (Map<?, ?>) Json.read(line.getBytes(StandardCharsets.UTF_8));
      Object answer =
          switch (String.valueOf(call.get("op"))) {
            case "validate_query" -> validation(call);
            case "fetch" -> fetches(call);
            default -> s1(call);
          };
      answers.println(new String(Json.write(answer), StandardCharsets.UTF_8));
    }
  }

  /** What the library and the JVM are, for the observation's identity. */
  static Map<String, Object> identity() {
    Map<String, Object> identity = new LinkedHashMap<>();
    identity.put("platform", NativeCore.platform());
    identity.put("contract", Host.contract());
    identity.put("java", System.getProperty("java.version"));
    identity.put("test_ops", NativeCore.load().get("test_ops"));
    identity.put("providers", Host.PROVIDERS);
    return identity;
  }

  // --- S0 -----------------------------------------------------------------------------------

  private static Map<String, Object> validation(Map<?, ?> call) {
    String query = (String) call.get("query");
    QuerySchema schema = schema((Map<?, ?>) call.get("schema"));
    Profile profile = Profile.valueOf((String) ((Map<?, ?>) call.get("profile")).get("name"));
    Map<String, Object> input = new LinkedHashMap<>();
    input.put("op", "validate_query");
    input.put("query", query);
    input.put("profile", Contract.profile(profile));
    input.put("schema", Contract.querySchema(schema));
    Map<String, Object> answer = new LinkedHashMap<>();
    answer.put("case", call.get("case"));
    answer.put("operations", List.of("validate_query"));
    answer.put("inputs", List.of(input));
    answer.put("identity", identity());
    try {
      Validation validation = QueryValidation.validate(query, schema, profile);
      answer.put("accepted", validation.accepted());
      answer.put("error", null);
    } catch (StructuredDataException raised) {
      Map<String, Object> error = new LinkedHashMap<>();
      error.put("kind", raised.kind());
      error.put("code", raised.code());
      answer.put("accepted", null);
      answer.put("error", error);
    }
    return answer;
  }

  private static String text(Map<?, ?> map, String key) {
    Object value = map.get(key);
    return value == null ? null : (String) value;
  }

  /** The case's QuerySchema JSON as the internal plain data. */
  static QuerySchema schema(Map<?, ?> json) {
    List<QuerySchema.Table> tables = new ArrayList<>();
    for (Object item : (List<?>) json.get("tables")) {
      Map<?, ?> table = (Map<?, ?>) item;
      List<QuerySchema.Column> columns = new ArrayList<>();
      for (Object c : (List<?>) table.get("columns")) {
        Map<?, ?> column = (Map<?, ?>) c;
        columns.add(
            new QuerySchema.Column(
                text(column, "name"), text(column, "dtype"), text(column, "stype")));
      }
      tables.add(
          new QuerySchema.Table(
              text(table, "name"),
              columns,
              text(table, "primary_key"),
              text(table, "time_column"),
              text(table, "end_time_column")));
    }
    List<List<QuerySchema.ColumnRef>> links = new ArrayList<>();
    for (Object g : (List<?>) json.get("links")) {
      List<QuerySchema.ColumnRef> group = new ArrayList<>();
      for (Object r : (List<?>) g) {
        Map<?, ?> ref = (Map<?, ?>) r;
        group.add(new QuerySchema.ColumnRef(text(ref, "table"), text(ref, "column")));
      }
      links.add(group);
    }
    return new QuerySchema(tables, links);
  }

  // --- L5: fetches answered by the provider alone ---------------------------------------------

  /**
   * L5 for Java: each fetch step the call names (the core's own plans, as the Python test
   * recorded them), answered by the JDBC provider on the database, read by the reviewed
   * conversion table or, with a {@code seed}, by the table with one Postgres type read as another
   * (a unit or scale change in one driver's path), any key sets ({@code keys}, by attachment)
   * handed over: each answer as an Arrow IPC stream in base64, or the provider's failure.
   */
  @SuppressWarnings("unchecked")
  private static Map<String, Object> fetches(Map<?, ?> call) throws Exception {
    String database = (String) call.get("database");
    ConversionTable table = ConversionTable.REVIEWED;
    if (call.get("seed") instanceof Map<?, ?> seed) {
      String read = (String) seed.get("read");
      String as = (String) seed.get("as");
      table = table.withPostgres(read, (p, s) -> ConversionTable.REVIEWED.postgres(as, p, s));
    }
    List<Object> answers = new ArrayList<>();
    try (Connection connection = connect(database, (Map<?, ?>) call.get("connect"))) {
      if ("duckdb".equals(database)) {
        try (Statement statement = connection.createStatement()) {
          for (Object setup : (List<?>) call.get("setup")) {
            statement.execute((String) setup);
          }
        }
      }
      JdbcProvider provider =
          new JdbcProvider(connection, JdbcProvider.Driver.valueOf(database.toUpperCase(java.util.Locale.ROOT)), table);
      for (Object step : (List<?>) call.get("steps")) {
        Map<String, Object> answer = new LinkedHashMap<>();
        try {
          // The key sets handed over, each an Arrow IPC stream in base64, read afresh per step.
          Map<Integer, ArrowReader> keys = new LinkedHashMap<>();
          if (call.get("keys") instanceof Map<?, ?> given) {
            for (Map.Entry<?, ?> each : given.entrySet()) {
              keys.put(
                  Integer.valueOf(String.valueOf(each.getKey())),
                  ArrowData.reader(Base64.getDecoder().decode((String) each.getValue()), ArrowData.allocator()));
            }
          }
          ArrowReader read = provider.answer((Map<String, Object>) step, keys, ArrowData.allocator());
          answer.put("arrow", Base64.getEncoder().encodeToString(ArrowData.ipc(read)));
        } catch (Exception failed) {
          answer.put("error", failed.getClass().getName());
          // The SQLSTATE a driver's refusal names, as Python's provider's carries it.
          answer.put(
              "sqlstate", failed instanceof java.sql.SQLException sql ? sql.getSQLState() : null);
        }
        answers.add(answer);
      }
    }
    Map<String, Object> answer = new LinkedHashMap<>();
    answer.put("case", call.get("case"));
    answer.put("answers", answers);
    return answer;
  }

  // --- S1 -----------------------------------------------------------------------------------

  /** What the listener was told during one case, as JSON values, in order. */
  static final class Recorder implements Trace {
    final List<Object> events = new ArrayList<>();
    String stage = "from_source";
    int cursors;

    private void add(String event, Map<String, Object> fields) {
      Map<String, Object> each = new LinkedHashMap<>();
      each.put("event", event);
      each.put("stage", stage);
      each.putAll(fields);
      events.add(each);
    }

    private static String base64(byte[] bytes) {
      return Base64.getEncoder().encodeToString(bytes);
    }

    @Override
    public void started(byte[] message, Map<Integer, Object> attachments) {
      Map<String, Object> told = new LinkedHashMap<>();
      for (Map.Entry<Integer, Object> each : attachments.entrySet()) {
        Map<String, Object> value = new LinkedHashMap<>();
        if (each.getValue() instanceof Host.ArrowIpc arrow) {
          value.put("arrow", base64(arrow.bytes()));
        } else {
          value.put("bytes", base64((byte[]) each.getValue()));
        }
        told.put(String.valueOf(each.getKey()), value);
      }
      Map<String, Object> fields = new LinkedHashMap<>();
      fields.put("message", base64(message));
      fields.put("attachments", told);
      add("started", fields);
    }

    @Override
    public void stepped(Map<String, Object> step) {
      add("stepped", Map.of("step", step));
    }

    @Override
    public void taken(int id, Object value, boolean arrow) {
      Map<String, Object> fields = new LinkedHashMap<>();
      fields.put("id", id);
      fields.put(arrow ? "arrow" : "bytes", base64((byte[]) value));
      add("taken", fields);
    }

    @Override
    public void done(Map<String, Object> result) {
      add("done", Map.of("result", result));
    }

    @Override
    public void failed(String kind, String code) {
      Map<String, Object> fields = new LinkedHashMap<>();
      fields.put("kind", kind);
      fields.put("code", code);
      add("failed", fields);
    }

    @Override
    public void warned(List<Map<String, Object>> warnings) {
      add("warned", Map.of("warnings", warnings));
    }

    @Override
    public void cursor(int handle) {
      cursors++;
      add("cursor", Map.of("handle", cursors));
    }

    @Override
    public void statement(int handle, String sql) {
      Map<String, Object> fields = new LinkedHashMap<>();
      fields.put("handle", cursors);
      fields.put("sql", sql);
      add("statement", fields);
    }

    @Override
    public void answered(byte[] ipc, List<Map<String, Object>> columns) {
      Map<String, Object> fields = new LinkedHashMap<>();
      fields.put("arrow", base64(ipc));
      fields.put("columns", columns);
      add("answered", fields);
    }

    @Override
    public void statementFailed(String error) {
      add("statement_failed", Map.of("error", error));
    }

    @Override
    public void bound(int handle, String name, String sql, byte[] ipc) {
      Map<String, Object> fields = new LinkedHashMap<>();
      fields.put("handle", cursors);
      fields.put("name", name);
      fields.put("sql", sql);
      fields.put("arrow", base64(ipc));
      add("bound", fields);
    }
  }

  private static Map<String, Object> s1(Map<?, ?> call) throws Exception {
    String database = (String) call.get("database");
    Map<String, Object> answer = new LinkedHashMap<>();
    answer.put("case", call.get("case"));
    answer.put("identity", identity());
    Recorder recorder = new Recorder();
    try (Connection connection = connect(database, (Map<?, ?>) call.get("connect"))) {
      if ("duckdb".equals(database)) {
        try (Statement statement = connection.createStatement()) {
          for (Object setup : (List<?>) call.get("setup")) {
            statement.execute((String) setup);
          }
        }
        answer.put("duckdb", version(connection));
        try (Statement statement = connection.createStatement()) {
          statement.execute("CALL enable_logging('QueryLog')");
        }
      }
      Trace.install(recorder);
      Map<String, Object> outcome;
      try {
        outcome = calls(call, source(database, connection), recorder);
      } finally {
        Trace.install(null);
      }
      answer.put("outcome", outcome);
      if ("duckdb".equals(database)) {
        answer.put("logged", queryLog(connection));
      }
    }
    answer.put("events", recorder.events);
    return answer;
  }

  /** The case's public calls: the outcome, a success or the exception's code, at its stage. */
  private static Map<String, Object> calls(Map<?, ?> call, Source source, Recorder recorder) {
    Map<String, Object> outcome = new LinkedHashMap<>();
    String until = (String) call.get("until");
    try {
      Schema schema = Schema.fromSource(source, fromSource((Map<?, ?>) call.get("from_source")));
      for (Object edit : (List<?>) call.get("edits")) {
        schema = edited(schema, (Map<?, ?>) edit);
      }
      if (!"from_source".equals(until)) {
        recorder.stage = "materialize";
        if (call.get("validate") instanceof Map<?, ?> validate) {
          // The case's own validation, which the Python runner's forced route asks first.
          @SuppressWarnings("unchecked")
          Map<String, Object> written = (Map<String, Object>) validate.get("schema");
          Profile profile =
              Profile.valueOf((String) ((Map<?, ?>) validate.get("profile")).get("name"));
          if (!QueryValidation.validate((String) call.get("pql"), written, profile).accepted()) {
            outcome.put("result", "refusal");
            outcome.put("code", "INVALID_REQUEST");
            outcome.put("stage", recorder.stage);
            return outcome;
          }
        }
        Request request =
            PredictiveQuery.parse((String) call.get("pql"))
                .materialize(schema, materialize((Map<?, ?>) call.get("materialize")));
        if ("predict".equals(until)) {
          recorder.stage = "predict";
          try (SdmClient client = SdmClient.connect((String) call.get("endpoint"))) {
            List<String> outputs = null;
            if (call.get("outputs") instanceof List<?> named) {
              outputs = new ArrayList<>();
              for (Object name : named) {
                outputs.add((String) name);
              }
            }
            PredictionResult result = client.predict(request, SdmClient.KUMO_RELATIONAL, outputs);
            outcome.put("rows", result.rows());
          }
        }
      }
      outcome.put("result", "success");
      outcome.put("code", null);
    } catch (ModelEndpointException refused) {
      // The endpoint's refusal, which the core ends the run with as INVALID_RESPONSE (decision
      // E45): the Java outcome is the core's answer, as rust-host's is.
      outcome.put("result", "refusal");
      outcome.put("code", "INVALID_RESPONSE");
      outcome.put("class", refused.getClass().getSimpleName());
    } catch (StructuredDataException raised) {
      outcome.put("result", "refusal");
      outcome.put("code", raised.code());
      outcome.put("class", raised.getClass().getSimpleName());
    }
    outcome.put("stage", recorder.stage);
    return outcome;
  }

  private static Connection connect(String database, Map<?, ?> how) throws SQLException {
    return switch (database) {
      case "duckdb" -> DriverManager.getConnection("jdbc:duckdb:");
      case "sqlite" -> DriverManager.getConnection("jdbc:sqlite:" + how.get("path"));
      case "postgres" ->
          DriverManager.getConnection(
              (String) how.get("url"), (String) how.get("user"), (String) how.get("password"));
      default -> throw new IllegalArgumentException("a database this runner does not open");
    };
  }

  private static Source source(String database, Connection connection) {
    return switch (database) {
      case "duckdb" -> DuckDbSource.of(connection);
      case "sqlite" -> SqliteSource.of(connection);
      default -> PostgresSource.of(connection);
    };
  }

  private static String version(Connection connection) throws SQLException {
    try (Statement statement = connection.createStatement();
        ResultSet rows = statement.executeQuery("SELECT version()")) {
      rows.next();
      String version = rows.getString(1);
      return version.startsWith("v") ? version.substring(1) : version;
    }
  }

  /** Each query DuckDB logged since the case's setup, in order: '' for a relation registered. */
  private static List<String> queryLog(Connection connection) throws SQLException {
    List<String> logged = new ArrayList<>();
    try (Statement statement = connection.createStatement();
        ResultSet rows =
            statement.executeQuery(
                "SELECT message FROM duckdb_logs WHERE type = 'QueryLog' ORDER BY context_id")) {
      while (rows.next()) {
        logged.add(rows.getString(1));
      }
    }
    return logged;
  }

  @SuppressWarnings("unchecked")
  private static FromSourceOptions fromSource(Map<?, ?> given) {
    FromSourceOptions.Builder options = FromSourceOptions.builder();
    if (given == null) {
      return options.build();
    }
    if (given.get("tables") instanceof Map<?, ?> tables) {
      tables.forEach((name, source) -> options.table((String) name, (String) source));
    }
    if (given.get("primary_keys") instanceof Map<?, ?> keys) {
      keys.forEach(
          (table, key) -> {
            if (key == null) {
              options.noPrimaryKey((String) table);
            } else if (key instanceof String one) {
              options.primaryKey((String) table, one);
            } else {
              options.primaryKey((String) table, ((List<String>) key).toArray(new String[0]));
            }
          });
    }
    if (given.get("time_columns") instanceof Map<?, ?> times) {
      times.forEach(
          (table, column) -> {
            if (column == null) {
              options.noTimeColumn((String) table);
            } else {
              options.timeColumn((String) table, (String) column);
            }
          });
    }
    if (given.get("relationships") instanceof List<?> links) {
      options.relationships(relationships(links));
    }
    if (given.get("check_keys") instanceof Boolean check) {
      options.checkKeys(check);
    }
    return options.build();
  }

  private static List<Relationship> relationships(List<?> links) {
    List<Relationship> made = new ArrayList<>();
    for (Object item : links) {
      Map<?, ?> link = (Map<?, ?>) item;
      made.add(
          new Relationship(
              (String) link.get("source_table"),
              (String) link.get("source_column"),
              (String) link.get("target_table"),
              (String) link.get("target_column")));
    }
    return made;
  }

  /** One of the case's edits, as the schema's edit of it. */
  private static Schema edited(Schema schema, Map<?, ?> edit) {
    String table = (String) edit.get("table");
    return switch ((String) edit.get("op")) {
      case "clear_time_column" -> schema.withoutTimeColumn(table);
      case "delete_column" -> schema.withoutColumn(table, (String) edit.get("column"));
      case "set_primary_key" -> schema.withPrimaryKey(table, (String) edit.get("column"));
      case "set_time_column" -> schema.withTimeColumn(table, (String) edit.get("column"));
      case "set_stype" -> schema.withStype(table, (String) edit.get("column"), (String) edit.get("stype"));
      default -> schema.withRelationships(relationships((List<?>) edit.get("relationships")));
    };
  }

  /** The case's materialize arguments, each as the public call takes it. */
  private static MaterializeOptions materialize(Map<?, ?> given) {
    MaterializeOptions.Builder options = MaterializeOptions.builder();
    List<Object> indices = new ArrayList<>();
    boolean decimal = "decimal".equals(given.get("indices_as"));
    for (Object value : (List<?>) given.get("indices")) {
      indices.add(decimal ? new BigDecimal((String) value) : value);
    }
    options.indices(indices);
    if (given.containsKey("anchor_time")) {
      Object anchor = given.get("anchor_time");
      if ("entity".equals(anchor)) {
        options.anchorAtEachEntity();
      } else {
        options.anchorTime(time((String) anchor));
      }
    }
    if (given.containsKey("context_anchor_time")) {
      Object anchor = given.get("context_anchor_time");
      if ("entity".equals(anchor)) {
        options.contextAnchorAtEachEntity();
      } else {
        options.contextAnchorTime(time((String) anchor));
      }
    }
    if (given.get("num_context_samples") instanceof Number samples) {
      options.numContextSamples(samples.intValue());
    }
    for (Map.Entry<?, ?> each : given.entrySet()) {
      String name = (String) each.getKey();
      if (!List.of("indices", "indices_as", "anchor_time", "context_anchor_time",
              "num_context_samples")
          .contains(name)) {
        options.option(name, each.getValue());
      }
    }
    return options.build();
  }

  /** A case's time, a date or a date and time, as pandas reads its text. */
  private static LocalDateTime time(String text) {
    if (text.length() == 10) {
      return LocalDate.parse(text).atStartOfDay();
    }
    return LocalDateTime.parse(text.replace(' ', 'T'));
  }

  static String read(byte[] bytes) throws IOException {
    return new String(bytes, StandardCharsets.UTF_8);
  }
}
