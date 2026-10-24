// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Trace;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Savepoint;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * The Java host's one data provider: it answers a fetch step by running the statements its plan
 * carries, in order, on the source's {@link Connection}, and hands back the last one's result as
 * Arrow, as Python's providers do ({@code sdm_client._rust.providers}). The statements are the
 * core's ({@code sdm-data} renders each dialect's SQL); this class writes none, reads no value's
 * meaning, and decides nothing (structure design rule 6). What differs by driver is two things
 * only:
 *
 * <ul>
 *   <li>how a key set the plan names is bound ({@link KeySets}): DuckDB registers it as Arrow and
 *       keeps it as a temporary table, SQLite ingests it as a temporary table as ADBC does,
 *       Postgres binds the JSON text the core wrote of it (T3.28), handed over as one text cell,
 *       as the statement's one parameter, as it was handed;
 *   <li>how a result becomes Arrow: the driver's column of the conversion table ({@link
 *       ConversionTable}).
 * </ul>
 *
 * A plan this provider cannot run (one for a host provider, one of another dialect, one with a
 * scalar parameter, one naming a key set it was not handed, two key sets under one name, a key
 * set's Arrow on Postgres or a text elsewhere) is
 * refused by an exception whose text names no value, and the run loop hands the core a provider
 * error. Not part of the API.
 */
public final class JdbcProvider implements Host.Provider {
  /** The databases this provider reads, by the dialect the core names each by. */
  public enum Driver {
    DUCKDB("duckdb"),
    SQLITE("sqlite"),
    POSTGRES("postgres");

    private final String dialect;

    Driver(String dialect) {
      this.dialect = dialect;
    }

    /** The dialect the core names its statements by. */
    public String dialect() {
      return dialect;
    }
  }

  /** Rows per Arrow batch of a result built from rows; a larger result comes as several. */
  static final int BATCH_ROWS = 100_000;

  /** Rows per batch DuckDB's driver exports, as the Python provider asks of DuckDB's. */
  static final int DUCKDB_BATCH_ROWS = 1_000_000;

  private final Connection connection;
  private final Driver driver;
  private final ConversionTable table;
  private final KeySets keySets;

  /** A provider over {@code connection}, read by the reviewed conversion table. */
  public JdbcProvider(Connection connection, Driver driver) {
    this(connection, driver, ConversionTable.REVIEWED);
  }

  /** A provider over {@code connection}, read by {@code table} (L5's seeded change only). */
  public JdbcProvider(Connection connection, Driver driver, ConversionTable table) {
    this.connection = connection;
    this.driver = driver;
    this.table = table;
    this.keySets = KeySets.of(driver);
  }

  /** A plan this provider does not run; its text names no value. */
  public static final class PlanNotRun extends Exception {
    private static final long serialVersionUID = 1L;

    PlanNotRun(String message) {
      super(message);
    }
  }

  /**
   * One statement of a plan: its text, the key set it binds by relation, if any, and the text it
   * binds as its one parameter, by attachment, if any.
   */
  record Planned(String text, Map<String, Integer> relations, Integer keyText) {}

  @Override
  @SuppressWarnings("unchecked")
  public ArrowReader answer(
      Map<String, Object> request, Map<Integer, ArrowReader> keys, BufferAllocator allocator)
      throws Exception {
    Map<String, Object> plan = (Map<String, Object>) request.get("plan");
    if (plan == null || !"sql".equals(plan.get("plan"))) {
      throw new PlanNotRun("a host plan is not a SQL provider's to run");
    }
    List<Object> statements = (List<Object>) plan.get("statements");
    if (statements == null || statements.isEmpty()) {
      throw new PlanNotRun("a plan with no statement or another dialect");
    }
    List<Planned> planned = new ArrayList<>();
    Map<String, Integer> relations = new LinkedHashMap<>();
    for (Object item : statements) {
      Map<String, Object> statement = (Map<String, Object>) item;
      if (!driver.dialect().equals(statement.get("dialect"))) {
        throw new PlanNotRun("a plan with no statement or another dialect");
      }
      Map<String, Integer> bound = new LinkedHashMap<>();
      Integer text = null;
      for (Object p : (List<Object>) statement.get("params")) {
        Map<String, Object> param = (Map<String, Object>) p;
        Object kind = param.get("param");
        if ("text".equals(kind)) {
          // The key set's JSON text the core wrote (T3.28), which Postgres binds.
          if (driver != Driver.POSTGRES) {
            throw new PlanNotRun("a key set's text on a driver binding key sets as relations");
          }
          int attachment = ((Number) param.get("attachment")).intValue();
          if (!keys.containsKey(attachment)) {
            throw new PlanNotRun("a plan naming a key set not handed over");
          }
          if (text != null) {
            throw new PlanNotRun("a statement binding more than one key set");
          }
          text = attachment;
          continue;
        }
        if (!"key_set".equals(kind)) {
          throw new PlanNotRun("a plan with a scalar parameter");
        }
        if (driver == Driver.POSTGRES) {
          throw new PlanNotRun("a key set Postgres binds only as the text the core wrote of it");
        }
        int attachment =
            ((Number) ((Map<String, Object>) param.get("keys")).get("attachment")).intValue();
        if (!keys.containsKey(attachment)) {
          throw new PlanNotRun("a plan naming a key set not handed over");
        }
        String relation = String.valueOf(param.get("relation"));
        Integer earlier = relations.putIfAbsent(relation, attachment);
        if (earlier != null && earlier != attachment) {
          throw new PlanNotRun("a plan binding two key sets to one name");
        }
        bound.put(relation, attachment);
      }
      planned.add(new Planned(String.valueOf(statement.get("text")), bound, text));
    }
    return switch (driver) {
      case DUCKDB -> duckdb(planned, relations, read(keys), allocator);
      case SQLITE -> sqlite(planned, relations, read(keys), allocator);
      case POSTGRES -> postgres(planned, texts(keys), allocator);
    };
  }

  /** Each key set read once, whole: a key set is the core's, and small. */
  private static Map<Integer, KeySets.Keys> read(Map<Integer, ArrowReader> keys) throws Exception {
    Map<Integer, KeySets.Keys> read = new LinkedHashMap<>();
    for (Map.Entry<Integer, ArrowReader> each : keys.entrySet()) {
      read.put(each.getKey(), KeySets.Keys.read(each.getValue()));
    }
    return read;
  }

  /** Each key set's text read once: the one cell the core wrote ({@link KeySets#text}). */
  private static Map<Integer, String> texts(Map<Integer, ArrowReader> keys) throws Exception {
    Map<Integer, String> texts = new LinkedHashMap<>();
    for (Map.Entry<Integer, ArrowReader> each : keys.entrySet()) {
      texts.put(each.getKey(), KeySets.text(each.getValue()));
    }
    return texts;
  }

  // --- DuckDB ---------------------------------------------------------------------------

  /**
   * On a cursor of its own (the connection {@code duplicate()}d, as the Python provider takes a
   * cursor), each key set bound, the statements run, and the last one's result exported by the
   * driver as Arrow, every batch it made.
   */
  private ArrowReader duckdb(
      List<Planned> planned,
      Map<String, Integer> relations,
      Map<Integer, KeySets.Keys> keys,
      BufferAllocator allocator)
      throws Exception {
    Trace trace = Trace.installed();
    try (Connection cursor = DuckDb.duplicate(connection)) {
      if (trace != null) {
        trace.cursor(0);
      }
      for (Map.Entry<String, Integer> relation : relations.entrySet()) {
        keySets.bind(cursor, relation.getKey(), keys.get(relation.getValue()), allocator);
      }
      try (Statement statement = cursor.createStatement()) {
        for (int at = 0; at < planned.size() - 1; at++) {
          run(statement, planned.get(at).text(), trace);
        }
        String last = planned.get(planned.size() - 1).text();
        if (trace != null) {
          trace.statement(0, last);
        }
        try (ResultSet rows = statement.executeQuery(last)) {
          List<Map<String, Object>> types = types(rows, trace);
          return answered(
              DuckDb.arrow(rows, allocator, DUCKDB_BATCH_ROWS), types, allocator, trace);
        } catch (SQLException e) {
          if (trace != null) {
            trace.statementFailed(e.getClass().getName());
          }
          throw e;
        }
      }
    }
  }

  // --- SQLite ---------------------------------------------------------------------------

  /** Each key set ingested, the statements run on one statement of the connection, as ADBC. */
  private ArrowReader sqlite(
      List<Planned> planned,
      Map<String, Integer> relations,
      Map<Integer, KeySets.Keys> keys,
      BufferAllocator allocator)
      throws Exception {
    Trace trace = Trace.installed();
    if (trace != null) {
      trace.cursor(0);
    }
    for (Map.Entry<String, Integer> relation : relations.entrySet()) {
      keySets.bind(connection, relation.getKey(), keys.get(relation.getValue()), allocator);
    }
    try (Statement statement = connection.createStatement()) {
      for (int at = 0; at < planned.size() - 1; at++) {
        run(statement, planned.get(at).text(), trace);
      }
      String last = planned.get(planned.size() - 1).text();
      if (trace != null) {
        trace.statement(0, last);
      }
      try (ResultSet rows = statement.executeQuery(last)) {
        List<Map<String, Object>> types = types(rows, trace);
        ConversionTable.Result result = ConversionTable.sqlite(rows);
        return answered(built(result, allocator), types, allocator, trace);
      } catch (SQLException e) {
        if (trace != null) {
          trace.statementFailed(e.getClass().getName());
        }
        throw e;
      }
    }
  }

  // --- Postgres -------------------------------------------------------------------------

  /**
   * The statements run in a transaction as the reference's reads take one (psycopg's {@code
   * connection.transaction()}): none under autocommit; outside a transaction, one of its own,
   * committed, or rolled back if the read fails; inside the caller's, a savepoint, so a failed read
   * leaves the caller's transaction usable and the caller's work is the caller's to commit. Each
   * statement binds its key set's text, as the core wrote it, as its one parameter, every value
   * read in text, and the last one's rows are read by the Postgres column of the conversion table.
   */
  private ArrowReader postgres(
      List<Planned> planned, Map<Integer, String> keys, BufferAllocator allocator)
      throws Exception {
    Trace trace = Trace.installed();
    if (trace != null) {
      trace.cursor(0);
    }
    // Every value read in text, as psycopg reads it: a binary transfer the caller's connection
    // forces is off for this read's statements, and given back after.
    PgSession.Forced forcedBinary = PgSession.binaryOff(connection);
    ArrowReader read;
    try {
      read = postgresRead(planned, keys, allocator, trace);
    } catch (Exception | Error failed) {
      // The read's own failure is the one told; a failure to restore rides on it.
      try {
        PgSession.restoreBinary(connection, forcedBinary);
      } catch (SQLException restoring) {
        failed.addSuppressed(restoring);
      }
      throw failed;
    }
    try {
      PgSession.restoreBinary(connection, forcedBinary);
    } catch (SQLException restoring) {
      read.close();
      throw restoring;
    }
    return read;
  }

  private ArrowReader postgresRead(
      List<Planned> planned,
      Map<Integer, String> keys,
      BufferAllocator allocator,
      Trace trace)
      throws Exception {
    boolean managed = !connection.getAutoCommit();
    Savepoint savepoint =
        managed && PgSession.inTransaction(connection) ? connection.setSavepoint() : null;
    boolean own = managed && savepoint == null;
    try {
      ArrowReader answer = null;
      for (int at = 0; at < planned.size(); at++) {
        Planned statement = planned.get(at);
        boolean last = at == planned.size() - 1;
        if (trace != null) {
          trace.statement(0, statement.text());
        }
        try (Statement prepared = prepared(statement, keys)) {
          if (!last) {
            execute(prepared, statement.text());
            continue;
          }
          try (ResultSet rows = executeQuery(prepared, statement.text())) {
            List<Map<String, Object>> types = types(rows, trace);
            answer = answered(postgresRows(rows, allocator), types, allocator, trace);
          }
        } catch (SQLException e) {
          if (trace != null) {
            trace.statementFailed(e.getClass().getName());
          }
          throw e;
        }
      }
      if (savepoint != null) {
        connection.releaseSavepoint(savepoint);
      } else if (own) {
        connection.commit();
      }
      return answer;
    } catch (Exception e) {
      try {
        if (savepoint != null) {
          connection.rollback(savepoint);
        } else if (own) {
          connection.rollback();
        }
      } catch (SQLException ignored) {
        // The read's own failure is the one told.
      }
      throw e;
    }
  }

  /**
   * The statement as the driver runs it: one with no key set as a plain statement, read with no
   * escape processing, so its text is sent as the core wrote it; one binding a key set's text
   * prepared with it as its parameter.
   */
  private Statement prepared(Planned statement, Map<Integer, String> keys)
      throws SQLException, PlanNotRun {
    if (statement.keyText() == null) {
      Statement plain = connection.createStatement();
      try {
        plain.setEscapeProcessing(false);
        PgSession.inText(plain);
      } catch (SQLException e) {
        plain.close();
        throw e;
      }
      return plain;
    }
    String bound = keys.get(statement.keyText());
    PreparedStatement prepared =
        connection.prepareStatement(KeySets.jdbcText(statement.text()));
    try {
      PgSession.inText(prepared);
      prepared.setString(1, bound);
    } catch (SQLException | RuntimeException e) {
      prepared.close();
      throw e;
    }
    return prepared;
  }

  private static void execute(Statement statement, String text) throws SQLException {
    if (statement instanceof PreparedStatement prepared) {
      prepared.execute();
    } else {
      statement.execute(text);
    }
  }

  private static ResultSet executeQuery(Statement statement, String text) throws SQLException {
    return statement instanceof PreparedStatement prepared
        ? prepared.executeQuery()
        : statement.executeQuery(text);
  }

  private ArrowReader postgresRows(ResultSet rows, BufferAllocator allocator) throws Exception {
    List<ConversionTable.Described> described =
        ConversionTable.described(connection, rows.getMetaData());
    List<ConversionTable.Column> columns = new ArrayList<>();
    List<org.apache.arrow.vector.types.pojo.Field> fields = new ArrayList<>();
    for (int at = 0; at < described.size(); at++) {
      ConversionTable.Described column = described.get(at);
      ConversionTable.Column reading =
          table.postgres(column.oid(), column.precision(), column.scale());
      columns.add(reading);
      fields.add(reading.field(rows.getMetaData().getColumnLabel(at + 1)));
    }
    List<List<Object>> values = new ArrayList<>();
    for (int at = 0; at < columns.size(); at++) {
      values.add(new ArrayList<>());
    }
    ConversionTable.Session session = new Session(connection);
    while (rows.next()) {
      for (int at = 0; at < columns.size(); at++) {
        values.get(at).add(columns.get(at).read(rows, at + 1, session));
      }
    }
    return built(new ConversionTable.Result(fields, values), allocator);
  }

  /** The session's {@code IntervalStyle}, asked once only when a value needs it. */
  private static final class Session implements ConversionTable.Session {
    private final Connection connection;
    private String intervalStyle;

    Session(Connection connection) {
      this.connection = connection;
    }

    @Override
    public String intervalStyle() throws SQLException {
      if (intervalStyle == null) {
        intervalStyle = String.valueOf(PgSession.parameter(connection, "IntervalStyle"));
      }
      return intervalStyle;
    }

  }

  // --- shared ---------------------------------------------------------------------------

  private static void run(Statement statement, String text, Trace trace) throws SQLException {
    if (trace != null) {
      trace.statement(0, text);
    }
    try {
      statement.execute(text);
    } catch (SQLException e) {
      if (trace != null) {
        trace.statementFailed(e.getClass().getName());
      }
      throw e;
    }
  }

  /** A result read from rows, as batches of at most {@link #BATCH_ROWS} rows. */
  static ArrowReader built(ConversionTable.Result result, BufferAllocator allocator) {
    Schema schema = new Schema(result.fields());
    int rows = result.columns().isEmpty() ? 0 : result.columns().get(0).size();
    List<VectorSchemaRoot> batches = new ArrayList<>();
    try {
      for (int start = 0; start < rows || (start == 0 && batches.isEmpty()); start += BATCH_ROWS) {
        int end = Math.min(rows, start + BATCH_ROWS);
        List<List<Object>> slice = new ArrayList<>();
        for (List<Object> column : result.columns()) {
          slice.add(column.subList(start, end));
        }
        batches.add(ArrowColumns.batch(schema, slice, allocator));
        if (rows == 0) {
          break;
        }
      }
    } catch (RuntimeException e) {
      batches.forEach(VectorSchemaRoot::close);
      throw e;
    }
    return ArrowData.reader(schema, batches, allocator);
  }

  /** Each column of {@code rows} as the driver describes it, for a listener. */
  private static List<Map<String, Object>> types(ResultSet rows, Trace trace)
      throws SQLException {
    if (trace == null) {
      return null;
    }
    java.sql.ResultSetMetaData meta = rows.getMetaData();
    List<Map<String, Object>> columns = new ArrayList<>();
    for (int at = 1; at <= meta.getColumnCount(); at++) {
      Map<String, Object> column = new LinkedHashMap<>();
      column.put("type", meta.getColumnTypeName(at));
      column.put("precision", meta.getPrecision(at));
      column.put("scale", meta.getScale(at));
      columns.add(column);
    }
    return columns;
  }

  /** The answer, told to the listener read whole when there is one. */
  private static ArrowReader answered(
      ArrowReader answer,
      List<Map<String, Object>> types,
      BufferAllocator allocator,
      Trace trace) {
    if (trace == null) {
      return answer;
    }
    byte[] ipc = ArrowData.ipc(answer);
    trace.answered(ipc, types);
    return ArrowData.reader(ipc, allocator);
  }
}
