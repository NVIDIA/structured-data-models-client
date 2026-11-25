// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.examples;

import com.nvidia.sdm.Source;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.DataRequest.AllOf;
import com.nvidia.sdm.provider.DataRequest.AnyOf;
import com.nvidia.sdm.provider.DataRequest.Bound;
import com.nvidia.sdm.provider.DataRequest.Compare;
import com.nvidia.sdm.provider.DataRequest.CountRows;
import com.nvidia.sdm.provider.DataRequest.Describe;
import com.nvidia.sdm.provider.DataRequest.Filter;
import com.nvidia.sdm.provider.DataRequest.FilterKind;
import com.nvidia.sdm.provider.DataRequest.HasRows;
import com.nvidia.sdm.provider.DataRequest.In;
import com.nvidia.sdm.provider.DataRequest.IsNull;
import com.nvidia.sdm.provider.DataRequest.KeyIn;
import com.nvidia.sdm.provider.DataRequest.KeySet;
import com.nvidia.sdm.provider.DataRequest.Lookup;
import com.nvidia.sdm.provider.DataRequest.NotNull;
import com.nvidia.sdm.provider.DataRequest.Preview;
import com.nvidia.sdm.provider.DataRequest.Relation;
import com.nvidia.sdm.provider.DataRequest.SampleEntities;
import com.nvidia.sdm.provider.DataRequest.Scan;
import com.nvidia.sdm.provider.DataRequest.TimeRange;
import com.nvidia.sdm.provider.DataRequest.Value;
import com.nvidia.sdm.provider.FailureFacts;
import com.nvidia.sdm.provider.OptionalRequest;
import com.nvidia.sdm.provider.ProviderFailure;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Types;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TimeStampMicroTZVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Your own {@link Source} over a query engine: Trino, through its JDBC driver ({@code
 * io.trino:trino-jdbc}).
 *
 * <p>A complete example of a source for an engine the client has no built-in source for, written
 * against the public API only ({@code com.nvidia.sdm} and {@code com.nvidia.sdm.provider}). It
 * turns each typed request the client sends into one Trino query:
 *
 * <ul>
 *   <li>{@code describe}: the tables, columns and namespace, from Trino's {@code
 *       information_schema}; Trino declares no keys, so the schema names them.
 *   <li>{@code scan} and {@code preview}: a {@code SELECT} with the request's filters.
 *   <li>{@code count_rows}: a {@code count(*)} with the request's filters.
 *   <li>{@code sample_entities}: the rows ordered by a hash of each key and the seed, so the same
 *       seed draws the same rows.
 *   <li>{@code lookup}: the seeds' key set joined to the table, one row per seed.
 * </ul>
 *
 * <p>Every other request ({@code neighbors} among them) is answered by the client in memory, from
 * scans, within the declared fallback. {@link #failure} reads Trino's error codes into what they
 * mean to the client, such as a table that does not exist.
 *
 * <pre>{@code
 * Connection trino = DriverManager.getConnection("jdbc:trino://localhost:8080", "sdm", null);
 * Schema schema = Schema.fromSource(new TrinoSource(trino, "memory", "shop"), options);
 * }</pre>
 *
 * <p>The Python version is {@code packages/structured-data-models-client/tests/trino_source.py};
 * {@code TrinoProviderTest} runs this one against a Trino server where {@code SDM_TRINO_TEST_URL}
 * names one. The connection is the caller's: the source never closes it.
 */
public final class TrinoSource implements Source {
  private final Connection connection;
  private final String catalog;
  private final String schema;
  private final Capabilities capabilities;

  /** The tables of {@code catalog}.{@code schema}, every value filter applied by Trino. */
  public TrinoSource(Connection connection, String catalog, String schema) {
    this(connection, catalog, schema, EnumSet.allOf(FilterKind.class), 1_000);
  }

  /**
   * The tables of {@code catalog}.{@code schema}.
   *
   * @param filters the value filters to have Trino apply; the client applies every value filter
   *     itself as well, so leaving a kind out only widens the reads
   * @param maxKeysPerRequest the most seeds one {@code lookup} carries; more reach the source as
   *     several lookups
   */
  public TrinoSource(
      Connection connection,
      String catalog,
      String schema,
      Set<FilterKind> filters,
      long maxKeysPerRequest) {
    this.connection = connection;
    this.catalog = catalog;
    this.schema = schema;
    this.capabilities =
        Capabilities.builder()
            .answers(
                OptionalRequest.PREVIEW,
                OptionalRequest.COUNT_ROWS,
                OptionalRequest.SAMPLE_ENTITIES,
                OptionalRequest.LOOKUP)
            .filters(filters.toArray(FilterKind[]::new))
            .maxKeysPerRequest(maxKeysPerRequest)
            // The client answers every other request from scans, within these limits.
            .fallback(10_000_000, 1L << 31)
            .namespace(List.of(catalog, schema))
            .build();
  }

  // --- the two methods every source implements ---------------------------------------------

  @Override
  public Capabilities capabilities() {
    return capabilities;
  }

  @Override
  public ArrowReader answer(
      DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
    if (request instanceof Describe describe) {
      // A describe answer's shape is fixed: the client gives it.
      return arrow(expected.orElseThrow(), describe(describe), allocator);
    }
    if (request instanceof Scan scan) {
      String limit = scan.limit().isPresent() ? " LIMIT " + scan.limit().getAsLong() : "";
      return query(
          "SELECT " + names(scan.columns()) + " FROM " + table(scan.table())
              + where(scan.filters(), false) + limit,
          allocator);
    }
    if (request instanceof Preview preview) {
      return query(
          "SELECT " + names(preview.columns()) + " FROM " + table(preview.table())
              + " LIMIT " + preview.limit(),
          allocator);
    }
    if (request instanceof CountRows count) {
      // A count is exact, so every filter must be written.
      return query(
          "SELECT count(*) AS \"rows\" FROM " + table(count.table()) + where(count.filters(), true),
          allocator);
    }
    if (request instanceof SampleEntities sample) {
      String key =
          sample.key().stream()
              .map(k -> "CAST(" + quoted(k) + " AS VARCHAR)")
              .collect(Collectors.joining(", '|', "));
      String seed = text(String.valueOf(sample.seed().value()));
      return query(
          "SELECT " + names(sample.columns()) + " FROM " + table(sample.table())
              + where(sample.filters(), false)
              + " ORDER BY xxhash64(to_utf8(concat(" + key + ", '|', " + seed + "))) LIMIT "
              + sample.n(),
          allocator);
    }
    if (request instanceof Lookup lookup) {
      return lookup(lookup, allocator);
    }
    // The client sends no request this source did not declare.
    throw new UnsupportedOperationException(request.name());
  }

  /**
   * What a Trino error means to the client, by its error code; any other failure is read by its
   * message alone.
   */
  @Override
  public ProviderFailure failure(Exception failure) {
    if (!(failure instanceof SQLException sql)) {
      return ProviderFailure.of(failure);
    }
    return new ProviderFailure(
        Optional.ofNullable(sql.getSQLState()),
        Optional.of(Integer.toString(sql.getErrorCode())),
        Optional.empty(),
        Optional.ofNullable(sql.getMessage()),
        Optional.ofNullable(FACTS.get(sql.getErrorCode())));
  }

  // --- describe ------------------------------------------------------------------------------

  private List<List<Object>> describe(Describe request) throws SQLException {
    String inSchema = "table_schema = " + text(schema);
    String information = quoted(catalog) + ".information_schema";
    switch (request.part()) {
      case TABLES -> {
        String only =
            request.tables().isEmpty()
                ? ""
                : " AND table_name IN ("
                    + request.tables().stream()
                        .map(r -> text(r.table()))
                        .collect(Collectors.joining(", "))
                    + ")";
        return rows(
            "SELECT table_name FROM " + information + ".tables WHERE " + inSchema + only
                + " ORDER BY table_name");
      }
      case COLUMNS -> {
        List<List<Object>> rows = new ArrayList<>();
        for (List<Object> row :
            rows(
                "SELECT column_name, data_type, is_nullable = 'NO' FROM " + information
                    + ".columns WHERE " + inSchema + " AND table_name = "
                    + text(request.tables().get(0).table()) + " ORDER BY ordinal_position")) {
          rows.add(List.of(row.get(0), row.get(1), row.get(2), false));
        }
        return rows;
      }
      case NAMESPACE -> {
        return List.of(List.of(catalog), List.of(schema));
      }
      default -> {
        // Trino declares no keys, foreign keys or indexes: an empty answer.
        return List.of();
      }
    }
  }

  // --- lookup --------------------------------------------------------------------------------

  /**
   * One row per seed whose key the table holds: the seed's batch position, the rows holding the
   * key, and the columns of one of them.
   */
  private ArrowReader lookup(Lookup request, BufferAllocator allocator) throws Exception {
    int width = request.key().size();
    // The seed key set's columns: the key's (key, or key_1, key_2, ...) and each seed's batch.
    List<String> keys = new ArrayList<>();
    if (width == 1) keys.add(DataRequest.SEED_KEY);
    else for (int i = 1; i <= width; i++) keys.add(DataRequest.SEED_KEY + "_" + i);
    keys.add(DataRequest.SEED_BATCH);
    List<String> tuples =
        keyRows(request.seeds(), keys).stream()
            .map(row -> row.stream().map(TrinoSource::literal).collect(Collectors.joining(", ", "(", ")")))
            .toList();
    // A VALUES list of no row is no SQL: no seed, an empty seed table.
    String values =
        tuples.isEmpty()
            ? "(" + String.join(", ", java.util.Collections.nCopies(width + 1, "NULL")) + ")"
            : String.join(", ", tuples);
    var seeded = new ArrayList<String>();
    var on = new ArrayList<String>();
    for (int i = 0; i < width; i++) {
      seeded.add("k" + i);
      on.add("t." + quoted(request.key().get(i)) + " = s.k" + i);
    }
    String picked =
        request.columns().stream()
            .map(c -> ", arbitrary(t." + quoted(c) + ") AS " + quoted(c))
            .collect(Collectors.joining());
    String sql =
        "SELECT s.batch AS " + quoted(DataRequest.BATCH_COLUMN) + ", count(*) AS "
            + quoted(DataRequest.MATCHES_COLUMN) + picked + " FROM (VALUES " + values + ") AS s("
            + String.join(", ", seeded) + ", batch) JOIN " + table(request.table()) + " AS t ON "
            + String.join(" AND ", on) + " GROUP BY s.batch ORDER BY s.batch";
    // The batch column is of the seed key set's batch type.
    Field batch = request.seeds().schema().findField(DataRequest.SEED_BATCH);
    try (Statement statement = connection.createStatement();
        ResultSet result = statement.executeQuery(sql)) {
      List<Field> fields = fields(result.getMetaData());
      fields.set(0, new Field(DataRequest.BATCH_COLUMN, batch.getFieldType(), null));
      return arrow(new Schema(fields), rowsOf(result), allocator);
    }
  }

  /** Each row of a key set as typed values, its {@code columns} in order. */
  private static List<List<Value>> keyRows(KeySet keys, List<String> columns) throws Exception {
    List<List<Value>> rows = new ArrayList<>();
    try (ArrowReader reader = keys.reader()) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        for (int at = 0; at < root.getRowCount(); at++) {
          List<Value> row = new ArrayList<>();
          for (String column : columns) {
            FieldVector vector = root.getVector(column);
            row.add(new Value(plain(vector.getObject(at)), vector.getField().getType()));
          }
          rows.add(row);
        }
      }
    }
    return rows;
  }

  /** An Arrow vector's value as a request's plain value. */
  private static Object plain(Object value) {
    return value instanceof org.apache.arrow.vector.util.Text text ? text.toString() : value;
  }

  // --- filters -------------------------------------------------------------------------------

  private String where(List<Filter> filters, boolean exact) throws Exception {
    List<String> written = new ArrayList<>();
    for (Filter filter : filters) {
      String condition = condition(filter);
      if (condition == null && exact) {
        throw new UnsupportedOperationException("an exact read with " + filter);
      }
      if (condition != null) written.add(condition);
    }
    return written.isEmpty() ? "" : " WHERE " + String.join(" AND ", written);
  }

  /** {@code filter} in Trino's SQL; null for a value filter left to the client. */
  private String condition(Filter filter) throws Exception {
    if (filter instanceof KeyIn f) return keyIn(f);
    if (filter instanceof NotNull f) return quoted(f.column()) + " IS NOT NULL";
    if (filter instanceof TimeRange f) {
      List<String> bounds = new ArrayList<>();
      f.lower().ifPresent(b -> bounds.add(bound(f.column(), b, ">")));
      f.upper().ifPresent(b -> bounds.add(bound(f.column(), b, "<")));
      return bounds.isEmpty() ? null : String.join(" AND ", bounds);
    }
    if (filter instanceof IsNull f) return quoted(f.column()) + " IS NULL";
    if (filter instanceof Compare f)
      return quoted(f.column()) + " " + OPERATORS.get(f.op()) + " " + literal(f.value());
    if (filter instanceof In f) {
      if (f.values().isEmpty()) return "FALSE";
      return quoted(f.column()) + " IN ("
          + f.values().stream().map(TrinoSource::literal).collect(Collectors.joining(", ")) + ")";
    }
    if (filter instanceof AllOf f) {
      List<String> parts = new ArrayList<>();
      for (Filter nested : f.filters()) {
        String c = condition(nested);
        if (c != null) parts.add(c);
      }
      return parts.isEmpty() ? null : "(" + String.join(" AND ", parts) + ")";
    }
    if (filter instanceof AnyOf f) {
      List<String> parts = new ArrayList<>();
      for (Filter nested : f.filters()) {
        String c = condition(nested);
        // An OR with any part left out is left out whole.
        if (c == null) return null;
        parts.add(c);
      }
      return parts.isEmpty() ? null : "(" + String.join(" OR ", parts) + ")";
    }
    if (filter instanceof HasRows f) {
      String key = quoted(f.key());
      List<String> within = new ArrayList<>(List.of(key + " IS NOT NULL"));
      for (Filter nested : f.filters()) {
        String c = condition(nested);
        if (c != null) within.add(c);
      }
      String counted =
          f.atLeast() > 1 ? " GROUP BY " + key + " HAVING count(*) >= " + f.atLeast() : "";
      return quoted(f.column()) + " IN (SELECT " + key + " FROM " + table(f.table()) + " WHERE "
          + String.join(" AND ", within) + counted + ")";
    }
    return null;
  }

  private static String bound(String column, Bound bound, String op) {
    return quoted(column) + " " + op + (bound.inclusive() ? "=" : "") + " " + literal(bound.value());
  }

  /** The row's columns equal one of the key set's rows. */
  private static String keyIn(KeyIn filter) throws Exception {
    List<List<Value>> rows = keyRows(filter.keys(), filter.keys().columns());
    if (rows.isEmpty()) return "FALSE";
    if (filter.columns().size() == 1) {
      return quoted(filter.columns().get(0)) + " IN ("
          + rows.stream().map(row -> literal(row.get(0))).collect(Collectors.joining(", ")) + ")";
    }
    String tuples =
        rows.stream()
            .map(row -> row.stream().map(TrinoSource::literal).collect(Collectors.joining(", ", "(", ")")))
            .collect(Collectors.joining(", "));
    return "(" + names(filter.columns()) + ") IN (VALUES " + tuples + ")";
  }

  // --- running a query -------------------------------------------------------------------------

  /** A table by its name's parts; one part is a table of this source's catalog and schema. */
  private String table(Relation relation) {
    List<String> parts =
        relation.parts().size() > 1 ? relation.parts() : List.of(catalog, schema, relation.table());
    return parts.stream().map(TrinoSource::quoted).collect(Collectors.joining("."));
  }

  /** The query's rows as Arrow, typed by the columns Trino says it returns. */
  private ArrowReader query(String sql, BufferAllocator allocator) throws Exception {
    try (Statement statement = connection.createStatement();
        ResultSet result = statement.executeQuery(sql)) {
      return arrow(new Schema(fields(result.getMetaData())), rowsOf(result), allocator);
    }
  }

  private List<List<Object>> rows(String sql) throws SQLException {
    try (Statement statement = connection.createStatement();
        ResultSet result = statement.executeQuery(sql)) {
      return rowsOf(result);
    }
  }

  private static List<List<Object>> rowsOf(ResultSet result) throws SQLException {
    int width = result.getMetaData().getColumnCount();
    List<List<Object>> rows = new ArrayList<>();
    while (result.next()) {
      List<Object> row = new ArrayList<>(width);
      for (int c = 1; c <= width; c++) {
        Object value =
            switch (result.getMetaData().getColumnType(c)) {
              case Types.DATE -> result.getObject(c, LocalDate.class);
              case Types.TIMESTAMP -> result.getObject(c, LocalDateTime.class);
              case Types.TIMESTAMP_WITH_TIMEZONE ->
                  result.getObject(c, java.time.ZonedDateTime.class) == null
                      ? null
                      : result.getObject(c, java.time.ZonedDateTime.class).toInstant();
              default -> result.getObject(c);
            };
        row.add(value);
      }
      rows.add(row);
    }
    return rows;
  }

  // --- Trino's types and values as Arrow's ---------------------------------------------------------

  private static List<Field> fields(ResultSetMetaData columns) throws SQLException {
    List<Field> fields = new ArrayList<>();
    for (int c = 1; c <= columns.getColumnCount(); c++) {
      ArrowType type =
          switch (columns.getColumnType(c)) {
            case Types.BOOLEAN -> ArrowType.Bool.INSTANCE;
            case Types.TINYINT -> new ArrowType.Int(8, true);
            case Types.SMALLINT -> new ArrowType.Int(16, true);
            case Types.INTEGER -> new ArrowType.Int(32, true);
            case Types.BIGINT -> new ArrowType.Int(64, true);
            case Types.REAL -> new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
            case Types.DOUBLE -> new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
            case Types.DECIMAL ->
                new ArrowType.Decimal(columns.getPrecision(c), columns.getScale(c), 128);
            case Types.DATE -> new ArrowType.Date(DateUnit.DAY);
            case Types.TIMESTAMP -> new ArrowType.Timestamp(TimeUnit.MICROSECOND, null);
            case Types.TIMESTAMP_WITH_TIMEZONE -> new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC");
            case Types.VARCHAR, Types.CHAR -> ArrowType.Utf8.INSTANCE;
            case Types.VARBINARY -> ArrowType.Binary.INSTANCE;
            default ->
                throw new UnsupportedOperationException(
                    "a Trino column of type " + columns.getColumnTypeName(c));
          };
      fields.add(new Field(columns.getColumnLabel(c), FieldType.nullable(type), null));
    }
    return fields;
  }

  /** {@code rows} of {@code schema} as an Arrow stream the client reads and closes. */
  private static ArrowReader arrow(Schema schema, List<List<Object>> rows, BufferAllocator allocator)
      throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
        ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bytes)) {
      root.allocateNew();
      for (int at = 0; at < rows.size(); at++)
        for (int c = 0; c < schema.getFields().size(); c++)
          set(root.getVector(c), at, rows.get(at).get(c));
      root.setRowCount(rows.size());
      writer.start();
      writer.writeBatch();
      writer.end();
    }
    return new ArrowStreamReader(new ByteArrayInputStream(bytes.toByteArray()), allocator);
  }

  private static void set(FieldVector vector, int at, Object value) {
    if (value == null) vector.setNull(at);
    else if (vector instanceof BitVector v) v.setSafe(at, (Boolean) value ? 1 : 0);
    else if (vector instanceof TinyIntVector v) v.setSafe(at, ((Number) value).byteValue());
    else if (vector instanceof SmallIntVector v) v.setSafe(at, ((Number) value).shortValue());
    else if (vector instanceof IntVector v) v.setSafe(at, ((Number) value).intValue());
    else if (vector instanceof BigIntVector v) v.setSafe(at, ((Number) value).longValue());
    else if (vector instanceof UInt1Vector v) v.setSafe(at, ((Number) value).intValue());
    else if (vector instanceof UInt2Vector v) v.setSafe(at, ((Number) value).intValue());
    else if (vector instanceof UInt4Vector v) v.setSafe(at, ((Number) value).intValue());
    else if (vector instanceof UInt8Vector v) v.setSafe(at, ((Number) value).longValue());
    else if (vector instanceof Float4Vector v) v.setSafe(at, ((Number) value).floatValue());
    else if (vector instanceof Float8Vector v) v.setSafe(at, ((Number) value).doubleValue());
    else if (vector instanceof DecimalVector v) v.setSafe(at, (BigDecimal) value);
    else if (vector instanceof DateDayVector v) v.setSafe(at, (int) ((LocalDate) value).toEpochDay());
    else if (vector instanceof TimeStampMicroVector v) v.setSafe(at, micros(((LocalDateTime) value).toInstant(ZoneOffset.UTC)));
    else if (vector instanceof TimeStampMicroTZVector v) v.setSafe(at, micros((Instant) value));
    else if (vector instanceof VarCharVector v) v.setSafe(at, value.toString().getBytes(StandardCharsets.UTF_8));
    else if (vector instanceof VarBinaryVector v) v.setSafe(at, (byte[]) value);
    else throw new IllegalStateException("no Arrow writer for " + vector.getField());
  }

  private static long micros(Instant instant) {
    return instant.getEpochSecond() * 1_000_000L + instant.getNano() / 1_000;
  }

  // --- what Trino's errors mean -----------------------------------------------------------------

  private static final FailureFacts MISSING =
      FailureFacts.builder()
          .notFound()
          .unusable(FailureFacts.Unusable.MISSING)
          .reason(FailureFacts.Reason.TABLE_UNRESOLVED)
          .build();

  /** Trino's error codes ({@code io.trino.spi.StandardErrorCode}), as the client reads them. */
  private static final Map<Integer, FailureFacts> FACTS =
      Map.of(
          44, MISSING, // CATALOG_NOT_FOUND
          45, MISSING, // SCHEMA_NOT_FOUND
          46, MISSING, // TABLE_NOT_FOUND
          47, FailureFacts.builder().reason(FailureFacts.Reason.COLUMN_UNRESOLVED).build(),
          4, // PERMISSION_DENIED
          FailureFacts.builder()
              .denied()
              .unusable(FailureFacts.Unusable.FORBIDDEN)
              .reason(FailureFacts.Reason.PERMISSION_DENIED)
              .build(),
          1, FailureFacts.builder().reason(FailureFacts.Reason.SYNTAX_REJECTED).build(),
          58, FailureFacts.builder().reason(FailureFacts.Reason.TYPES_INCOMPATIBLE).build(),
          8, FailureFacts.builder().reason(FailureFacts.Reason.DIVISION_BY_ZERO).build(),
          19, FailureFacts.builder().reason(FailureFacts.Reason.OUT_OF_RANGE).build(),
          9, FailureFacts.builder().reason(FailureFacts.Reason.VALUE_NOT_CONVERTED).build());

  // --- SQL text ----------------------------------------------------------------------------------

  private static final Map<DataRequest.CompareOp, String> OPERATORS =
      Map.of(
          DataRequest.CompareOp.EQ, "=",
          DataRequest.CompareOp.NE, "<>",
          DataRequest.CompareOp.LT, "<",
          DataRequest.CompareOp.LE, "<=",
          DataRequest.CompareOp.GT, ">",
          DataRequest.CompareOp.GE, ">=");

  static String quoted(String name) {
    return "\"" + name.replace("\"", "\"\"") + "\"";
  }

  private static String names(List<String> names) {
    return names.stream().map(TrinoSource::quoted).collect(Collectors.joining(", "));
  }

  private static String text(String text) {
    return "'" + text.replace("'", "''") + "'";
  }

  /**
   * A request's value as a Trino literal of its type: the type the client holds it as says how to
   * write it exactly.
   */
  static String literal(Value given) {
    Object value = given.value();
    ArrowType type = given.type();
    if (value == null) return "NULL";
    if (type instanceof ArrowType.Bool) return (Boolean) value ? "TRUE" : "FALSE";
    if (type instanceof ArrowType.Int) {
      // An unsigned 64-bit value past a BIGINT's range is a BigInteger.
      return value instanceof BigInteger big && big.bitLength() > 63
          ? "CAST(" + big + " AS DECIMAL(20, 0))"
          : value.toString();
    }
    if (type instanceof ArrowType.FloatingPoint point) {
      String name = point.getPrecision() == FloatingPointPrecision.SINGLE ? "REAL" : "DOUBLE";
      double number = ((Number) value).doubleValue();
      if (Double.isNaN(number)) return "CAST(nan() AS " + name + ")";
      if (Double.isInfinite(number))
        return "CAST(" + (number < 0 ? "-" : "") + "infinity() AS " + name + ")";
      return name + " '" + value + "'";
    }
    if (type instanceof ArrowType.Decimal decimal)
      return "CAST('" + ((BigDecimal) value).toPlainString() + "' AS DECIMAL("
          + decimal.getPrecision() + ", " + decimal.getScale() + "))";
    if (type instanceof ArrowType.Utf8 || type instanceof ArrowType.LargeUtf8)
      return text(value.toString());
    if (type instanceof ArrowType.Binary) return "X'" + HexFormat.of().formatHex((byte[]) value) + "'";
    if (type instanceof ArrowType.Date) return "DATE '" + value + "'";
    if (type instanceof ArrowType.Timestamp) {
      // An instant is compared as the UTC wall clock Trino stores.
      LocalDateTime wall =
          value instanceof Instant instant
              ? LocalDateTime.ofInstant(instant, ZoneOffset.UTC)
              : (LocalDateTime) value;
      return "TIMESTAMP '" + wall.format(WALL) + "'";
    }
    if (type instanceof ArrowType.Time) return "TIME '" + ((LocalTime) value) + "'";
    throw new UnsupportedOperationException("a value of type " + type);
  }

  private static final DateTimeFormatter WALL =
      DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSSSSSSSS");
}
