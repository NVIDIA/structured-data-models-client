// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.provider;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * One request the core sends a {@link com.nvidia.sdm.Source}: the Java view of the contract's {@code
 * DataRequest}. Every request is one of the records below.
 *
 * <p>Answers name their columns as the request does, in its order. A request that carries seeds
 * ({@link Neighbors}, {@link Lookup}) answers with the seed's {@code batch} position first, as
 * documented on each.
 *
 * <p>Each value ({@link Bound#value}, {@link Compare#value}, {@link In#values}, {@link
 * SampleEntities#seed}) is a {@link Value}: a plain Java value with the Arrow type the core holds it
 * as. The plain values are {@code null}, {@link Boolean}, {@link Byte}, {@link Short}, {@link
 * Integer}, {@link Long}, {@link java.math.BigInteger} (unsigned 64-bit), {@link Float}, {@link
 * Double}, {@link java.math.BigDecimal}, {@link String}, {@code byte[]}, {@link
 * java.time.LocalDate}, {@link java.time.Instant} (a timestamp with a time zone, which its type
 * names), {@link java.time.LocalDateTime} (a timestamp without one), {@link java.time.Duration} or
 * {@link java.time.LocalTime}.
 */
public sealed interface DataRequest {

  /** The answer column naming the seed a {@link Neighbors} or {@link Lookup} row answers. */
  String BATCH_COLUMN = "__kumo_batch__";

  /** The answer column of a {@link Lookup} counting the rows that hold the seed's key. */
  String MATCHES_COLUMN = "__kumo_matches__";

  /** The column of a seed key set holding each seed's key ({@code key}; {@code key_1}, {@code key_2}, ... for a key of several columns). */
  String SEED_KEY = "key";

  /** The column of a seed key set holding each seed's batch position. */
  String SEED_BATCH = "batch";

  /** The request's name in the core's contract, such as {@code "neighbors"}. */
  String name();

  // --- requests ------------------------------------------------------------------------------

  /** One part of the catalog, for the tables named (every table when {@code tables} is empty). */
  record Describe(List<Relation> tables, DescribePart part) implements DataRequest {
    public Describe {
      tables = List.copyOf(tables);
      Objects.requireNonNull(part, "part");
    }

    public String name() {
      return "describe";
    }
  }

  /**
   * Every row of {@code table} meeting all {@code filters}, {@code columns} in order. With a
   * {@code limit}, any {@code limit} of those rows.
   */
  record Scan(Relation table, List<String> columns, List<Filter> filters, OptionalLong limit)
      implements DataRequest {
    public Scan {
      columns = List.copyOf(columns);
      filters = List.copyOf(filters);
    }

    public String name() {
      return "scan";
    }
  }

  /** The first {@code limit} rows of {@code table}, used to infer column types. */
  record Preview(Relation table, List<String> columns, long limit) implements DataRequest {
    public Preview {
      columns = List.copyOf(columns);
    }

    public String name() {
      return "preview";
    }
  }

  /** One row, one {@code int64} column: the rows of {@code table} meeting all filters. */
  record CountRows(Relation table, List<Filter> filters) implements DataRequest {
    public CountRows {
      filters = List.copyOf(filters);
    }

    public String name() {
      return "count_rows";
    }
  }

  /**
   * One row per key, in order: its label, the table's rows, the key's distinct values among rows
   * holding all of it, and the rows missing any part of it (schema given as {@code expected}).
   */
  record KeyCounts(List<TableKey> keys) implements DataRequest {
    public KeyCounts {
      keys = List.copyOf(keys);
    }

    public String name() {
      return "key_counts";
    }
  }

  /**
   * One row per column, in order: {@code table_index}, its position (an {@code int32} or {@code
   * int64}), then {@code min_date} and {@code max_date}, its smallest and largest value, null where
   * no row holds one, each of the column's own type: a timestamp of any unit or zone, a date, or
   * text as a source with no time type stores it (read by {@code Capabilities.textTimes}).
   */
  record TimeBounds(List<TableColumn> columns) implements DataRequest {
    public TimeBounds {
      columns = List.copyOf(columns);
    }

    public String name() {
      return "time_bounds";
    }
  }

  /**
   * One row: the rows of {@code table}, then for each column in order its smallest value, largest
   * value (each of the column's own type, null when no row holds one) and the rows holding a value.
   */
  record ColumnRanges(Relation table, List<String> columns) implements DataRequest {
    public ColumnRanges {
      columns = List.copyOf(columns);
    }

    public String name() {
      return "column_ranges";
    }
  }

  /** At most one row: a value of {@code key} that more than one row of {@code table} holds. */
  record RepeatedKey(Relation table, List<String> key) implements DataRequest {
    public RepeatedKey {
      key = List.copyOf(key);
    }

    public String name() {
      return "repeated_key";
    }
  }

  /**
   * A sample of {@code n} entities of {@code table} by {@code key}, drawn with {@code seed}: {@code
   * columns} in order, from the rows meeting every filter. {@code rows} is the table's row count,
   * where the core counted it ({@code Capabilities.sampleByRowCount}); {@code keyDtype} the
   * schema's data type of a key of one column, where the source samples by it ({@code
   * Capabilities.sampleByKeyType}).
   */
  record SampleEntities(
      Relation table,
      List<String> key,
      long n,
      Value seed,
      List<String> columns,
      List<Filter> filters,
      OptionalLong rows,
      Optional<String> keyDtype)
      implements DataRequest {
    public SampleEntities {
      key = List.copyOf(key);
      columns = List.copyOf(columns);
      filters = List.copyOf(filters);
      Objects.requireNonNull(seed, "seed");
      Objects.requireNonNull(rows, "rows");
      Objects.requireNonNull(keyDtype, "keyDtype");
    }

    public String name() {
      return "sample_entities";
    }
  }

  /**
   * The rows of {@code table} whose {@code key} equals a seed's key: first {@link #BATCH_COLUMN},
   * the seed's {@code batch} position (of the seed key set's {@code batch} column's type), then
   * {@code columns} in order.
   *
   * <p>A row is kept for a seed only within the seed's {@code window}, where one is given. With
   * {@code perSeed}, at most that many rows per seed, the first in {@code order}, and the answer is
   * ordered by {@code (batch, order)}; without it, there is no order.
   */
  record Neighbors(
      Relation table,
      List<String> key,
      List<String> columns,
      KeySet seeds,
      Optional<Window> window,
      Optional<NeighborLimit> perSeed,
      List<TieTerm> order)
      implements DataRequest {
    public Neighbors {
      key = List.copyOf(key);
      columns = List.copyOf(columns);
      order = List.copyOf(order);
    }

    public String name() {
      return "neighbors";
    }
  }

  /**
   * One row per seed whose key {@code table} holds: {@link #BATCH_COLUMN}, the seed's {@code
   * batch} position; {@link #MATCHES_COLUMN}, the rows holding that key (non-null {@code int64});
   * then {@code columns} in order. Which row of a key several rows hold is not defined.
   */
  record Lookup(Relation table, List<String> key, List<String> columns, KeySet seeds)
      implements DataRequest {
    public Lookup {
      key = List.copyOf(key);
      columns = List.copyOf(columns);
    }

    public String name() {
      return "lookup";
    }
  }

  // --- parts of requests -----------------------------------------------------------------------

  /**
   * A table, by its name's parts, each quoted alone ({@code ["main", "t"]} is {@code t} of {@code
   * main}).
   *
   * <p>A source declaring {@code Capabilities.derivedRelations} may be asked for a derived relation,
   * as a semantic model's definition gives one: its {@code columns}, each a name with the source
   * column or SQL expression it reads, and a {@code filter}, an SQL condition every row of it holds.
   * Both are text of the source's own SQL. With {@code found}, the columns are the ones the source's
   * catalog found, each read by its own name, and the source may read the table through them.
   */
  record Relation(
      List<String> parts, List<RelationColumn> columns, Optional<String> filter, boolean found) {
    public Relation {
      parts = List.copyOf(parts);
      columns = List.copyOf(columns);
      Objects.requireNonNull(filter, "filter");
      if (parts.isEmpty()) {
        throw new IllegalArgumentException("a relation has at least one name part");
      }
    }

    /** The table {@code parts} names, read whole. */
    public Relation(List<String> parts) {
      this(parts, List.of(), Optional.empty(), false);
    }

    /** The last part: the table's own name. */
    public String table() {
      return parts.get(parts.size() - 1);
    }
  }

  /**
   * A column of a derived {@link Relation}: its name, and what it reads, exactly one of a column of
   * the source ({@code column}) or an expression in the source's own SQL ({@code expression}).
   */
  record RelationColumn(String name, Optional<String> column, Optional<String> expression) {
    public RelationColumn {
      Objects.requireNonNull(name, "name");
      Objects.requireNonNull(column, "column");
      Objects.requireNonNull(expression, "expression");
    }
  }

  /**
   * A value of a request: the plain Java value, and the Arrow type the core holds it as, such as
   * {@code new ArrowType.Int(64, true)}.
   */
  record Value(Object value, ArrowType type) {
    public Value {
      Objects.requireNonNull(type, "type");
    }
  }

  /** Which part of the catalog a {@link Describe} asks for. */
  enum DescribePart {
    TABLES,
    COLUMNS,
    KEYS,
    FOREIGN_KEYS,
    INDEXES,
    INDEX_COLUMNS,
    NAMESPACE,
    KEY_ENFORCEMENT,
    VIEW_DEFINITION
  }

  /** A condition every row read must hold; a request's filters all hold. */
  sealed interface Filter {}

  /** The row's {@code columns} equal one of the key set's rows. */
  record KeyIn(List<String> columns, KeySet keys) implements Filter {
    public KeyIn {
      columns = List.copyOf(columns);
    }
  }

  /** The row's {@code column} lies within the bounds given. */
  record TimeRange(String column, Optional<Bound> lower, Optional<Bound> upper) implements Filter {}

  /** The row's {@code column} is not null. */
  record NotNull(String column) implements Filter {}

  /**
   * A value filter: the row's {@code column} compared with {@code value} by {@code op}; a null
   * meets no comparison. Sent only to a provider declaring {@link FilterKind#COMPARE}.
   *
   * <p>A value filter narrows what a read returns and never decides a result: a provider may
   * return rows it does not hold for, but must never leave out a row it holds for. The library
   * checks every row it keeps by the query's own rules.
   */
  record Compare(String column, CompareOp op, Value value) implements Filter {
    public Compare {
      Objects.requireNonNull(value, "value");
    }
  }

  /** A value filter: the row's {@code column} equals one of {@code values}; a null equals none. */
  record In(String column, List<Value> values) implements Filter {
    public In {
      values = List.copyOf(values);
    }
  }

  /** A value filter: the row's {@code column} is null. */
  record IsNull(String column) implements Filter {}

  /** A value filter: at least one of {@code filters} holds. */
  record AnyOf(List<Filter> filters) implements Filter {
    public AnyOf {
      filters = List.copyOf(filters);
    }
  }

  /** A value filter: every one of {@code filters} holds. */
  record AllOf(List<Filter> filters) implements Filter {
    public AllOf {
      filters = List.copyOf(filters);
    }
  }

  /**
   * A value filter: the row's {@code column} is referred to by at least {@code atLeast} rows of
   * {@code table} through its column {@code key} that meet {@code filters} (a {@link TimeRange},
   * not-null and value filters on {@code table}'s rows).
   */
  record HasRows(String column, Relation table, String key, List<Filter> filters, long atLeast)
      implements Filter {
    public HasRows {
      filters = List.copyOf(filters);
    }
  }

  /** How a {@link Compare} compares. */
  enum CompareOp {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE
  }

  /** A kind of value filter a provider may declare it applies ({@code Capabilities.filters}). */
  enum FilterKind {
    COMPARE("compare"),
    IN("in"),
    IS_NULL("is_null"),
    ANY_OF("any_of"),
    ALL_OF("all_of"),
    HAS_ROWS("has_rows");

    private final String wireName;

    FilterKind(String wireName) {
      this.wireName = wireName;
    }

    /** The kind's name on the wire. */
    public String wireName() {
      return wireName;
    }
  }

  /** A bound on a value. */
  record Bound(Value value, boolean inclusive) {
    public Bound {
      Objects.requireNonNull(value, "value");
    }
  }

  /**
   * A set of keys, as Arrow rows: {@code schema} declares its columns, each with what the core notes
   * of it in its field metadata (such as the joined key's declared type), which a source may ignore.
   * A provider reads {@code reader} once; the library closes it.
   */
  record KeySet(Schema schema, ArrowReader reader) {
    public KeySet {
      Objects.requireNonNull(schema, "schema");
      Objects.requireNonNull(reader, "reader");
    }

    /** The key set's column names, in order: what each column means to the request. */
    public List<String> columns() {
      return schema.getFields().stream().map(Field::getName).toList();
    }
  }

  /**
   * Which of a seed's rows count: the row's {@code column} against the seed's own time columns,
   * named by {@link SeedBound#seedColumn} in the seed key set. {@code sourceType} is the column's
   * type as the source's catalog declares it, where the core read it.
   */
  record Window(
      String column,
      Optional<SeedBound> lower,
      Optional<SeedBound> upper,
      Optional<String> sourceType) {
    public Window {
      Objects.requireNonNull(sourceType, "sourceType");
    }

    /** The window of {@code column} between {@code lower} and {@code upper}. */
    public Window(String column, Optional<SeedBound> lower, Optional<SeedBound> upper) {
      this(column, lower, upper, Optional.empty());
    }
  }

  /** A window bound: the seed column holding it, and whether it is inclusive. */
  record SeedBound(String seedColumn, boolean inclusive) {}

  /** The most rows per seed. */
  sealed interface NeighborLimit {}

  /** At most {@code count} rows per seed. */
  record Count(long count) implements NeighborLimit {}

  /**
   * All but the last {@code |value|} rows per seed, as a negative limit in the reference means;
   * {@code value} is the number as written.
   */
  record AllBut(String value) implements NeighborLimit {}

  /** As many rows per seed as {@code expression}, in the source's own SQL, says. */
  record SqlLimit(String expression) implements NeighborLimit {}

  /** One term of a row order. */
  record TieTerm(
      String column, boolean asTimestamp, boolean descending, boolean nullsLast, boolean stringLike) {}

  /** A column of a table. */
  record TableColumn(Relation table, String column) {}

  /** A key of a table, with the label its answer row carries. */
  record TableKey(Relation table, List<String> columns, long label, boolean unique) {
    public TableKey {
      columns = List.copyOf(columns);
    }
  }
}
