// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * What {@link Schema#fromSource(Source, FromSourceOptions)} is told beside its source, as Python's
 * {@code Schema.from_source} takes it: the tables to describe (each by the name the schema gives
 * it and the name the source holds it under; none named, every table the source holds), the
 * primary keys and time columns the caller declares, the relationships in place of the declared
 * and inferred ones, and whether keys are checked against every row. The core takes or refuses
 * each, by name; this holds them as given.
 *
 * <pre>{@code
 * FromSourceOptions options = FromSourceOptions.builder()
 *     .table("customers", "adventureworks_customers")
 *     .table("orders", "adventureworks_orders")
 *     .timeColumn("orders", "order_date")
 *     .build();
 * }</pre>
 */
public final class FromSourceOptions {
  private final Map<String, String> tables;
  private final Map<String, List<String>> primaryKeys;
  private final Map<String, String> timeColumns;
  private final List<Relationship> relationships;
  private final boolean checkKeys;

  private FromSourceOptions(Builder builder) {
    this.tables = new LinkedHashMap<>(builder.tables);
    this.primaryKeys = new LinkedHashMap<>(builder.primaryKeys);
    this.timeColumns = new LinkedHashMap<>(builder.timeColumns);
    this.relationships = builder.relationships == null ? null : List.copyOf(builder.relationships);
    this.checkKeys = builder.checkKeys;
  }

  /** No table named, nothing declared, keys checked. */
  public static FromSourceOptions defaults() {
    return builder().build();
  }

  public static Builder builder() {
    return new Builder();
  }

  /** The source as the contract's {@code SourceSpec} JSON value. */
  Map<String, Object> spec() {
    List<Object> named = new ArrayList<>();
    tables.forEach(
        (name, source) -> {
          Map<String, Object> table = new LinkedHashMap<>();
          table.put("name", name);
          table.put("source", Map.of("source", List.of(source)));
          named.add(table);
        });
    Map<String, Object> keys = new LinkedHashMap<>();
    primaryKeys.forEach((table, key) -> keys.put(table, given(key)));
    Map<String, Object> times = new LinkedHashMap<>();
    timeColumns.forEach((table, column) -> times.put(table, given(column)));
    Map<String, Object> spec = new LinkedHashMap<>();
    spec.put("tables", named);
    spec.put("primary_keys", keys);
    spec.put("time_columns", times);
    if (relationships != null) {
      spec.put("relationships", Schema.links(relationships));
    }
    return spec;
  }

  /** A value as the contract's {@code Given} writes one the host can write faithfully. */
  static Map<String, Object> given(Object value) {
    Map<String, Object> given = new LinkedHashMap<>();
    given.put("value", value);
    return given;
  }

  boolean checkKeys() {
    return checkKeys;
  }

  /** Builds {@link FromSourceOptions}. */
  public static final class Builder {
    private final Map<String, String> tables = new LinkedHashMap<>();
    private final Map<String, List<String>> primaryKeys = new LinkedHashMap<>();
    private final Map<String, String> timeColumns = new LinkedHashMap<>();
    private List<Relationship> relationships;
    private boolean checkKeys = true;

    private Builder() {}

    /** Describes the source's table {@code sourceName} as the schema's table {@code name}. */
    public Builder table(String name, String sourceName) {
      tables.put(Objects.requireNonNull(name, "name"), Objects.requireNonNull(sourceName, "sourceName"));
      return this;
    }

    /** Describes the source's table {@code name} under its own name. */
    public Builder table(String name) {
      return table(name, name);
    }

    /** Declares {@code table}'s primary key: its column, or its columns in order. */
    public Builder primaryKey(String table, String... columns) {
      primaryKeys.put(Objects.requireNonNull(table, "table"), List.copyOf(Arrays.asList(columns)));
      return this;
    }

    /** Declares {@code table} has no primary key, as Python's {@code None} does. */
    public Builder noPrimaryKey(String table) {
      primaryKeys.put(Objects.requireNonNull(table, "table"), null);
      return this;
    }

    /** Declares {@code table}'s time column. */
    public Builder timeColumn(String table, String column) {
      timeColumns.put(Objects.requireNonNull(table, "table"), Objects.requireNonNull(column, "column"));
      return this;
    }

    /** Declares {@code table} has no time column, as Python's {@code None} does. */
    public Builder noTimeColumn(String table) {
      timeColumns.put(Objects.requireNonNull(table, "table"), null);
      return this;
    }

    /** The relationships, in place of the declared and inferred ones; empty says none. */
    public Builder relationships(List<Relationship> relationships) {
      this.relationships = List.copyOf(relationships);
      return this;
    }

    /** Whether each key is checked against every row (one full scan per table); default true. */
    public Builder checkKeys(boolean checkKeys) {
      this.checkKeys = checkKeys;
      return this;
    }

    public FromSourceOptions build() {
      return new FromSourceOptions(this);
    }
  }
}
