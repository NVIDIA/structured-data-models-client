// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * One table of a schema, as Python's {@code SourceTable}: its name in the schema, the name its
 * source holds it under, its columns in order, its primary key (its columns, empty for none), its
 * time column, and, where keys were checked, how many rows share a key value. A table is plain
 * data: change one with the schema's edits ({@link Schema#withPrimaryKey} and the rest).
 */
public final class Table {
  private final String name;
  private final String sourceName;
  private final List<Column> columns;
  private final List<String> primaryKey;
  private final String timeColumn;
  private final Long duplicateKeys;

  Table(
      String name,
      String sourceName,
      List<Column> columns,
      List<String> primaryKey,
      String timeColumn,
      Long duplicateKeys) {
    this.name = Objects.requireNonNull(name, "name");
    this.sourceName = sourceName;
    this.columns = List.copyOf(columns);
    this.primaryKey = List.copyOf(primaryKey);
    this.timeColumn = timeColumn;
    this.duplicateKeys = duplicateKeys;
  }

  /** The table's name in the schema, which queries use. */
  public String name() {
    return name;
  }

  /** The name the source holds it under. */
  public String sourceName() {
    return sourceName;
  }

  /** Its columns, in order. */
  public List<Column> columns() {
    return columns;
  }

  /**
   * The column named {@code name}.
   *
   * @throws StructuredDataException {@code INVALID_REQUEST} when the table has no such column
   */
  public Column column(String name) {
    for (Column column : columns) {
      if (column.name().equals(name)) {
        return column;
      }
    }
    throw new StructuredDataException(
        "table " + this.name + " has no column " + name,
        "INVALID_REQUEST",
        java.util.Map.of("table", this.name, "column", String.valueOf(name)),
        null);
  }

  /** Its primary key's columns, in order; empty when it has none. */
  public List<String> primaryKey() {
    return primaryKey;
  }

  /** Its time column, when it has one. */
  public Optional<String> timeColumn() {
    return Optional.ofNullable(timeColumn);
  }

  /** How many rows share a key value, when keys were checked. */
  public OptionalLong duplicateKeys() {
    return duplicateKeys == null ? OptionalLong.empty() : OptionalLong.of(duplicateKeys);
  }

  Long duplicateKeysOrNull() {
    return duplicateKeys;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Table table
        && name.equals(table.name)
        && Objects.equals(sourceName, table.sourceName)
        && columns.equals(table.columns)
        && primaryKey.equals(table.primaryKey)
        && Objects.equals(timeColumn, table.timeColumn)
        && Objects.equals(duplicateKeys, table.duplicateKeys);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, sourceName, columns, primaryKey, timeColumn, duplicateKeys);
  }

  @Override
  public String toString() {
    return "Table(" + name + ", " + columns + ", primaryKey=" + primaryKey + ", timeColumn="
        + timeColumn + ")";
  }
}
