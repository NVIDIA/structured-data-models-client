// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.List;
import java.util.Objects;

/**
 * The schema a query is validated against, as plain data: the tables, each with its columns' PQL
 * data and semantic types, its primary key and time columns, and the groups of linked columns
 * (a primary key and the foreign keys pointing at it). The contract's {@code QuerySchema}.
 *
 * @param tables the tables, in order
 * @param links groups of linked columns
 */
public record QuerySchema(List<Table> tables, List<List<ColumnRef>> links) {
  /** Copies both lists, so the schema cannot change under a call. */
  public QuerySchema {
    tables = List.copyOf(tables);
    links = links.stream().map(List::copyOf).toList();
  }

  /**
   * One table.
   *
   * @param name the table's name
   * @param columns its columns, in order
   * @param primaryKey its primary key column, or null
   * @param timeColumn its time column, or null
   * @param endTimeColumn its end time column, or null
   */
  public record Table(
      String name,
      List<Column> columns,
      String primaryKey,
      String timeColumn,
      String endTimeColumn) {
    /** Copies the columns. */
    public Table {
      Objects.requireNonNull(name, "name");
      columns = List.copyOf(columns);
    }
  }

  /**
   * One column: its PQL data type ({@code int}, {@code float}, {@code string}, {@code time}, ...)
   * and semantic type ({@code ID}, {@code numerical}, {@code categorical}, {@code timestamp},
   * ...), as the contract spells them.
   *
   * @param name the column's name
   * @param dtype its PQL data type
   * @param stype its semantic type
   */
  public record Column(String name, String dtype, String stype) {
    /** Requires every part. */
    public Column {
      Objects.requireNonNull(name, "name");
      Objects.requireNonNull(dtype, "dtype");
      Objects.requireNonNull(stype, "stype");
    }
  }

  /**
   * A column of a table.
   *
   * @param table the table's name
   * @param column the column's name
   */
  public record ColumnRef(String table, String column) {
    /** Requires both parts. */
    public ColumnRef {
      Objects.requireNonNull(table, "table");
      Objects.requireNonNull(column, "column");
    }
  }
}
