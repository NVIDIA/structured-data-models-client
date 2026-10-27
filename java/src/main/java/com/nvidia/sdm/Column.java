// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.Map;
import java.util.Objects;

/**
 * One column of a table, as Python's {@code Column}: its name, its data type on the client's
 * wire ({@code int64}, {@code float64}, {@code string}, {@code timestamp[us]}, ...) and what it
 * means ({@code ID}, {@code numerical}, {@code categorical}, {@code timestamp}, ...). Two columns
 * are equal when these three are. The column also keeps what it meant before the schema joined on
 * it and what the source said of it, which the schema hands back to the core whole and which no
 * caller reads (the structure design's fix A).
 */
public final class Column {
  private final String name;
  private final String dtype;
  private final String stype;
  private final String stypeBeforeKey;
  private final Map<String, Object> source;

  Column(
      String name, String dtype, String stype, String stypeBeforeKey, Map<String, Object> source) {
    this.name = Objects.requireNonNull(name, "name");
    this.dtype = dtype;
    this.stype = stype;
    this.stypeBeforeKey = stypeBeforeKey;
    this.source = source;
  }

  /** The column's name. */
  public String name() {
    return name;
  }

  /** Its data type, as the client's wire names it. */
  public String dtype() {
    return dtype;
  }

  /** What it means. */
  public String stype() {
    return stype;
  }

  String stypeBeforeKey() {
    return stypeBeforeKey;
  }

  Map<String, Object> source() {
    return source;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Column column
        && name.equals(column.name)
        && Objects.equals(dtype, column.dtype)
        && Objects.equals(stype, column.stype);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, dtype, stype);
  }

  @Override
  public String toString() {
    return "Column(" + name + ", " + dtype + ", " + stype + ")";
  }
}
