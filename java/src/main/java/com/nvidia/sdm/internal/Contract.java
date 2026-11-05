// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The API's plain data as the contract's JSON values, written in one place, so what a call sends
 * is what the contract runner records it sent. Not part of the API.
 */
public final class Contract {
  private Contract() {}

  /** {@code schema} as the contract's {@code QuerySchema}. */
  public static Map<String, Object> querySchema(QuerySchema schema) {
    List<Object> tables = new ArrayList<>();
    for (QuerySchema.Table table : schema.tables()) {
      List<Object> columns = new ArrayList<>();
      for (QuerySchema.Column column : table.columns()) {
        Map<String, Object> written = new LinkedHashMap<>();
        written.put("name", column.name());
        written.put("dtype", column.dtype());
        written.put("stype", column.stype());
        columns.add(written);
      }
      Map<String, Object> written = new LinkedHashMap<>();
      written.put("name", table.name());
      written.put("columns", columns);
      written.put("primary_key", table.primaryKey());
      written.put("time_column", table.timeColumn());
      written.put("end_time_column", table.endTimeColumn());
      tables.add(written);
    }
    List<Object> links = new ArrayList<>();
    for (List<QuerySchema.ColumnRef> group : schema.links()) {
      List<Object> written = new ArrayList<>();
      for (QuerySchema.ColumnRef ref : group) {
        Map<String, Object> column = new LinkedHashMap<>();
        column.put("table", ref.table());
        column.put("column", ref.column());
        written.add(column);
      }
      links.add(written);
    }
    Map<String, Object> written = new LinkedHashMap<>();
    written.put("tables", tables);
    written.put("links", links);
    return written;
  }

  /** {@code profile} as the contract's {@code ProfileSpec}: one of the named five, no model. */
  public static Map<String, Object> profile(Profile profile) {
    Map<String, Object> spec = new LinkedHashMap<>();
    spec.put("name", profile.name());
    spec.put("model", null);
    return spec;
  }
}
