// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.util.List;
import java.util.ArrayList;

/** A detached table descriptor without row data. */
public record TableMetadata(String name, List<ColumnMetadata> columns, List<String> primaryKey, String timeColumn) {
  public TableMetadata { columns = List.copyOf(columns); primaryKey = List.copyOf(primaryKey); }
  public String toText() {
    List<String> lines = new ArrayList<>(); lines.add(name);
    for (ColumnMetadata column : columns) {
      List<String> flags = new ArrayList<>();
      if (primaryKey.contains(column.name())) flags.add("primary key");
      if (column.name().equals(timeColumn)) flags.add("time column");
      lines.add("  " + column.name() + ": " + column.dtype() + ", " + column.stype()
          + (flags.isEmpty() ? "" : " (" + String.join(", ", flags) + ")"));
    }
    return String.join("\n", lines);
  }
}
