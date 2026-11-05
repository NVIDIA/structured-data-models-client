// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.Objects;

/**
 * A relationship between two tables, as Python's {@code Relationship}: a foreign key, {@code
 * sourceTable.sourceColumn}, holding values of the key {@code targetTable.targetColumn}.
 *
 * @param sourceTable the table holding the foreign key
 * @param sourceColumn the foreign key's column
 * @param targetTable the table it points at
 * @param targetColumn that table's key column
 */
public record Relationship(
    String sourceTable, String sourceColumn, String targetTable, String targetColumn) {
  /** Requires every part. */
  public Relationship {
    Objects.requireNonNull(sourceTable, "sourceTable");
    Objects.requireNonNull(sourceColumn, "sourceColumn");
    Objects.requireNonNull(targetTable, "targetTable");
    Objects.requireNonNull(targetColumn, "targetColumn");
  }
}
