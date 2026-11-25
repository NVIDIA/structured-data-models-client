// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

/** Exclusive counts of rows excluded from one prepared graph table. */
public record TableSanitizationReport(String table, long inputRows, long outputRows,
    long nullPrimaryKeyRows, long duplicatePrimaryKeyRows, long nullTimeRows) {
  public TableSanitizationReport {
    java.util.Objects.requireNonNull(table, "table");
    if (inputRows < 0 || outputRows < 0 || nullPrimaryKeyRows < 0 || duplicatePrimaryKeyRows < 0 || nullTimeRows < 0
        || outputRows > inputRows || nullPrimaryKeyRows > inputRows - outputRows
        || duplicatePrimaryKeyRows > inputRows - outputRows - nullPrimaryKeyRows
        || nullTimeRows != inputRows - outputRows - nullPrimaryKeyRows - duplicatePrimaryKeyRows) {
      throw new IllegalArgumentException("invalid sanitization row counts");
    }
  }
}
