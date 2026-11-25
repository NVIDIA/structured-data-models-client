// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/** Immutable signed scores, optionally normalized by aggregate absolute mass. */
public record FeatureImportance(List<FeatureImportanceRow> rows, boolean normalized) {
  public FeatureImportance { rows = List.copyOf(rows); }
  public String toText() { return rows.stream().map(v -> v.table() + "." + v.column() + ": "
      + String.format(Locale.ROOT, "%.6g", v.score())).collect(Collectors.joining("\n")); }
}
