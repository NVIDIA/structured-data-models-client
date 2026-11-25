// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.util.List;
import java.util.stream.Collectors;

/** Schema tables in existing descriptor order. */
public record MetadataReport(List<TableMetadata> tables) {
  public MetadataReport { tables = List.copyOf(tables); }
  public String toText() { return tables.stream().map(TableMetadata::toText).collect(Collectors.joining("\n")); }
}
