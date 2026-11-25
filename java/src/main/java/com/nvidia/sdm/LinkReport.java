// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.util.List;
import java.util.stream.Collectors;

/** Relationships in existing descriptor order. */
public record LinkReport(List<LinkMetadata> links) {
  public LinkReport { links = List.copyOf(links); }
  public String toText() { return links.stream().map(v -> v.sourceTable() + "." + v.sourceColumn()
      + " -> " + v.targetTable() + "." + v.targetColumn()).collect(Collectors.joining("\n")); }
}
