// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

/** A relationship and its engine-recorded declared, inferred or unknown origin. */
public record RelationshipOrigin(Relationship relationship, String origin) {
  public RelationshipOrigin {
    java.util.Objects.requireNonNull(relationship, "relationship");
    if (!java.util.List.of("declared", "inferred", "unknown").contains(origin))
      throw new IllegalArgumentException("unknown relationship origin");
  }
}
