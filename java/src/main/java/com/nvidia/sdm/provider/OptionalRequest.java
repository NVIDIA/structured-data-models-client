// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.provider;

/** The requests a provider may choose to answer itself ({@link Capabilities#optional}). */
public enum OptionalRequest {
  PREVIEW("preview"),
  COUNT_ROWS("count_rows"),
  KEY_COUNTS("key_counts"),
  TIME_BOUNDS("time_bounds"),
  COLUMN_RANGES("column_ranges"),
  REPEATED_KEY("repeated_key"),
  SAMPLE_ENTITIES("sample_entities"),
  NEIGHBORS("neighbors"),
  LOOKUP("lookup");

  private final String wire;

  OptionalRequest(String wire) {
    this.wire = wire;
  }

  /** The request's name in the core's contract. */
  public String wireName() {
    return wire;
  }
}
