// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.bridge.kumo;

/**
 * How the bridge reads through a prototype {@code DataAccess}.
 *
 * @param maxKeysPerQuery the most keys one {@code IN} carries; more are read in several queries
 * @param fallbackMaxRows the most rows the core reads from one table to answer, in memory, a
 *     request the bridge does not answer itself (sampling entities, key statistics)
 * @param fallbackMaxBytes the same limit in bytes
 * @param purposePrefix the prefix of every query's {@code purpose}, for the caller's logs
 */
public record BridgeOptions(
    int maxKeysPerQuery, long fallbackMaxRows, long fallbackMaxBytes, String purposePrefix) {
  public BridgeOptions {
    if (maxKeysPerQuery < 1) {
      throw new IllegalArgumentException("maxKeysPerQuery must be at least 1");
    }
    if (fallbackMaxRows < 1 || fallbackMaxBytes < 1) {
      throw new IllegalArgumentException("fallback limits must be positive");
    }
    if (purposePrefix == null) {
      purposePrefix = "";
    }
  }

  /** 1,000 keys per query; fallback of up to 5,000,000 rows and 2 GB; purposes prefixed {@code sdm:}. */
  public static BridgeOptions defaults() {
    return new BridgeOptions(1_000, 5_000_000L, 2L << 30, "sdm:");
  }

  public BridgeOptions withMaxKeysPerQuery(int max) {
    return new BridgeOptions(max, fallbackMaxRows, fallbackMaxBytes, purposePrefix);
  }

  public BridgeOptions withFallback(long maxRows, long maxBytes) {
    return new BridgeOptions(maxKeysPerQuery, maxRows, maxBytes, purposePrefix);
  }

  public BridgeOptions withPurposePrefix(String prefix) {
    return new BridgeOptions(maxKeysPerQuery, fallbackMaxRows, fallbackMaxBytes, prefix);
  }
}
