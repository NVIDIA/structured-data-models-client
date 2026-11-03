// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.nio.file.Path;
import java.util.Objects;

/** Cache and offline settings for {@link Schema#fromRelbench(String, RelBenchOptions)}. */
public final class RelBenchOptions {
  private final Path cacheDirectory;
  private final boolean offline;

  private RelBenchOptions(Builder builder) {
    cacheDirectory = builder.cacheDirectory;
    offline = builder.offline;
  }

  public static Builder builder() { return new Builder(); }
  public Path cacheDirectory() { return cacheDirectory; }
  public boolean offline() { return offline; }

  public static final class Builder {
    private Path cacheDirectory = Path.of(System.getProperty("user.home"), ".cache", "sdm", "relbench");
    private boolean offline;
    private Builder() {}
    public Builder cacheDirectory(Path path) {
      cacheDirectory = Objects.requireNonNull(path, "cacheDirectory"); return this;
    }
    public Builder offline(boolean value) { offline = value; return this; }
    public RelBenchOptions build() { return new RelBenchOptions(this); }
  }
}
