// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import java.time.Duration;
import java.time.Instant;

/** Input column names and fixed-window options for history preparation. */
public final class TimeSeriesOptions {
  final String history, timestamps, entity;
  final Duration interval;
  final Instant anchor;
  final int forecasts;

  private TimeSeriesOptions(Builder b) {
    history = b.history;
    timestamps = b.timestamps;
    entity = b.entity;
    interval = b.interval;
    anchor = b.anchor;
    forecasts = b.forecasts;
  }

  public static Builder builder(String timeseriesColumn) {
    return new Builder(timeseriesColumn);
  }

  public static final class Builder {
    private final String history;
    private String timestamps, entity;
    private Duration interval;
    private Instant anchor;
    private int forecasts = 1;

    private Builder(String history) {
      this.history = history;
    }

    public Builder timestampsColumn(String value) {
      timestamps = value;
      return this;
    }

    public Builder entityColumn(String value) {
      entity = value;
      return this;
    }

    public Builder timeDelta(Duration value) {
      interval = value;
      return this;
    }

    public Builder anchorTime(Instant value) {
      anchor = value;
      return this;
    }

    public Builder numTimeframes(int value) {
      forecasts = value;
      return this;
    }

    public TimeSeriesOptions build() {
      return new TimeSeriesOptions(this);
    }
  }
}
