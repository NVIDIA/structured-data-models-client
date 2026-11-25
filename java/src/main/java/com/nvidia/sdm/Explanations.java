// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

/** Model gradients and descriptive context associations, kept separate. */
public final class Explanations {
  private final GradientExplanations gradients;
  private final ColumnExplanations columns;

  /** An unexplained result: empty gradients and no column analysis. */
  public Explanations() { this(null, null); }

  /** Public presentation container; construction does not invoke the engine. */
  public Explanations(GradientExplanations gradients, ColumnExplanations columns) {
    this.gradients = gradients == null ? new GradientExplanations(null) : gradients;
    this.columns = columns;
  }

  /** Per-answer model gradients, empty when none were returned. */
  public GradientExplanations gradients() { return gradients; }

  /** Context associations, or null when not requested or unavailable. */
  public ColumnExplanations columns() { return columns; }
}
