// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.List;

/**
 * How one part's context examples were drawn from the source.
 *
 * @param draws the draws of the entity table: more than one where the first kept too few examples
 * @param rowsDrawn the rows those draws read, together
 * @param candidates the examples of the last draw labelled
 * @param kept the candidates with an answer the query's conditions keep, before the context is cut
 *     to the size asked for
 * @param asked the context examples asked for
 * @param stopped why drawing stopped: {@code full}, {@code source_exhausted} or {@code row_budget}
 * @param filteredAtSource whether any of the query's conditions went to the source with the draw
 */
public record ContextDraw(
    long draws,
    long rowsDrawn,
    long candidates,
    long kept,
    long asked,
    String stopped,
    boolean filteredAtSource) {
  public ContextDraw {
    if (draws < 0 || rowsDrawn < 0 || candidates < 0 || kept < 0 || asked < 0) {
      throw new IllegalArgumentException("invalid context draw count");
    }
    if (!List.of("full", "source_exhausted", "row_budget").contains(stopped)) {
      throw new IllegalArgumentException("invalid context draw stop");
    }
  }
}
