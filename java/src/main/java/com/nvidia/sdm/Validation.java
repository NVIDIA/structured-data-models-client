// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.List;
import java.util.Map;

/**
 * The core's answer to a query's validation: whether it accepts the query, every diagnostic it
 * gave, and the warnings the run gave, in order.
 *
 * @param accepted whether the query is valid against the schema and profile
 * @param diagnostics the diagnostics, errors and warnings, in order
 * @param warnings the run's warnings, in the order given
 */
public record Validation(
    boolean accepted, List<Diagnostic> diagnostics, List<Warning> warnings) {
  /** Copies both lists. */
  public Validation {
    diagnostics = List.copyOf(diagnostics);
    warnings = List.copyOf(warnings);
  }

  /** The messages of the error diagnostics: why the query was refused. */
  public List<String> reasons() {
    return diagnostics.stream().filter(Diagnostic::isError).map(Diagnostic::message).toList();
  }

  /**
   * One PQL diagnostic.
   *
   * @param code its code
   * @param message what it says
   * @param severity {@code error} or {@code warning}
   */
  public record Diagnostic(String code, String message, String severity) {
    /** Whether it refuses the query. */
    public boolean isError() {
      return "error".equals(severity);
    }
  }

  /**
   * One warning the core gave: its code, what it says, and its structured fields.
   *
   * @param code its code, such as {@code DATE_LITERAL_DAY_FIRST}
   * @param message what it says
   * @param fields its structured fields, by name
   */
  public record Warning(String code, String message, Map<String, Object> fields) {
    /** Copies the fields. */
    public Warning {
      fields =
          fields == null
              ? Map.of()
              : java.util.Collections.unmodifiableMap(new java.util.LinkedHashMap<>(fields));
    }
  }
}
