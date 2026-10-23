// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The core's validation of a query against a schema, for a profile ({@code validate_query}): what
 * {@code PredictiveQuery.materialize} asks first, as Python's materialize resolves a query on the
 * core route, and the contract runner's S0 leg. Not part of the API: Python has no public
 * validation either (T3.11's review).
 */
public final class QueryValidation {
  private QueryValidation() {}

  /** The core's validation of {@code text} against {@code schema} for {@code profile}. */
  public static Validation validate(String text, QuerySchema schema, Profile profile) {
    return validate(text, Contract.querySchema(Objects.requireNonNull(schema, "schema")), profile);
  }

  /**
   * The core's validation of {@code text} against {@code schema}, the contract's {@code
   * QuerySchema} as JSON, for {@code profile}. A query the core refuses is answered, not thrown:
   * {@link Validation#accepted} is false and {@link Validation#reasons} says why. The query text
   * and the schema travel as attachments, so neither meets the control cap (decision H5).
   */
  public static Validation validate(String text, Map<String, Object> schema, Profile profile) {
    Objects.requireNonNull(text, "text");
    Objects.requireNonNull(profile, "profile");
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "validate_query");
    operation.put("query", 1);
    operation.put("schema", 2);
    operation.put("profile", Contract.profile(profile));
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(text));
    attachments.put(2, Host.attachment(schema));
    try (Host.Finished finished = Host.run(operation, attachments)) {
      Results.Validated validated =
          (Results.Validated) Results.expect(finished.result(), "validate_query");
      List<Validation.Diagnostic> diagnostics = new ArrayList<>();
      for (Results.Diagnostic each : validated.diagnostics()) {
        diagnostics.add(new Validation.Diagnostic(each.code(), each.message(), each.severity()));
      }
      List<Validation.Warning> given = new ArrayList<>();
      for (Map<String, Object> warning : finished.warnings()) {
        Map<String, Object> fields = new LinkedHashMap<>();
        if (warning.get("fields") instanceof Map<?, ?> map) {
          map.forEach((name, value) -> fields.put((String) name, Messages.detail(value)));
        }
        given.add(
            new Validation.Warning(
                (String) warning.get("code"), (String) warning.get("message"), fields));
      }
      Results.QueryFacts query = validated.query();
      return new Validation(
          validated.accepted(),
          diagnostics,
          given,
          validated.accepted() && query != null ? query.text() : null);
    }
  }
}
