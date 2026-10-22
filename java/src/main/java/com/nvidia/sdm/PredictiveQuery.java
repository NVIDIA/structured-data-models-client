// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Contract;
import com.nvidia.sdm.internal.Host;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A question expressed in PQL, as Python's {@code PredictiveQuery} is (design choice 10): {@link
 * #parse} keeps the text, refusing only an empty one, and the query is resolved and validated
 * against a schema by the core when it is asked ({@link #validate}).
 *
 * <pre>{@code
 * PredictiveQuery query = PredictiveQuery.parse(
 *     "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id");
 * Validation validation = query.validate(schema, Profile.RELATIONAL);
 * if (!validation.accepted()) {
 *   System.out.println(validation.reasons());
 * }
 * }</pre>
 */
public final class PredictiveQuery {
  private final String text;

  private PredictiveQuery(String text) {
    this.text = text;
  }

  /**
   * The query {@code text}, kept as given; nothing is resolved until it is validated, as Python's
   * constructor keeps it.
   *
   * @throws StructuredDataException {@code INVALID_REQUEST} when the text is null or blank
   */
  public static PredictiveQuery parse(String text) {
    if (text == null || blank(text)) {
      throw new StructuredDataException(
          "query must be a non-empty string", "INVALID_REQUEST", Map.of(), null);
    }
    return new PredictiveQuery(text);
  }

  /**
   * Whether {@code text} is empty once stripped as Python's {@code str.strip()} strips: of every
   * character {@code str.isspace()} holds, Java's whitespace and space separators, the
   * no-break spaces among them, and NEL (U+0085).
   */
  private static boolean blank(String text) {
    return text.codePoints()
        .allMatch(c -> Character.isWhitespace(c) || Character.isSpaceChar(c) || c == 0x85);
  }

  /** The query's text, as given. */
  public String text() {
    return text;
  }

  /**
   * The core's validation of this query against {@code schema} for {@code profile}. A query the
   * core refuses is answered, not thrown: {@link Validation#accepted} is false and {@link
   * Validation#reasons} says why. The query text and the schema travel as attachments, so neither
   * a long query nor a wide schema meets the control cap (decision H5).
   *
   * @throws StructuredDataException the core's own error (a value JSON cannot hold, such as text
   *     with an unpaired surrogate, is {@code INVALID_REQUEST}); {@code UNSUPPORTED_REQUEST} when
   *     this JAR holds no native library for the platform
   */
  public Validation validate(QuerySchema schema, Profile profile) {
    java.util.Objects.requireNonNull(schema, "schema");
    java.util.Objects.requireNonNull(profile, "profile");
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "validate_query");
    operation.put("query", 1);
    operation.put("schema", 2);
    operation.put("profile", Contract.profile(profile));
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(text));
    attachments.put(2, Host.attachment(Contract.querySchema(schema)));
    try (Host.Finished finished = Host.run(operation, attachments)) {
      return validation(finished.result(), finished.warnings());
    }
  }

  @SuppressWarnings("unchecked")
  private static Validation validation(
      Map<String, Object> result, List<Map<String, Object>> warnings) {
    List<Validation.Diagnostic> diagnostics = new ArrayList<>();
    for (Object item : (List<Object>) result.get("diagnostics")) {
      Map<String, Object> diagnostic = (Map<String, Object>) item;
      diagnostics.add(
          new Validation.Diagnostic(
              (String) diagnostic.get("code"),
              (String) diagnostic.get("message"),
              (String) diagnostic.get("severity")));
    }
    List<Validation.Warning> given = new ArrayList<>();
    for (Map<String, Object> warning : warnings) {
      Map<String, Object> fields = new LinkedHashMap<>();
      if (warning.get("fields") instanceof Map<?, ?> map) {
        map.forEach((name, value) -> fields.put((String) name, com.nvidia.sdm.internal.Messages.detail(value)));
      }
      given.add(
          new Validation.Warning(
              (String) warning.get("code"), (String) warning.get("message"), fields));
    }
    return new Validation(Boolean.TRUE.equals(result.get("accepted")), diagnostics, given);
  }
}
