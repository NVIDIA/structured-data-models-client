// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The core's results, as this host reads them (the structure design's fix D, rule 10): every
 * result a run finishes with (the contract's {@code OperationResult}) decoded once into the
 * records here, with the same fields as the Python host's ({@code sdm_client._rust.results}); a
 * result of another operation or shape than the contract's is the core's defect, {@code
 * INTERNAL_ERROR}, never read as it is. Not part of the API.
 */
public final class Results {
  private Results() {}

  /** What the core resolved of a query (the contract's {@code QueryFacts}). */
  public record QueryFacts(
      String entityTable, String problemType, boolean windowed, long numForecasts, String text) {}

  /** One diagnostic of a query: its code, message and severity. */
  public record Diagnostic(String code, String message, String severity) {}

  /** {@code validate_query}'s result. */
  public record Validated(boolean accepted, List<Diagnostic> diagnostics, QueryFacts query) {}

  /** {@code describe_source}'s result: the attachment of the {@code SourceDescription}. */
  public record Described(
      int description, List<ViewNote> notes, Integer frameRows, Integer frameDecimals) {
    public Described(int description, List<ViewNote> notes) {
      this(description, notes, null, null);
    }
    public Described(int description, List<ViewNote> notes, Integer frameRows) {
      this(description, notes, frameRows, null);
    }
  }

  /** A view conversion's core-owned sentence, and which table or time it describes. */
  public record ViewNote(
      String sentence, Map<String, Object> fields, String table, String timeColumn) {}

  /** Shared internal frame inference data, without a Java public constructor. */
  public record InferredFrames(int value) {}

  /** {@code edit_schema}'s result: the attachment of the schema edited. */
  public record Edited(int schema, Map<String, Object> declared) {}

  /** A later call of a request asked in several: its payload's attachment and rows. */
  public record Batch(int payload, long rows) {}

  /** The request state {@code materialize} wrote, its attachments by id. */
  public record State(
      long version, int payload, int query, List<Batch> batches, boolean inParts) {}

  /** {@code materialize}'s result. */
  public record Gathered(State state, QueryFacts query) {}

  /** One internal direct preparation value, without a Java feature surface. */
  public record Prepared(int value) {}

  /** Owned response data for the internal native builtin conversion boundary. */
  public record DecodedDirectResponse(int value) {}

  /** {@code predict}'s result. */
  public record Predicted(
      int result,
      String model,
      String requestId,
      int metadata,
      List<String> outputs,
      List<Long> unscoredCandidates) {}

  /** {@code result}, the result of {@code op}, decoded. */
  public static Object expect(Map<String, Object> result, String op) {
    try {
      if (result == null || !op.equals(result.get("op"))) {
        throw new IllegalArgumentException("another operation than the one started");
      }
      return switch (op) {
        case "validate_query" -> validated(result);
        case "describe_source" -> described(result);
        case "inferred_frames" -> new InferredFrames(count(result, "value"));
        case "edit_schema" -> new Edited(count(result, "schema"), map(result.get("declared")));
        case "materialize" -> new Gathered(state(map(result.get("state"))), facts(result.get("query")));
        case "predict" -> predicted(result);
        case "prepare_request" -> new Prepared(count(result, "value"));
        case "decoded_direct_response" -> new DecodedDirectResponse(count(result, "value"));
        default -> throw new IllegalArgumentException("an operation this host does not know");
      };
    } catch (IllegalArgumentException | ClassCastException | NullPointerException e) {
      throw new StructuredDataException(
          "the core answered a result this host does not read: " + e.getMessage(),
          "INTERNAL_ERROR",
          Map.of(),
          null);
    }
  }

  private static Described described(Map<String, Object> read) {
    List<ViewNote> notes = new ArrayList<>();
    if (read.get("notes") != null) {
      for (Object item : list(read.get("notes"))) {
        Map<String, Object> note = map(item);
        Map<String, Object> fields = note.get("fields") == null ? Map.of() : map(note.get("fields"));
        notes.add(new ViewNote(text(note, "sentence"), Map.copyOf(fields),
            (String) note.get("table"), (String) note.get("time_column")));
      }
    }
    return new Described(count(read, "description"), List.copyOf(notes),
        read.get("frame_rows") == null ? null : count(read, "frame_rows"),
        read.get("frame_decimals") == null ? null : count(read, "frame_decimals"));
  }

  private static Validated validated(Map<String, Object> read) {
    List<Diagnostic> diagnostics = new ArrayList<>();
    for (Object item : list(read.get("diagnostics"))) {
      Map<String, Object> each = map(item);
      diagnostics.add(
          new Diagnostic(
              text(each, "code"), text(each, "message"), text(each, "severity")));
    }
    return new Validated(
        Boolean.TRUE.equals(read.get("accepted")), List.copyOf(diagnostics), facts(read.get("query")));
  }

  private static QueryFacts facts(Object value) {
    if (value == null) {
      return null;
    }
    Map<String, Object> read = map(value);
    return new QueryFacts(
        text(read, "entity_table"),
        (String) read.get("problem_type"),
        Boolean.TRUE.equals(read.get("windowed")),
        ((Number) read.get("num_forecasts")).longValue(),
        (String) read.get("text"));
  }

  private static State state(Map<String, Object> read) {
    List<Batch> batches = null;
    if (read.get("batches") != null) {
      batches = new ArrayList<>();
      for (Object item : list(read.get("batches"))) {
        Map<String, Object> each = map(item);
        batches.add(new Batch(count(each, "payload"), ((Number) each.get("rows")).longValue()));
      }
      batches = List.copyOf(batches);
    }
    boolean inParts = read.get("forecast") != null || read.get("ranking") != null;
    return new State(
        ((Number) read.get("version")).longValue(),
        count(read, "payload"),
        count(read, "query"),
        batches,
        inParts);
  }

  private static Predicted predicted(Map<String, Object> read) {
    List<String> outputs = new ArrayList<>();
    for (Object item : list(read.get("outputs"))) {
      outputs.add((String) item);
    }
    List<Long> unscored = new ArrayList<>();
    if (read.get("unscored_candidates") != null) {
      for (Object item : list(read.get("unscored_candidates"))) {
        unscored.add(((Number) item).longValue());
      }
    }
    return new Predicted(
        count(read, "result"),
        text(read, "model"),
        text(read, "request_id"),
        count(read, "metadata"),
        List.copyOf(outputs),
        List.copyOf(unscored));
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> map(Object value) {
    return value == null ? null : (Map<String, Object>) value;
  }

  private static List<?> list(Object value) {
    return (List<?>) value;
  }

  private static int count(Map<String, Object> read, String field) {
    return ((Number) read.get(field)).intValue();
  }

  private static String text(Map<String, Object> read, String field) {
    Object value = read.get(field);
    if (!(value instanceof String text)) {
      throw new IllegalArgumentException(field + " is not text");
    }
    return text;
  }
}
