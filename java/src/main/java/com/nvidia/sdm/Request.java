// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.Results;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A request the core gathered for one query ({@link PredictiveQuery#materialize}), as Python's
 * {@code Request}: its payloads, one per call the endpoint is asked, and what the core resolved of
 * the query. It holds what {@link SdmClient#predict} hands back to the core, and is plain data:
 * the schema it was gathered from never changes under it.
 */
public final class Request {
  private final Schema schema;
  private final List<byte[]> payloads;
  private final String query;
  private final long version;
  private final List<Long> batchRows;
  private final Results.QueryFacts facts;
  private final List<String> inferenceAsked;

  Request(
      Schema schema,
      List<byte[]> payloads,
      String query,
      long version,
      List<Long> batchRows,
      Results.QueryFacts facts,
      List<String> inferenceAsked,
      String stateQuery) {
    this.schema = schema;
    this.payloads = List.copyOf(payloads);
    this.query = query;
    this.version = version;
    this.batchRows = batchRows == null ? null : List.copyOf(batchRows);
    this.facts = facts;
    this.inferenceAsked = List.copyOf(inferenceAsked);
  }

  /** The schema the request was gathered from. */
  public Schema schema() {
    return schema;
  }

  /** The query the request answers, as the core resolved it. */
  public String query() {
    return query;
  }

  /** Each call's payload, as the JSON text the core wrote, in the order the calls are made. */
  public List<String> payloads() {
    List<String> texts = new ArrayList<>();
    for (byte[] payload : payloads) {
      texts.add(new String(payload, StandardCharsets.UTF_8));
    }
    return texts;
  }

  /** The table whose entities the query is about. */
  public String entityTable() {
    return facts.entityTable();
  }

  /** The task the query is, as the core resolved it, when it resolved one. */
  public Optional<String> problemType() {
    return Optional.ofNullable(facts.problemType());
  }

  /** Whether the answer is a window rather than a standing value, as Python's attribute. */
  public boolean targetMovesWithTime() {
    return facts.windowed();
  }

  /**
   * Whether the request needs a model that reads related tables, as Python's attribute: its
   * context carries related tables, or its schema holds a relationship or more than one table.
   */
  public boolean isRelational() {
    if (schema.tables().size() > 1 || !schema.relationships().isEmpty()) {
      return true;
    }
    for (byte[] payload : payloads) {
      try {
        if (Json.read(payload) instanceof Map<?, ?> envelope
            && envelope.get("context") instanceof Map<?, ?> context
            && context.get("related_tables") instanceof Map<?, ?> related
            && !related.isEmpty()) {
          return true;
        }
      } catch (Json.JsonException e) {
        return false;
      }
    }
    return false;
  }

  /** How many rows the model learns the answer from, across every call. */
  public long contextRows() {
    return rows("context");
  }

  /** How many rows are asked about, across every call. */
  public long predictRows() {
    return rows("predict");
  }

  /** Whether the request is known to ask about no row at all. */
  public boolean nothingToScore() {
    for (byte[] payload : payloads) {
      Long rows = rows(payload, "predict");
      if (rows == null || rows != 0) {
        return false;
      }
    }
    return true;
  }

  private long rows(String half) {
    long total = 0;
    for (byte[] payload : payloads) {
      Long rows = rows(payload, half);
      total += rows == null ? 0 : rows;
    }
    return total;
  }

  /** How many instance rows one payload carries in {@code half}, or null when it cannot say. */
  private static Long rows(byte[] payload, String half) {
    try {
      Object read = Json.read(payload);
      if (read instanceof Map<?, ?> envelope
          && envelope.get(half) instanceof Map<?, ?> part
          && part.get("instance_table") instanceof Map<?, ?> table
          && table.get("rows") instanceof List<?> rows) {
        return (long) rows.size();
      }
    } catch (Json.JsonException e) {
      return null;
    }
    return null;
  }

  List<byte[]> payloadBytes() {
    return payloads;
  }

  long version() {
    return version;
  }

  List<Long> batchRows() {
    return batchRows;
  }

  List<String> inferenceAsked() {
    return inferenceAsked;
  }

  @Override
  public String toString() {
    return "Request(" + query + ", " + predictRows() + " rows to predict, " + contextRows()
        + " context rows in " + payloads.size() + " call(s))";
  }
}
