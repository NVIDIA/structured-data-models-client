// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.bridge.kumo;

import ai.kumo.data.DataAccess;
import ai.kumo.graph.Graph;
import ai.kumo.graph.Table;
import com.nvidia.sdm.FromSourceOptions;
import com.nvidia.sdm.ProviderSource;
import com.nvidia.sdm.Relationship;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Runs the SDM client over code written for the prototype Java SDK ({@code ai.kumo}): the
 * prototype's {@link Graph} describes the tables, and its {@link DataAccess} reads every row.
 *
 * <pre>{@code
 * ProviderSource source = KumoBridge.source(graph, dataAccess);
 * Schema schema = Schema.fromSource(source, KumoBridge.options(graph));
 * Request request = dataAccess.snapshot(() -> query.materialize(schema, options));
 * }</pre>
 *
 * <p>The bridge answers each of the core's requests on its own, so it cannot open one snapshot
 * for a whole prediction. Wrap the call that reads (here {@code materialize}) in {@link
 * DataAccess#snapshot}, as the prototype wraps its own prepare, so a {@code DataAccess} that reads
 * one consistent snapshot, or caches per snapshot, does so across all of the bridge's queries.
 *
 * <p>Each request the core sends becomes one or more prototype {@link ai.kumo.data.Query}s, each
 * in a shape the prototype itself sends, with no alias and no order:
 *
 * <ul>
 *   <li>rows: {@code Query.scan}'s shape (an empty projection, so every column) with a limit or
 *       none;
 *   <li>aggregates: one per query, {@code COUNT()}, or {@code MIN} or {@code MAX} of {@code
 *       TIME_MICROS(column)};
 *   <li>predicates: none, or an {@code AND} of terms: {@code IN}, an {@code OR} of {@code AND}s of
 *       {@code EQ} (keys of several columns), {@code IS_NOT null}, and {@code TIME_MICROS(column)}
 *       {@code GT} or {@code LE} a micros literal.
 * </ul>
 *
 * <p>No joins, scalar subqueries or text matching. Keys are batched into one {@code IN} per
 * request (at most {@link BridgeOptions#maxKeysPerQuery} keys each), and per-key windows, order and
 * limits are applied in Java. A column type the bridge does not know (one a modified prototype
 * adds) is read as text.
 *
 * <p>The prototype jar is not bundled: the caller already has it on the class path.
 */
public final class KumoBridge {
  private KumoBridge() {}

  /** A source reading through {@code access}, with the default options. */
  public static ProviderSource source(Graph graph, DataAccess access) {
    return source(graph, access, BridgeOptions.defaults());
  }

  /** A source reading through {@code access}. The bridge never closes {@code access}. */
  public static ProviderSource source(Graph graph, DataAccess access, BridgeOptions options) {
    Objects.requireNonNull(graph, "graph");
    Objects.requireNonNull(access, "access");
    Objects.requireNonNull(options, "options");
    return ProviderSource.of(new QueryProvider(graph, access, options));
  }

  /**
   * The schema options {@code graph} declares: every table, each one's primary key (or none) and
   * time column (or none), and every relationship. Relationships of several columns are left out:
   * the SDM client links tables by one column.
   */
  public static FromSourceOptions options(Graph graph) {
    Objects.requireNonNull(graph, "graph");
    FromSourceOptions.Builder builder = FromSourceOptions.builder();
    for (Table table : graph.tables()) {
      builder.table(table.name());
      if (table.primaryKey().isEmpty()) {
        builder.noPrimaryKey(table.name());
      } else {
        builder.primaryKey(table.name(), table.primaryKey().toArray(String[]::new));
      }
      if (table.timeColumn() == null) {
        builder.noTimeColumn(table.name());
      } else {
        builder.timeColumn(table.name(), table.timeColumn());
      }
    }
    List<Relationship> links = new ArrayList<>();
    for (Graph.Relationship link : graph.relationships()) {
      if (link.sourceColumns().size() == 1 && link.targetColumns().size() == 1) {
        links.add(
            new Relationship(
                link.sourceTable(),
                link.sourceColumns().get(0),
                link.targetTable(),
                link.targetColumns().get(0)));
      }
    }
    builder.relationships(links);
    return builder.build();
  }
}
