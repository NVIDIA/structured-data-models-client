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
 * }</pre>
 *
 * <p>Each request the core sends becomes one or more prototype {@link ai.kumo.data.Query}s with
 * only these parts: a projection of plain columns, {@code COUNT()} or {@code MIN}/{@code MAX} of
 * {@code TIME_MICROS(column)}; a predicate of {@code AND}, {@code OR}, {@code IN}, {@code EQ},
 * {@code IS_NOT null}, and {@code TIME_MICROS(column)} compared with a micros literal; and a
 * limit. No joins, scalar subqueries or text matching. Keys are batched into one {@code IN} per
 * request (at most {@link BridgeOptions#maxKeysPerQuery} keys each), and per-key windows, order and
 * limits are applied in Java.
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
