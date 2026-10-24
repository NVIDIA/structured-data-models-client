// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.DatabricksSource;
import com.nvidia.sdm.DuckDbSource;
import com.nvidia.sdm.PostgresSource;
import com.nvidia.sdm.SqliteSource;
import com.nvidia.sdm.Source;
import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The provider choice a run over a source names, and the provider answering its fetches, as the
 * Python host's {@code providers.for_source} gives them: the source's built-in SQL dialect, on its
 * connection, with the schema a {@link PostgresSource} was given, or the catalog and schema of a
 * {@link DatabricksSource}, as the namespace each table is read in. Not part of the API.
 */
public final class Providers {
  private Providers() {}

  /** The choice and provider of one source. */
  public record Chosen(Map<String, Object> choice, Host.Provider provider) {}

  public static Chosen of(Source source) {
    JdbcProvider.Driver driver;
    if (source instanceof DuckDbSource) {
      driver = JdbcProvider.Driver.DUCKDB;
    } else if (source instanceof SqliteSource) {
      driver = JdbcProvider.Driver.SQLITE;
    } else if (source instanceof PostgresSource) {
      driver = JdbcProvider.Driver.POSTGRES;
    } else if (source instanceof DatabricksSource) {
      driver = JdbcProvider.Driver.DATABRICKS;
    } else {
      throw new IllegalArgumentException("a source of no known kind");
    }
    Map<String, Object> choice = new LinkedHashMap<>();
    choice.put("provider", "sql");
    choice.put("dialect", driver.dialect());
    if (source instanceof PostgresSource postgres && postgres.schema().isPresent()) {
      choice.put("namespace", List.of(postgres.schema().get()));
    }
    // The catalog and the schema, each null where the connection's current one is meant, as
    // Python's DatabricksSource names them (namespace_parts); left out when neither is given.
    if (source instanceof DatabricksSource databricks
        && (databricks.catalog().isPresent() || databricks.schema().isPresent())) {
      choice.put(
          "namespace",
          Arrays.asList(databricks.catalog().orElse(null), databricks.schema().orElse(null)));
    }
    return new Chosen(choice, new JdbcProvider(source.connection(), driver));
  }
}
