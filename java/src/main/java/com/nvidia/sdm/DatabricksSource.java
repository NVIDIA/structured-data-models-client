// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Providers;
import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import java.sql.Connection;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Tables in a Databricks catalog and schema, read through a SQL warehouse with the Databricks JDBC
 * driver ({@code com.databricks:databricks-jdbc}), as Python's {@code DatabricksSource} reads them
 * through the Databricks SQL connector: the catalog and the schema it is given, each defaulting to
 * the connection's current one. The core writes every statement; the library runs them on the
 * connection and never closes it.
 *
 * <p>Authentication is the connection's: the caller opens it with a token of its own.
 */
public final class DatabricksSource extends BuiltInSource {
  private final Connection connection;
  private final String catalog;
  private final String schema;
  private final String semanticModel;
  private final boolean trustExpressions;

  private DatabricksSource(Connection connection, String catalog, String schema) {
    this(connection, catalog, schema, null, false);
  }

  private DatabricksSource(Connection connection, String catalog, String schema,
      String semanticModel, boolean trustExpressions) {
    this.connection = Objects.requireNonNull(connection, "connection");
    this.catalog = catalog;
    this.schema = schema;
    this.semanticModel = semanticModel;
    this.trustExpressions = trustExpressions;
  }

  /** The tables of {@code connection}'s current catalog and schema. */
  public static DatabricksSource of(Connection connection) {
    return new DatabricksSource(connection, null, null);
  }

  /**
   * The tables of {@code catalog}.{@code schema}, read through {@code connection}; either may be
   * null for the connection's current one.
   */
  public static DatabricksSource of(Connection connection, String catalog, String schema) {
    return new DatabricksSource(connection, catalog, schema);
  }

  /**
   * This source, reading the metric view {@code name} in its namespace, whose expressions calling
   * functions other than pure built-ins are left out.
   */
  public DatabricksSource semanticModel(String name) {
    return semanticModel(name, false);
  }

  /**
   * This source, reading the metric view {@code name} in its namespace; {@code trustExpressions}
   * keeps the expressions that call functions other than pure built-ins, which run as SQL under
   * the connection's credentials.
   */
  public DatabricksSource semanticModel(String name, boolean trustExpressions) {
    return new DatabricksSource(connection, catalog, schema,
        Objects.requireNonNull(name, "name"), trustExpressions);
  }

  /** The metric view, when one was given. */
  public Optional<String> semanticModel() {
    return Optional.ofNullable(semanticModel);
  }

  /** Whether the caller allows the metric view's expressions without validation. */
  public boolean trustExpressions() {
    return trustExpressions;
  }

  /** The caller-owned connection, kept open while reading source data. */
  public Connection connection() {
    return connection;
  }

  /** The catalog the tables are read in, when one was given. */
  public Optional<String> catalog() {
    return Optional.ofNullable(catalog);
  }

  /** The schema the tables are read in, when one was given. */
  public Optional<String> schema() {
    return Optional.ofNullable(schema);
  }

  @Override
  Map<String, Object> sqlChoice() {
    return withMicroseconds(
        Providers.sqlChoice("databricks", semanticModel, trustExpressions, catalog, schema));
  }

  // Native getters can reinterpret timestamps through session or JVM zones. The core selects the
  // source meaning and a lossless integer projection.
  private static Map<String, Object> withMicroseconds(Map<String, Object> choice) {
    choice.put("timestamp_encoding", "microseconds");
    return choice;
  }

  @Override
  Host.Provider driver() {
    return new JdbcProvider(connection, JdbcProvider.Driver.DATABRICKS);
  }

  @Override
  public String toString() {
    String source = "DatabricksSource(" + connection + ", " + catalog + ", " + schema + ")";
    return semanticModel == null ? source
        : "SemanticModel(" + semanticModel + " in " + source
            + ", trustExpressions=" + trustExpressions + ")";
  }
}
