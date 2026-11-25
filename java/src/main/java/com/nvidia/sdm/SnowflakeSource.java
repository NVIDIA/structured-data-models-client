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
 * Tables in a Snowflake database and schema, read through a warehouse with the Snowflake JDBC
 * driver ({@code net.snowflake:snowflake-jdbc}), as Python's {@code SnowflakeSource} reads them
 * through the Snowflake connector: the database and the schema it is given, each defaulting to the
 * connection's current one. The core plans catalog and sampling statements; the Java host
 * executes them and reads JDBC conversion metadata without closing the connection.
 *
 * <p>The caller opens and authenticates the connection.
 */
public final class SnowflakeSource extends BuiltInSource {
  private final Connection connection;
  private final String database;
  private final String schema;
  private final String semanticModel;
  private final boolean trustExpressions;

  private SnowflakeSource(Connection connection, String database, String schema) {
    this(connection, database, schema, null, false);
  }

  private SnowflakeSource(
      Connection connection,
      String database,
      String schema,
      String semanticModel,
      boolean trustExpressions) {
    this.connection = Objects.requireNonNull(connection, "connection");
    this.database = database;
    this.schema = schema;
    this.semanticModel = semanticModel;
    this.trustExpressions = trustExpressions;
  }

  /** The tables of {@code connection}'s current database and schema. */
  public static SnowflakeSource of(Connection connection) {
    return new SnowflakeSource(connection, null, null);
  }

  /**
   * The tables of {@code database}.{@code schema}, read through {@code connection}; either may be
   * null for the connection's current one.
   */
  public static SnowflakeSource of(Connection connection, String database, String schema) {
    return new SnowflakeSource(connection, database, schema);
  }

  /**
   * This source, reading the semantic view {@code name} in its namespace, whose expressions calling
   * functions other than pure built-ins are left out.
   */
  public SnowflakeSource semanticModel(String name) {
    return semanticModel(name, false);
  }

  /**
   * This source, reading the semantic view {@code name} in its namespace; {@code trustExpressions}
   * keeps the expressions that call functions other than pure built-ins, which run as SQL under
   * the connection's credentials.
   */
  public SnowflakeSource semanticModel(String name, boolean trustExpressions) {
    return new SnowflakeSource(
        connection, database, schema, Objects.requireNonNull(name, "name"), trustExpressions);
  }

  /** The semantic view, when one was given. */
  public Optional<String> semanticModel() {
    return Optional.ofNullable(semanticModel);
  }

  /** Whether the caller allows the semantic view's expressions without validation. */
  public boolean trustExpressions() {
    return trustExpressions;
  }

  /** The caller-owned connection, kept open while reading source data. */
  public Connection connection() {
    return connection;
  }

  /** The database the tables are read in, when one was given. */
  public Optional<String> database() {
    return Optional.ofNullable(database);
  }

  /** The schema the tables are read in, when one was given. */
  public Optional<String> schema() {
    return Optional.ofNullable(schema);
  }

  @Override
  Map<String, Object> sqlChoice() {
    return Providers.sqlChoice("snowflake", semanticModel, trustExpressions, database, schema);
  }

  @Override
  Host.Provider driver() {
    return new JdbcProvider(connection, JdbcProvider.Driver.SNOWFLAKE);
  }

  @Override
  public String toString() {
    String source =
        "SnowflakeSource("
            + connection.getClass().getSimpleName()
            + ", "
            + database
            + ", "
            + schema
            + ")";
    return semanticModel == null
        ? source
        : "SemanticModel("
            + semanticModel
            + " in "
            + source
            + ", trustExpressions="
            + trustExpressions
            + ")";
  }
}
