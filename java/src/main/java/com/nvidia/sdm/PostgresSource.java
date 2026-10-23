// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.sql.Connection;
import java.util.Objects;
import java.util.Optional;

/**
 * A Postgres database, through Postgres's JDBC driver ({@code org.postgresql:postgresql}), as
 * Python's {@code PostgresSource}: the tables of the schema it is given, or of the connection's
 * current schema when it is given none. Each read runs in a transaction of its own (none under
 * autocommit, else a savepoint inside the caller's), so a failed read leaves the connection
 * usable.
 */
public final class PostgresSource implements Source {
  private final Connection connection;
  private final String schema;

  private PostgresSource(Connection connection, String schema) {
    this.connection = Objects.requireNonNull(connection, "connection");
    this.schema = schema;
  }

  /** The tables of {@code connection}'s current schema. */
  public static PostgresSource of(Connection connection) {
    return new PostgresSource(connection, null);
  }

  /** The tables of the Postgres schema {@code schema}, read through {@code connection}. */
  public static PostgresSource of(Connection connection, String schema) {
    return new PostgresSource(connection, Objects.requireNonNull(schema, "schema"));
  }

  @Override
  public Connection connection() {
    return connection;
  }

  /** The schema the tables are read in, when one was given. */
  public Optional<String> schema() {
    return Optional.ofNullable(schema);
  }

  @Override
  public String toString() {
    return "PostgresSource(" + connection + (schema == null ? "" : ", " + schema) + ")";
  }
}
