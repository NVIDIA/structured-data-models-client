// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import org.postgresql.PGConnection;
import org.postgresql.PGStatement;
import org.postgresql.core.BaseConnection;
import org.postgresql.core.TransactionState;

/**
 * What the Postgres provider asks of pgjdbc beyond JDBC, in one place, so no other class names
 * the driver's own types (and a host without pgjdbc never loads them): a column type's OID, a
 * session parameter, whether a transaction is open, and a statement read in text. Not part of the
 * API.
 */
final class PgSession {
  private PgSession() {}

  /** pgjdbc's connection under {@code connection}, or a refusal naming what it is not. */
  private static BaseConnection base(Connection connection) throws SQLException {
    if (!connection.isWrapperFor(BaseConnection.class)) {
      throw new SQLException("a Postgres connection this client reads is pgjdbc's");
    }
    return connection.unwrap(BaseConnection.class);
  }

  /** The OID of the type pgjdbc names {@code name} (as its result metadata names it). */
  static int oid(Connection connection, String name) throws SQLException {
    return base(connection).getTypeInfo().getPGType(name);
  }

  /** A parameter the server reports for the session ({@code IntervalStyle}, ...). */
  static String parameter(Connection connection, String name) throws SQLException {
    return connection.unwrap(PGConnection.class).getParameterStatus(name);
  }

  /** Whether the session is inside a transaction (open, or failed). */
  static boolean inTransaction(Connection connection) throws SQLException {
    return base(connection).getTransactionState() != TransactionState.IDLE;
  }

  /**
   * {@code statement} read in text, as psycopg reads every value: never prepared on the server,
   * so pgjdbc never switches it to binary from a later run.
   */
  static void inText(Statement statement) throws SQLException {
    statement.unwrap(PGStatement.class).setPrepareThreshold(0);
  }
}
