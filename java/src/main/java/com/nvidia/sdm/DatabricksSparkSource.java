// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import org.apache.spark.sql.SparkSession;
import java.util.Objects;
import java.util.Optional;

/** Tables accessed through the caller's Databricks Spark session; Rust writes sampling SQL. */
public final class DatabricksSparkSource implements Source {
  private final SparkSession session;
  private final String catalog;
  private final String schema;
  private final String semanticModel;
  private final boolean trustExpressions;

  private DatabricksSparkSource(SparkSession session, String catalog, String schema) {
    this(session, catalog, schema, null, false);
  }

  private DatabricksSparkSource(SparkSession session, String catalog, String schema,
      String semanticModel, boolean trustExpressions) {
    this.session = Objects.requireNonNull(session, "session");
    this.catalog = catalog;
    this.schema = schema;
    this.semanticModel = semanticModel;
    this.trustExpressions = trustExpressions;
  }

  /** The tables of {@code session}'s current catalog and schema. */
  public static DatabricksSparkSource of(SparkSession session) {
    return new DatabricksSparkSource(session, null, null);
  }

  /**
   * The tables of {@code catalog}.{@code schema}, read through {@code session}; either may be
   * null for the session's current one.
   */
  public static DatabricksSparkSource of(SparkSession session, String catalog, String schema) {
    return new DatabricksSparkSource(session, catalog, schema);
  }

  /** A metric view in this source's namespace, interpreted by the core. */
  public DatabricksSparkSource semanticModel(String name, boolean trustExpressions) {
    return new DatabricksSparkSource(session, catalog, schema,
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

  /** The caller-owned Spark session. */
  public SparkSession session() {
    return session;
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
  public String toString() {
    String source = "DatabricksSparkSource(" + session + ", " + catalog + ", " + schema + ")";
    return semanticModel == null ? source
        : "SemanticModel(" + semanticModel + " in " + source
            + ", trustExpressions=" + trustExpressions + ")";
  }
}
