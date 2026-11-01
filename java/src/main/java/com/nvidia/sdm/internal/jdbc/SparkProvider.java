// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import com.nvidia.sdm.internal.Host;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/** Runs existing Databricks plans in a caller-owned Spark session, without JDBC emulation. */
public final class SparkProvider implements Host.Provider {
  private final SparkSession session;
  public SparkProvider(SparkSession session) { this.session = session; }

  @Override
  @SuppressWarnings("unchecked")
  public ArrowReader answer(Map<String, Object> request, Map<Integer, ArrowReader> keys,
      BufferAllocator allocator) throws Exception {
    Object given = request.get("plan");
    if (!(given instanceof Map<?, ?> plan) || !"sql".equals(plan.get("plan"))
        || !(plan.get("statements") instanceof List<?> statements) || statements.isEmpty()) {
      throw new JdbcProvider.PlanNotRun("Spark requires a nonempty Databricks SQL plan");
    }
    if (plan.containsKey("result_conversions") && !List.of().equals(plan.get("result_conversions"))) {
      throw new JdbcProvider.PlanNotRun("Spark reads native Arrow timestamp meanings");
    }
    var prepared = JdbcProvider.databricksPlanned((List<Object>) statements, keys);
    var texts = JdbcProvider.texts(keys);
    Dataset<Row> result = null;
    for (var statement : prepared) {
      Object[] values = statement.values().stream().map(value -> value instanceof JdbcProvider.KeyText text
          ? texts.get(text.attachment()) : value).toArray();
      result = values.length == 0 ? session.sql(statement.text()) : session.sql(statement.text(), (Object) values);
      if (statement != prepared.get(prepared.size() - 1)) result.collectAsList();
    }
    return rows(java.util.Objects.requireNonNull(result), allocator);
  }

  public ArrowReader readQuery(String sql, BufferAllocator allocator) throws Exception {
    return rows(session.sql(sql), allocator);
  }

  private ArrowReader rows(Dataset<Row> data, BufferAllocator allocator) {
    Schema schema = SparkValues.schema(data.schema(), session.conf().get("spark.sql.session.timeZone"));
    Cursor iterator = new Cursor(data.toLocalIterator());
    return new ArrowReader(allocator) {
      @Override protected Schema readSchema() { return schema; }
      @Override public long bytesRead() { return 0; }
      @Override protected void closeReadSource() throws java.io.IOException { iterator.close(); }
      @Override public boolean loadNextBatch() throws java.io.IOException {
        ensureInitialized();
        if (!iterator.hasNext()) return false;
        List<List<Object>> columns = new ArrayList<>();
        for (int column = 0; column < schema.getFields().size(); column++) columns.add(new ArrayList<>());
        int size = 0;
        while (size < JdbcProvider.BATCH_ROWS && iterator.hasNext()) {
          Row row = iterator.next();
          for (int column = 0; column < columns.size(); column++) {
            columns.get(column).add(SparkValues.value(row.get(column), data.schema().apply(column).dataType()));
          }
          size++;
        }
        try (VectorSchemaRoot batch = ArrowColumns.batch(schema, columns, allocator);
            var record = new org.apache.arrow.vector.VectorUnloader(batch).getRecordBatch()) {
          new org.apache.arrow.vector.VectorLoader(getVectorSchemaRoot()).load(record);
        }
        return true;
      }
    };
  }

  /** Close a remote iterator when its public implementation provides that capability. */
  private static final class Cursor implements java.util.Iterator<Row>, java.io.Closeable {
    private final java.util.Iterator<Row> delegate;
    Cursor(java.util.Iterator<Row> delegate) { this.delegate = delegate; }
    @Override public boolean hasNext() { return delegate.hasNext(); }
    @Override public Row next() { return delegate.next(); }
    @Override public void close() throws java.io.IOException {
      if (delegate instanceof AutoCloseable closeable) {
        try { closeable.close(); }
        catch (InterruptedException interrupted) {
          Thread.currentThread().interrupt();
          throw new java.io.IOException("Interrupted while closing Spark result", interrupted);
        } catch (Exception failure) { throw new java.io.IOException("Could not close Spark result", failure); }
      }
    }
  }

  @Override
  public Map<String, Object> failure(Exception failure) {
    Map<String, Object> facts = new LinkedHashMap<>();
    if (failure instanceof org.apache.spark.SparkThrowable spark) {
      if (spark.getSqlState() != null) facts.put("sqlstate", spark.getSqlState());
      if (spark.getCondition() != null) facts.put("error_class", spark.getCondition());
    }
    if (failure.getMessage() != null) facts.put("driver_message", failure.getMessage());
    return facts;
  }
}
