// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Providers;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * What every built-in source shares: the core writes the SQL for its database, and the driver's
 * provider runs it on the caller's connection. Not part of the API.
 */
abstract sealed class BuiltInSource extends Source
    permits DuckDbSource,
        SqliteSource,
        PostgresSource,
        DatabricksSource,
        SnowflakeSource,
        DatabricksSparkSource {

  /** Refused: a built-in source is answered by the library's own SQL, never asked directly. */
  @Override
  public final Capabilities capabilities() {
    throw notAsked();
  }

  /** Refused: a built-in source is answered by the library's own SQL, never asked directly. */
  @Override
  public final ArrowReader answer(
      DataRequest request, Optional<Schema> expected, BufferAllocator allocator) {
    throw notAsked();
  }

  private StructuredDataException notAsked() {
    return new StructuredDataException(
        getClass().getSimpleName()
            + " is answered by the library's own SQL for its database; pass it where a source is"
            + " taken, such as Schema.fromSource, rather than asking it directly",
        "UNSUPPORTED_REQUEST",
        Map.of(),
        null);
  }

  /**
   * The contract's SQL choice of this source: its dialect, its namespace (each part null where the
   * connection's current one is meant; left out where none is given), and the semantic model it
   * reads, where it reads one.
   */
  abstract Map<String, Object> sqlChoice();

  /** The provider running the kit's statements on the caller's connection or session. */
  abstract Host.Provider driver();

  @Override
  final Providers.Chosen chosen() {
    return Providers.sql(sqlChoice(), driver());
  }

  @Override
  final Map<String, Function<Map<String, Object>, Object>> held() {
    Map<String, Function<Map<String, Object>, Object>> held = new LinkedHashMap<>(super.held());
    if (readsSemanticModel()) {
      held.put("semantic_model", fields -> this);
    }
    return Map.copyOf(held);
  }

  @Override
  final String kind() {
    return super.kind() + (readsSemanticModel() ? ".semantic_model" : "");
  }

  private boolean readsSemanticModel() {
    return sqlChoice().containsKey("view");
  }
}
