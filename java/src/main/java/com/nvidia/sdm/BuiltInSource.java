// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.KitRun;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.ProviderFailure;
import java.util.Map;
import java.util.Optional;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * What every built-in source shares: a connection, the namespace its tables live in, the semantic
 * model it reads, and the driver that runs the SQL the library writes for its database, as
 * Python's {@code _BuiltInSource}. Not part of the API.
 *
 * <p>Each operation's answers are its own: the core reaches a built-in source through {@link
 * #operation}, a fresh SQL kit for that operation ({@link SourceRun#of}), which settles a
 * namespace left to the connection once and reads an identical catalog read once, within that
 * operation only. Asked directly, {@link #capabilities}, {@link #answer} and {@link #failure} each
 * answer through a kit of their own.
 */
abstract sealed class BuiltInSource implements Source
    permits DuckDbSource,
        SqliteSource,
        PostgresSource,
        DatabricksSource,
        SnowflakeSource,
        DatabricksSparkSource {

  BuiltInSource() {}

  /** What this source answers, as the library's SQL for its database declares it. */
  @Override
  public final Capabilities capabilities() {
    return operation().capabilities();
  }

  /**
   * Answers {@code request} with the library's SQL for this database, run on the connection, in the
   * contract's shape ({@code expected}, where the request fixes one).
   *
   * @throws StructuredDataException a request this database's SQL cannot answer, refused by name
   * @throws Exception the driver's failure, as it raised it
   */
  @Override
  public final ArrowReader answer(
      DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
    return operation().answer(request, expected, allocator);
  }

  /** What the driver's failure tells the core: its codes, and what it means on this database. */
  @Override
  public final ProviderFailure failure(Exception failure) {
    return operation().failure(failure);
  }

  /**
   * One operation's answers: a fresh SQL kit over the connection, the source the core asks for that
   * operation.
   */
  final KitRun operation() {
    return new KitRun(sqlChoice(), driver());
  }

  /**
   * The contract's SQL choice of this source: its dialect, its namespace (each part null where the
   * connection's current one is meant; left out where none is given), and the semantic model it
   * reads, where it reads one.
   */
  abstract Map<String, Object> sqlChoice();

  /** The provider running the kit's statements on the caller's connection or session. */
  abstract Host.Provider driver();
}
