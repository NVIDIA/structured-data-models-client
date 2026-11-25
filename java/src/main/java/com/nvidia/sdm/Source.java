// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.ProviderFailure;
import java.util.Optional;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Where a schema's data is read, answering the core's data requests.
 *
 * <p>The core never sends SQL to a source. It sends one of a fixed set of typed requests ({@link
 * DataRequest}), and the source answers each with Arrow rows. Every request a source receives is
 * one its {@link Capabilities} declared, or {@code describe} and {@code scan}, which every source
 * answers.
 *
 * <p>Implement it to read data your own way, and pass an instance wherever the library takes a
 * source:
 *
 * <pre>{@code
 * Schema schema = Schema.fromSource(new MySource(), options);
 * }</pre>
 *
 * <p>The built-in sources ({@link DuckDbSource}, {@link SqliteSource}, {@link PostgresSource},
 * {@link DatabricksSource}, {@link SnowflakeSource}, {@link DatabricksSparkSource}) are sources like
 * any other: each answers these requests with the library's own SQL for its database, and the core
 * reaches every source the same way.
 *
 * <p>The library reads {@link #capabilities} once at the start of each operation, before any
 * request of it, then calls {@link #answer} from the thread that runs the operation, one request at
 * a time, and never closes the source.
 */
public interface Source {

  /** What this source answers; read once at the start of each operation, before any request. */
  Capabilities capabilities();

  /**
   * Answers one request with every row it asks for, as an Arrow reader the library closes.
   *
   * @param request the request
   * @param expected the exact schema the answer must have, where the request fixes one ({@code
   *     describe}, {@code key_counts}); otherwise empty, and the answer holds the columns the request
   *     names, in order, typed as the source stores them
   * @param allocator the allocator the answer's vectors are allocated from
   * @throws Exception any failure; the operation then fails with {@code PROVIDER_ERROR}, carrying
   *     what {@link #failure} tells of it
   */
  ArrowReader answer(DataRequest request, Optional<Schema> expected, BufferAllocator allocator)
      throws Exception;

  /**
   * What a failure tells the core beyond its class: by default, a {@link java.sql.SQLException}'s
   * state, vendor code and message. The message is kept internal and never shown to end users.
   */
  default ProviderFailure failure(Exception failure) {
    return ProviderFailure.of(failure);
  }
}
