// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.HostProviders;
import com.nvidia.sdm.internal.Providers;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.ProviderFailure;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
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
 * <p>Extend it to read data your own way, and pass an instance wherever the library takes a source:
 *
 * <pre>{@code
 * Schema schema = Schema.fromSource(new MySource(), options);
 * }</pre>
 *
 * <p>The built-in sources ({@link DuckDbSource}, {@link SqliteSource}, {@link PostgresSource},
 * {@link DatabricksSource}, {@link SnowflakeSource}, {@link DatabricksSparkSource}) are sources
 * too, each answered by the library's own SQL for its database.
 *
 * <p>The library calls a source from the thread that runs the operation, one request at a time,
 * and never closes it.
 */
public abstract class Source {

  /** A source of the caller's own. */
  protected Source() {}

  /** What this source answers; read once per operation. */
  public abstract Capabilities capabilities();

  /**
   * Answers one request with every row it asks for, as an Arrow reader the library closes.
   *
   * @param request the request
   * @param expected the exact schema the answer must have, where the request fixes one ({@code
   *     describe}, {@code key_counts}, {@code time_bounds}); otherwise empty, and the answer holds
   *     the columns the request names, in order, typed as the source stores them
   * @param allocator the allocator the answer's vectors are allocated from
   * @throws Exception any failure; the operation then fails with {@code PROVIDER_ERROR}, carrying
   *     what {@link #failure} tells of it
   */
  public abstract ArrowReader answer(
      DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception;

  /**
   * What a failure tells the core beyond its class: by default, a {@link java.sql.SQLException}'s
   * state, vendor code and message. The message is kept internal and never shown to end users.
   */
  public ProviderFailure failure(Exception failure) {
    return ProviderFailure.of(failure);
  }

  /**
   * The core's provider choice for this source, and the run loop's provider answering its
   * fetches: the caller's {@link #capabilities} and {@link #answer}. A built-in source answers
   * through its SQL kit instead.
   */
  Providers.Chosen chosen() {
    return HostProviders.chosen(this);
  }

  /** The host values the core's sentences name of this source. */
  Map<String, Function<Map<String, Object>, Object>> held() {
    return Map.of("source", fields -> this);
  }

  /**
   * What the core compares this source and its replacement by when a schema is rebound, beyond
   * their choices: its class, and whether it reads a semantic model, as the Python host names it.
   */
  String kind() {
    return getClass().getSimpleName();
  }
}
