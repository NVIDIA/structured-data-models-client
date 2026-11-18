// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.provider;

import java.util.Optional;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * The caller's own data access, answering the core's data requests.
 *
 * <p>The core never sends SQL to a provider. It sends one of a fixed set of typed requests ({@link
 * DataRequest}), and the provider answers each with Arrow rows. Every request a provider receives
 * is one its {@link Capabilities} declared, or {@code describe} and {@code scan}, which every
 * provider answers.
 *
 * <p>Wrap a provider in {@link com.nvidia.sdm.ProviderSource} to use it wherever the library takes
 * a {@link com.nvidia.sdm.Source}: {@code Schema.fromSource}, predictions, and so on.
 *
 * <p>The library calls a provider from the thread that runs the operation, one request at a time.
 */
public interface DataProvider {

  /** What this provider answers; read once per operation. */
  Capabilities capabilities();

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
