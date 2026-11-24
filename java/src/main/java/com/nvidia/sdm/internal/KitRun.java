// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.Source;
import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.ProviderFailure;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * One operation's answers of a built-in source, as Python's {@code sql_kit.KitRun}: its SQL kit,
 * the engine's SQL written once in the core's library, answering each typed request on the
 * source's driver ({@code driver}), which runs the statements it is handed and writes no SQL.
 *
 * <p>The kit is one per operation: it settles the namespace a source leaves to its connection
 * once, and reads an identical catalog read once, for that operation only. The core asks it in
 * place of the built-in source for that operation, through the same {@link Source} methods as any
 * other source. Not part of the API.
 */
public final class KitRun implements Source {
  private final NativeSqlKit kit;
  private final Host.Provider driver;

  /**
   * The operation of the built-in source whose SQL choice is {@code choice} (the contract's {@code
   * SqlChoice}), its statements run by {@code driver}.
   */
  public KitRun(Map<String, Object> choice, Host.Provider driver) {
    try {
      this.kit = new NativeSqlKit(Json.write(choice));
    } catch (Json.JsonException e) {
      throw new IllegalStateException("a built-in source's choice could not be written", e);
    }
    this.driver = driver;
  }

  /** What the source answers, as its kit declares it. */
  @Override
  @SuppressWarnings("unchecked")
  public Capabilities capabilities() {
    try {
      return HostProviders.readCapabilities((Map<String, Object>) Json.read(kit.capabilities()));
    } catch (Json.JsonException e) {
      throw new IllegalStateException("the kit's capabilities could not be read", e);
    }
  }

  /**
   * The answer to {@code request}, in the contract's shape, which the kit writes itself, so {@code
   * expected} is not read; its vectors allocated from {@code allocator}.
   *
   * @throws NativeError the kit refused the request by name, the error the operation fails with
   * @throws Exception the driver's failure, as it raised it
   */
  @Override
  public ArrowReader answer(
      DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
    Map<Integer, ArrowReader> keys = new LinkedHashMap<>();
    byte[] asked = Json.write(HostProviders.Requests.write(request, keys));
    NativeArrow answer =
        kit.answer(
            asked,
            keys,
            allocator,
            (plan, fetchRequest, binds) -> {
              Map<String, Object> fetch = new LinkedHashMap<>();
              fetch.put("plan", Json.read(plan));
              fetch.put("request", Json.read(fetchRequest));
              Map<Integer, ArrowReader> readers = new LinkedHashMap<>();
              for (Map.Entry<Integer, NativeArrow> bind : binds.entrySet()) {
                readers.put(bind.getKey(), bind.getValue().reader(allocator));
              }
              return driver.answer(fetch, readers, allocator);
            });
    try (answer) {
      return answer.reader(allocator);
    }
  }

  /**
   * What the driver's failure tells the core: what the driver reads of it, and what it means, as
   * the kit reads it.
   */
  @Override
  @SuppressWarnings("unchecked")
  public ProviderFailure failure(Exception failure) {
    Map<String, Object> provided = Host.providerError(failure, driver.failure(failure));
    Map<String, Object> told = new LinkedHashMap<>(provided);
    try {
      told.put("facts", Json.read(kit.failure(Json.write(provided))));
    } catch (Json.JsonException | RuntimeException unread) {
      // Unread facts leave the failure as the driver said it.
    }
    return HostProviders.readFailure(told);
  }
}
