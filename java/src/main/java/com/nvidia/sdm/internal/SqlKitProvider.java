// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * A built-in source answered through its SQL kit, as Python's {@code sql_kit.KitProvider}: the core
 * sends it the typed requests it sends every source, and the kit, the engine's SQL written once in
 * the core's library, runs each on the caller's driver ({@code driver}), which writes no SQL. One
 * answers one run.
 */
final class SqlKitProvider implements Host.Provider {

  /** The kit refused a request by name: the error the core gives, as the contract writes it. */
  static final class Refused extends Exception {
    private static final long serialVersionUID = 1L;

    private final transient Map<String, Object> error;

    Refused(Map<String, Object> error) {
      super("the source refused the request", null, false, false);
      this.error = error;
    }

    Map<String, Object> error() {
      return error;
    }
  }

  private final NativeSqlKit kit;
  private final Host.Provider driver;

  SqlKitProvider(NativeSqlKit kit, Host.Provider driver) {
    this.kit = kit;
    this.driver = driver;
  }

  /**
   * The provider choice and provider of the built-in source whose SQL choice is {@code choice}, its
   * statements run by {@code driver}: a source declaring what its kit declares.
   */
  @SuppressWarnings("unchecked")
  static Providers.Chosen chosen(Map<String, Object> choice, Host.Provider driver) {
    try {
      NativeSqlKit kit = new NativeSqlKit(Json.write(choice));
      Map<String, Object> capabilities = (Map<String, Object>) Json.read(kit.capabilities());
      Map<String, Object> host = new LinkedHashMap<>();
      host.put("provider", "host");
      host.put("capabilities", capabilities);
      return new Providers.Chosen(host, new SqlKitProvider(kit, driver));
    } catch (NativeError e) {
      throw e;
    } catch (Json.JsonException e) {
      throw new IllegalStateException("a built-in source's choice could not be written", e);
    }
  }

  @Override
  @SuppressWarnings("unchecked")
  public ArrowReader answer(
      Map<String, Object> request, Map<Integer, ArrowReader> keys, BufferAllocator allocator)
      throws Exception {
    byte[] asked = Json.write(request.get("request"));
    NativeArrow answer;
    try {
      answer =
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
    } catch (NativeError refused) {
      Map<String, Object> control =
          (Map<String, Object>) Json.read(refused.message().control());
      throw new Refused((Map<String, Object>) control.get("error"));
    }
    try (answer) {
      return answer.reader(allocator);
    }
  }

  @Override
  public Map<String, Object> failure(Exception failure) {
    return driver.failure(failure);
  }

  @Override
  @SuppressWarnings("unchecked")
  public Map<String, Object> facts(Map<String, Object> provided) throws Exception {
    return (Map<String, Object>) Json.read(kit.failure(Json.write(provided)));
  }
}
