// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * A built-in source's SQL kit, held by the native library ({@code sdm_host::kit}): it answers each
 * typed request the core sends the source by running the statements it writes on the caller's
 * driver. Internal; the built-in sources hold one.
 */
public final class NativeSqlKit implements AutoCloseable {

  /** One fetch run on the caller's driver. */
  @FunctionalInterface
  public interface Fetch {
    /**
     * The answer to the fetch whose plan and request are the contract's JSON, binding the key sets
     * {@code keys} by id; the caller closes the reader.
     */
    ArrowReader run(byte[] plan, byte[] request, Map<Integer, NativeArrow> keys) throws Exception;
  }

  /** The native side's view of a {@link Fetch}: the address of the stream holding its answer. */
  interface Run {
    long run(byte[] plan, byte[] request, int[] ids, Object[] keys) throws Exception;
  }

  private static final java.lang.ref.Cleaner CLEANER = java.lang.ref.Cleaner.create();

  /** The handle, shared with the cleaning action; zero once released. */
  private final java.util.concurrent.atomic.AtomicLong handle;

  private final java.lang.ref.Cleaner.Cleanable cleanable;

  /**
   * The kit of the built-in source {@code provider} names, the contract's JSON; released when
   * closed, or once unreachable.
   */
  public NativeSqlKit(byte[] provider) throws NativeError {
    NativeCore.load();
    this.handle = new java.util.concurrent.atomic.AtomicLong(NativeCore.sqlKit0(provider));
    java.util.concurrent.atomic.AtomicLong held = this.handle;
    this.cleanable =
        CLEANER.register(
            this,
            () -> {
              long released = held.getAndSet(0);
              if (released != 0) {
                NativeCore.releaseSqlKit(released);
              }
            });
  }

  /** What a driver's failure means, for the provider error {@code error}, as the contract's JSON. */
  public synchronized byte[] failure(byte[] error) throws NativeError {
    return NativeCore.sqlKitFailure0(held(), error);
  }

  /** What the source answers, as the contract's JSON. */
  public synchronized byte[] capabilities() throws NativeError {
    return NativeCore.sqlKitCapabilities0(held());
  }

  /**
   * The answer to {@code request} (the contract's JSON), its key sets {@code keys} by id, each fetch
   * run by {@code fetch}; the caller closes it.
   */
  public synchronized NativeArrow answer(
      byte[] request, Map<Integer, ArrowReader> keys, BufferAllocator allocator, Fetch fetch)
      throws Exception {
    List<ArrowArrayStream> exported = new ArrayList<>();
    try {
      int[] ids = new int[keys.size()];
      long[] streams = new long[keys.size()];
      int at = 0;
      for (Map.Entry<Integer, ArrowReader> key : keys.entrySet()) {
        ArrowArrayStream stream = ArrowExchange.export(allocator, key.getValue());
        exported.add(stream);
        ids[at] = key.getKey();
        streams[at] = stream.memoryAddress();
        at++;
      }
      Run run =
          (plan, asked, bindIds, bindKeys) -> {
            Map<Integer, NativeArrow> binds = new java.util.LinkedHashMap<>();
            for (int i = 0; i < bindIds.length; i++) {
              binds.put(bindIds[i], (NativeArrow) bindKeys[i]);
            }
            try {
              ArrowReader answer = fetch.run(plan, asked, binds);
              // Kept live until the native call returns, which reads it first.
              ArrowArrayStream stream = ArrowExchange.export(allocator, answer);
              exported.add(stream);
              return stream.memoryAddress();
            } finally {
              for (NativeArrow bound : binds.values()) {
                bound.close();
              }
            }
          };
      return (NativeArrow) NativeCore.sqlKitAnswer0(held(), request, ids, streams, run);
    } finally {
      for (ArrowArrayStream stream : exported) {
        stream.close();
      }
    }
  }

  private long held() {
    long current = handle.get();
    if (current == 0) {
      throw new IllegalStateException("this SQL kit was closed");
    }
    return current;
  }

  /** Releases the native kit; closing again does nothing. */
  @Override
  public synchronized void close() {
    cleanable.clean();
  }
}
