// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * Arrow data the core made, held by the native library until it is closed: the one owner of its
 * native handle, which {@link #close} releases once (and a {@link Cleaner}, should it never be
 * closed). {@link #reader} exports it through the Arrow C Data interface with no copy; the reader
 * may outlive this object. Not part of the API.
 */
public final class NativeArrow implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();

  /** The handle, shared with the cleaning action; zero once released. */
  private final AtomicLong handle;

  private final Cleaner.Cleanable cleanable;

  /** Made by the native library only. */
  NativeArrow(long handle) {
    this.handle = new AtomicLong(handle);
    AtomicLong held = this.handle;
    this.cleanable = CLEANER.register(this, () -> release(held));
  }

  private static void release(AtomicLong held) {
    long released = held.getAndSet(0);
    if (released != 0) {
      NativeCore.releaseArrow(released);
    }
  }

  /**
   * The data as an Arrow reader, its buffers the core's, imported with no copy; the caller closes
   * the reader.
   *
   * @throws IllegalStateException when this was closed
   * @throws com.nvidia.sdm.StructuredDataException {@code INVALID_CONFIGURATION} when the JVM
   *     was started without the flag Arrow Java needs ({@link ArrowExchange#FLAG})
   */
  public synchronized ArrowReader reader(BufferAllocator allocator) {
    long current = handle.get();
    if (current == 0) {
      throw new IllegalStateException("this Arrow attachment was closed");
    }
    return ArrowExchange.importing(allocator, current);
  }

  /** Releases the native data; closing again does nothing. */
  @Override
  public synchronized void close() {
    cleanable.clean();
  }
}
