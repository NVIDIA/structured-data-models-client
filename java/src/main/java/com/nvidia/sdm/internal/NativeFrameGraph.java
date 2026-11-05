// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.lang.ref.Cleaner;
import java.lang.ref.Reference;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The native owner of a retained relational graph and its sampling state. Not public API.
 * Each call leases a reference before entering the core, so closing this object cannot release
 * a graph underneath an admitted call. The materializer separately serializes sampling order.
 */
public final class NativeFrameGraph implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();
  private final AtomicLong handle;
  private final Cleaner.Cleanable cleanable;

  /** Made by the native library only. */
  NativeFrameGraph(long handle) {
    this.handle = new AtomicLong(handle);
    AtomicLong held = this.handle;
    this.cleanable = CLEANER.register(this, () -> release(held));
  }

  private static void release(AtomicLong held) {
    long released = held.getAndSet(0);
    if (released != 0) NativeCore.releaseFrameGraph(released);
  }

  /** The caller owns this temporary handle and releases it after the JNI attachment transfer. */
  synchronized long retainHandle() {
    long current = handle.get();
    if (current == 0) throw new IllegalStateException("this frame graph was closed");
    try {
      return NativeCore.retainFrameGraph(current);
    } finally {
      Reference.reachabilityFence(this);
    }
  }

  /** Release this owner once; leases already admitted remain valid. */
  @Override
  public synchronized void close() {
    cleanable.clean();
  }
}
