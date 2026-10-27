// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

/**
 * A control message the core wrote, as the native library hands it over: its control bytes, and
 * the attachments it names by id, each a {@code byte[]} or a {@link NativeArrow}. Whoever takes a
 * {@link NativeArrow} from it owns it and closes it; {@link #close} closes those not taken. Not
 * part of the API.
 */
public final class NativeMessage implements AutoCloseable {
  private final byte[] control;
  private final int[] ids;
  private final Object[] values;

  /** Made by the native library only. */
  NativeMessage(byte[] control, int[] ids, Object[] values) {
    this.control = control;
    this.ids = ids;
    this.values = values;
  }

  /** The control message's bytes. */
  public byte[] control() {
    return control;
  }

  /** The bytes attachment {@code id}, or null when the message names no bytes by that id. */
  public byte[] bytes(int id) {
    for (int i = 0; i < ids.length; i++) {
      if (ids[i] == id && values[i] instanceof byte[] bytes) {
        return bytes;
      }
    }
    return null;
  }

  /**
   * Takes the Arrow attachment {@code id}: the caller owns it from here on. Null when the message
   * names no Arrow by that id, or it was taken.
   */
  public synchronized NativeArrow takeArrow(int id) {
    for (int i = 0; i < ids.length; i++) {
      if (ids[i] == id && values[i] instanceof NativeArrow arrow) {
        values[i] = null;
        return arrow;
      }
    }
    return null;
  }

  /** The ids of the attachments the message carries. */
  public int[] ids() {
    return ids.clone();
  }

  /** Closes every Arrow attachment no one took. */
  @Override
  public synchronized void close() {
    for (int i = 0; i < values.length; i++) {
      if (values[i] instanceof NativeArrow arrow) {
        values[i] = null;
        arrow.close();
      }
    }
  }
}
