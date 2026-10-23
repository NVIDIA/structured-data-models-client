// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.List;
import java.util.Map;

/**
 * What crossed the binding and what the JDBC providers ran, told as it happens to a listener the
 * contract runner installs ({@code contract/runners/java/ContractRunner.java}), as the Python
 * runner's capture records it from the Python host: each run started with its attachments, each
 * step, each attachment taken, the run's result or error and warnings, and each statement a
 * provider ran with its result. Nothing is recorded when no listener is installed, which is
 * always outside the runner: every method here is then never called. An Arrow attachment or
 * result is told as the bytes of an Arrow IPC stream, written only for a listener. Not part of
 * the API.
 */
public interface Trace {
  /** A run started: the {@code start} message's bytes, and its attachments by id. */
  default void started(byte[] message, Map<Integer, Object> attachments) {}

  /** A step the core gave, as read. */
  default void stepped(Map<String, Object> step) {}

  /** An attachment the host took from the run: bytes, or an Arrow IPC stream's bytes. */
  default void taken(int id, Object value, boolean arrow) {}

  /** The run finished with {@code result}. */
  default void done(Map<String, Object> result) {}

  /** The run ended with the core's error, its {@code kind} and {@code code}. */
  default void failed(String kind, String code) {}

  /** The warnings the run gave, as the core wrote them. */
  default void warned(List<Map<String, Object>> warnings) {}

  /** A provider made a new cursor (a statement handle of its own). */
  default void cursor(int handle) {}

  /** A provider is about to run {@code sql} on {@code handle}. */
  default void statement(int handle, String sql) {}

  /**
   * The statement just run answered: its result read whole as an Arrow IPC stream, and each
   * column as the driver describes it ({@code type}, {@code precision}, {@code scale}).
   */
  default void answered(byte[] ipc, List<Map<String, Object>> columns) {}

  /** The statement just run failed. */
  default void statementFailed(String error) {}

  /** A key set bound as the relation {@code name} on {@code handle} by {@code sql}. */
  default void bound(int handle, String name, String sql, byte[] ipc) {}

  /** The listener, or null: set by the contract runner only. */
  static Trace installed() {
    return Holder.trace;
  }

  /** Installs {@code listener} (null removes it); the contract runner's alone. */
  static void install(Trace listener) {
    Holder.trace = listener;
  }

  /** Where the listener is held. */
  final class Holder {
    private static volatile Trace trace;

    private Holder() {}
  }
}
