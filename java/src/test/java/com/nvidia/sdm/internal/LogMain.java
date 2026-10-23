// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.Map;

/** The logging test's subprocess: one run of the test-only operation that logs one event. */
public final class LogMain {
  private LogMain() {}

  public static void main(String[] arguments) throws Exception {
    NativeCore.load();
    // The level is read from SLF4J at each run start (Host.run); here the
    // run is started directly, so it is set as Host.run sets it.
    NativeCore.setLogLevel(NativeLog.level());
    try (Run run = Run.start(TestOps.start("log", Map.of("hold_ms", 0)), Map.of())) {
      run.next().close();
    }
    System.out.println("logged");
  }
}
