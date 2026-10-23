// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Where the core's {@code tracing} events arrive (design choice 10, KO3-30): each is written to
 * the SLF4J logger its target names, the Rust path's {@code ::} read as {@code .} ({@code
 * sdm_core::test_ops} is the logger {@code sdm_core.test_ops}). The JAR declares the SLF4J 2 API
 * and binds no implementation. Not part of the API.
 */
public final class NativeLog {
  /** The loggers of the core's crates, whose levels decide what crosses from the core. */
  private static final String[] ROOTS = {
    "sdm_contract", "sdm_core", "sdm_data", "sdm_engine", "sdm_host", "sdm_pql", "sdm_request",
    "sdm_schema", "sdm_wire",
  };

  private NativeLog() {}

  /** Called by the native library with one event: level 1 error to 5 trace. */
  static void log(int level, String target, String message) {
    Logger logger = LoggerFactory.getLogger(target.replace("::", "."));
    switch (level) {
      case 1 -> logger.error(message);
      case 2 -> logger.warn(message);
      case 3 -> logger.info(message);
      case 4 -> logger.debug(message);
      default -> logger.trace(message);
    }
  }

  /**
   * The most verbose level any of the core's loggers is enabled at, as the native library numbers
   * it (0 none); read at each run start, so a change to SLF4J's configuration takes effect.
   */
  static int level() {
    int most = 0;
    for (String root : ROOTS) {
      Logger logger = LoggerFactory.getLogger(root);
      int enabled =
          logger.isTraceEnabled()
              ? 5
              : logger.isDebugEnabled()
                  ? 4
                  : logger.isInfoEnabled() ? 3 : logger.isWarnEnabled() ? 2 : logger.isErrorEnabled() ? 1 : 0;
      most = Math.max(most, enabled);
    }
    return most;
  }
}
