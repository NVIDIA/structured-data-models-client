// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Providers;
import java.util.Map;
import java.util.function.Function;

/** What a source tells the library, for tests outside its package. */
public final class SourceAccess {
  private SourceAccess() {}

  /** The provider choice and provider a run over {@code source} names. */
  public static Providers.Chosen chosen(Source source) {
    return source.chosen();
  }

  /** The host values the core's sentences name of {@code source}. */
  public static Map<String, Function<Map<String, Object>, Object>> held(Source source) {
    return source.held();
  }

  /** A built-in source's SQL choice, before its kit declares it. */
  public static Map<String, Object> sqlChoice(Source source) {
    return ((BuiltInSource) source).sqlChoice();
  }

  /** The provider running a built-in source's statements. */
  public static Host.Provider driver(Source source) {
    return ((BuiltInSource) source).driver();
  }
}
