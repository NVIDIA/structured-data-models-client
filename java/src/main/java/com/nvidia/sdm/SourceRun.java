// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.HostProviders;
import com.nvidia.sdm.internal.Providers;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * An operation's reach of its source, every source's alike: the core's provider choice and the
 * provider the core asks through ({@link HostProviders#chosen}, which reads the capabilities of the
 * source the operation asks as it starts), and the host values the core's sentences name of it. Not
 * part of the API.
 */
record SourceRun(Source source, Providers.Chosen chosen) {

  /**
   * The operation over {@code source} starting. A built-in source is asked through its operation's
   * own answers ({@link BuiltInSource#operation}, a fresh SQL kit), so what the kit keeps within an
   * operation is that operation's alone; any other source is asked itself.
   */
  static SourceRun of(Source source) {
    Source asked = source instanceof BuiltInSource builtIn ? builtIn.operation() : source;
    return new SourceRun(source, HostProviders.chosen(asked));
  }

  /** The core's provider choice. */
  Map<String, Object> choice() {
    return chosen.choice();
  }

  /** The provider the run loop asks. */
  Host.Provider provider() {
    return chosen.provider();
  }

  /**
   * The host values the core's sentences name of the source: the source, and, where it declares
   * one, its semantic model.
   */
  Map<String, Function<Map<String, Object>, Object>> held() {
    Map<String, Function<Map<String, Object>, Object>> held = new LinkedHashMap<>();
    held.put("source", fields -> source);
    if (chosen.readsSemanticModel()) {
      held.put("semantic_model", fields -> source);
    }
    return Map.copyOf(held);
  }

  /**
   * What the core compares a source and its replacement by when a schema is rebound, beyond their
   * choices: its class, and whether it reads a semantic model, as the Python host names it.
   */
  String kind() {
    return source.getClass().getSimpleName() + (chosen.readsSemanticModel() ? ".semantic_model" : "");
  }
}
