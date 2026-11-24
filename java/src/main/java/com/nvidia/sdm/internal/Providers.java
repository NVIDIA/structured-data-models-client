// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The provider choice a run over a source names, and the provider answering its fetches, every
 * source's alike ({@link HostProviders#chosen}); and a built-in source's SQL choice, which its
 * kit is opened with ({@link KitRun}). Not part of the API.
 */
public final class Providers {
  private Providers() {}

  /** The choice and provider of one source. */
  public record Chosen(Map<String, Object> choice, Host.Provider provider) {
    /** Whether the source declares a semantic model it is read through. */
    @SuppressWarnings("unchecked")
    public boolean readsSemanticModel() {
      return ((Map<String, Object>) choice.get("capabilities")).containsKey("view");
    }
  }

  /**
   * The contract's SQL choice of a built-in source: {@code dialect}, the namespace {@code parts}
   * outermost first (each null where the connection's current one is meant), left out when none is
   * given, and the semantic model {@code view}, where one is read.
   */
  public static Map<String, Object> sqlChoice(
      String dialect, String view, boolean trustExpressions, String... parts) {
    Map<String, Object> choice = new LinkedHashMap<>();
    choice.put("provider", "sql");
    choice.put("dialect", dialect);
    if (Arrays.stream(parts).anyMatch(part -> part != null)) {
      choice.put("namespace", Arrays.asList(parts));
    }
    if (view != null) {
      choice.put("view", Map.of("name", view, "trust_expressions", trustExpressions));
    }
    return choice;
  }
}
