// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The size limits a run's start carries ({@code com.nvidia.sdm.Limits}): the values the caller set
 * and the text of each setting's environment variable that is set, read when the run starts, which
 * the core resolves by the one rule every host reads them with. The settings' names and variables
 * are the core's ({@code sdm_contract::limits::SETTINGS}), read from the library's description.
 */
public final class LimitSettings {
  private LimitSettings() {}

  /** What a run's start carries of {@code values}, the settings the caller set by name. */
  @SuppressWarnings("unchecked")
  public static Map<String, Object> given(Map<String, Long> values) {
    Map<String, String> environment = new LinkedHashMap<>();
    for (Map<String, Object> setting :
        (List<Map<String, Object>>) NativeCore.load().get("limits")) {
      String name = (String) setting.get("env");
      String text = System.getenv(name);
      if (text != null) {
        environment.put(name, text);
      }
    }
    Map<String, Object> given = new LinkedHashMap<>();
    if (values != null && !values.isEmpty()) {
      given.put("given", new LinkedHashMap<>(values));
    }
    if (!environment.isEmpty()) {
      given.put("environment", environment);
    }
    return given;
  }
}
