// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;

/** A JVM on a platform the JAR holds no native for: the named refusal, twice alike. */
public final class NoNativeMain {
  private NoNativeMain() {}

  public static void main(String[] arguments) {
    for (int i = 0; i < 2; i++) {
      try {
        NativeCore.load();
        System.out.println("loaded");
      } catch (StructuredDataException refused) {
        System.out.println(refused.code() + " " + refused.details().get("platform"));
      }
    }
  }
}
