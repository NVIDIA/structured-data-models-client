// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;

/**
 * Which of the run's errors carry a source's exception: a {@code PROVIDER_ERROR} only, so the core
 * refusing a failed fetch's reply with its own error (a reply the run does not await, say) names no
 * driver class and carries nothing, as Python's run loop does.
 */
class HostSourceErrorTest {
  @Test
  void onlyAProviderErrorCarriesTheSourcesException() {
    Exception failure = new IllegalStateException("the source failed");
    assertSame(failure, Host.sourceError("PROVIDER_ERROR", failure));
    assertNull(Host.sourceError("UNKNOWN_REPLY", failure));
    assertNull(Host.sourceError("INVALID_REQUEST", failure));
    assertNull(Host.sourceError("PROVIDER_ERROR", null));
  }
}
