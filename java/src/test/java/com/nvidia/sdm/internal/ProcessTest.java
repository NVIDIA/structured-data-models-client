// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** What only a process shows: a contained panic, the logging bridge, a platform with no native. */
class ProcessTest {
  @Test
  void aPanicInsideTheCoreIsAnExceptionAndTheJvmRunsOn() throws Exception {
    TestOps.required();
    TestOps.Ended ended = TestOps.jvm(PanicMain.class);
    assertEquals(0, ended.status(), ended.err());
    assertTrue(ended.out().contains("start: INTERNAL_ERROR panic=true"), ended.out());
    assertTrue(ended.out().contains("body: INTERNAL_ERROR panic=true"), ended.out());
    assertTrue(ended.out().contains("adapter_stack: INTERNAL_ERROR panic=true"), ended.out());
    assertTrue(ended.out().contains("survived accepted=true"), ended.out());
    // The panic's message, which may quote input, is never written.
    assertFalse(ended.err().contains("boom"), ended.err());
  }

  @Test
  void theCoresEventsReachSlf4j() throws Exception {
    TestOps.required();
    TestOps.Ended ended =
        TestOps.jvm(LogMain.class, "org.slf4j.simpleLogger.log.sdm_core=info");
    assertEquals(0, ended.status(), ended.err());
    assertTrue(ended.err().contains("sdm_core.test_ops"), ended.err());
    assertTrue(ended.err().contains("a test-only event"), ended.err());
  }

  @Test
  void anEventSlf4jDropsIsNotWritten() throws Exception {
    TestOps.required();
    TestOps.Ended ended = TestOps.jvm(LogMain.class);
    assertEquals(0, ended.status(), ended.err());
    assertFalse(ended.err().contains("a test-only event"), ended.err());
  }

  @Test
  void aPlatformWithNoNativeIsRefusedByName() throws Exception {
    TestOps.Ended ended = TestOps.jvm(NoNativeMain.class, "os.arch=sparcv9", "sdm.native.library=");
    assertEquals(0, ended.status(), ended.err());
    assertTrue(ended.out().contains("UNSUPPORTED_REQUEST"), ended.out());
    assertTrue(ended.out().contains("-sparcv9"), ended.out());
  }

  @Test
  void withoutArrowsFlagArrowIsRefusedByNameAndValidationStillWorks() throws Exception {
    TestOps.required();
    TestOps.Ended ended = TestOps.jvmWithout(ArrowExchange.FLAG, NoFlagMain.class);
    assertEquals(0, ended.status(), ended.err());
    assertTrue(ended.out().contains("validated accepted=true"), ended.out());
    assertTrue(
        ended.out().contains("INVALID_CONFIGURATION " + ArrowExchange.FLAG), ended.out());
  }
}
