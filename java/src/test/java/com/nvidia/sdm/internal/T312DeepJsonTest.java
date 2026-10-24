// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.Test;

/**
 * Codex T3.12 r2 P1-2: no JSON value crashes the JVM. The public call runs in a process of its
 * own ({@link DeepJsonMain}), so a crash fails this test rather than the test run.
 */
class T312DeepJsonTest {
  @Test
  void aDeepJsonValueIsReadOrRefusedByNameAndTheJvmRunsOn() throws Exception {
    String dsn = System.getenv("SDM_POSTGRES_TEST_DSN");
    if (dsn == null || dsn.isEmpty()) {
      assertTrue(
          !"1".equals(System.getenv("SDM_REQUIRE_POSTGRES")),
          "SDM_REQUIRE_POSTGRES is set and SDM_POSTGRES_TEST_DSN names no server");
      assumeTrue(false, "no Postgres server: set SDM_POSTGRES_TEST_DSN");
    }
    TestOps.Ended ended = TestOps.jvm(DeepJsonMain.class);
    assertEquals(0, ended.status(), ended.out() + ended.err());
    assertTrue(ended.out().contains("described 920 tables=1"), ended.out());
    assertTrue(ended.out().contains("described 1900 tables=1"), ended.out());
    assertTrue(ended.out().contains("refused 2100 "), ended.out());
    assertTrue(ended.out().contains("survived"), ended.out());
  }
}
