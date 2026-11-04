// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Set;
import org.junit.jupiter.api.Test;

class ParityRouteTest {
  @Test
  void onlyExactSdkAuthenticationCommandsAreObserved() {
    String executable = "/usr/local/bin/databricks";
    var callers = Set.of(
        "com.databricks.sdk.core.DatabricksCliCredentialsProvider",
        "com.databricks.sdk.core.CliTokenSource");
    String version = "/bin/bash -c " + executable + " version --output json";
    String token = "/bin/bash -c " + executable + " auth token --profile test --force-refresh";
    assertTrue(ParityRoute.databricksAuthentication(version, executable, callers));
    assertTrue(ParityRoute.databricksAuthentication(token, executable, callers));
    assertEquals(java.util.List.of("databricks", "auth", "token", "--profile", "<redacted>", "--force-refresh"),
        ParityRoute.authenticationReceipt(token));
    assertEquals(java.util.List.of("databricks", "version", "--output", "json"),
        ParityRoute.authenticationReceipt(version));
    for (String altered : new String[] {
        token + "; echo unsafe", token + " | cat", token.replace("test", "$(id)"),
        token.replace("test", "${HOME}"), token.replace("test", "`id`"),
        token.replace("/bin/bash", "/bin/sh"), token.replace("/usr/local/bin", "/tmp"),
        token.replace("auth token", "jobs run-now"), token.replace(" --force-refresh", "")})
      assertFalse(ParityRoute.databricksAuthentication(altered, executable, callers), altered);
    assertFalse(ParityRoute.databricksAuthentication(token, executable, Set.of()));
    assertFalse(ParityRoute.databricksAuthentication(token, executable,
        Set.of("com.databricks.sdk.core.DatabricksCliCredentialsProvider")));
    assertFalse(ParityRoute.databricksAuthentication(version, "/tmp/databricks;id", callers));
  }
}
