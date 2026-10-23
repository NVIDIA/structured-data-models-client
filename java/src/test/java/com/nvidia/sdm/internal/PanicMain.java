// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.PredictiveQuery;
import com.nvidia.sdm.Profile;
import com.nvidia.sdm.QuerySchema;
import com.nvidia.sdm.StructuredDataException;
import java.util.List;
import java.util.Map;

/**
 * The panic test's subprocess (exit 3 of T3.11): a panic inside the core, at a run's start and in
 * its body, reaches Java as an exception, and the JVM runs on to validate a query. It prints one
 * line per panic, then {@code survived}.
 */
public final class PanicMain {
  private PanicMain() {}

  public static void main(String[] arguments) throws Exception {
    for (String at : new String[] {"start", "body", "adapter_stack"}) {
      try {
        try (Run run =
            Run.start(
                TestOps.start("panic", Map.of("at", at, "payload", "boom")),
                Map.of())) {
          run.next().close();
        }
        System.out.println(at + ": no exception");
      } catch (NativeError failed) {
        StructuredDataException raised = failed;
        System.out.println(at + ": " + raised.code() + " panic=" + raised.details().get("panic"));
      }
    }
    QuerySchema schema =
        new QuerySchema(
            List.of(
                new QuerySchema.Table(
                    "users",
                    List.of(
                        new QuerySchema.Column("id", "int", "ID"),
                        new QuerySchema.Column("age", "float", "numerical")),
                    "id",
                    null,
                    null)),
            List.of());
    boolean accepted =
        PredictiveQuery.parse("PREDICT users.age FOR EACH users.id")
            .validate(schema, Profile.RELATIONAL)
            .accepted();
    System.out.println("survived accepted=" + accepted);
  }
}
