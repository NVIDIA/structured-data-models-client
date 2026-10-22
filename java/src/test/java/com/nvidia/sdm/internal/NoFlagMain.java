// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.PredictiveQuery;
import com.nvidia.sdm.Profile;
import com.nvidia.sdm.QuerySchema;
import com.nvidia.sdm.StructuredDataException;
import java.util.List;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * A JVM started without Arrow Java's flag: validation, which crosses no Arrow, works, and the
 * first Arrow exchange is refused by name.
 */
public final class NoFlagMain {
  private NoFlagMain() {}

  public static void main(String[] arguments) throws Exception {
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
    System.out.println("validated accepted=" + accepted);
    try (BufferAllocator allocator = new RootAllocator();
        ArrowReader empty =
            new ArrowReader(allocator) {
              @Override
              public boolean loadNextBatch() {
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected org.apache.arrow.vector.types.pojo.Schema readSchema() {
                return new org.apache.arrow.vector.types.pojo.Schema(List.of());
              }
            }) {
      ArrowExchange.export(allocator, empty).close();
      System.out.println("exported");
    } catch (StructuredDataException refused) {
      System.out.println(refused.code() + " " + refused.details().get("flag"));
    }
  }
}
