// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.List;

/** The schemas the tests validate against. */
final class Schemas {
  private Schemas() {}

  /** S0's shop base: users, and orders linked to them. */
  static QuerySchema shop() {
    return new QuerySchema(
        List.of(
            new QuerySchema.Table(
                "orders",
                List.of(
                    new QuerySchema.Column("order_id", "int", "ID"),
                    new QuerySchema.Column("user_id", "int", "ID"),
                    new QuerySchema.Column("price", "float", "numerical"),
                    new QuerySchema.Column("ts", "time", "timestamp")),
                "order_id",
                "ts",
                null),
            new QuerySchema.Table(
                "users",
                List.of(
                    new QuerySchema.Column("user_id", "int", "ID"),
                    new QuerySchema.Column("age", "int", "numerical"),
                    new QuerySchema.Column("country", "string", "categorical")),
                "user_id",
                null,
                null)),
        List.of(
            List.of(
                new QuerySchema.ColumnRef("orders", "user_id"),
                new QuerySchema.ColumnRef("users", "user_id"))));
  }
}
