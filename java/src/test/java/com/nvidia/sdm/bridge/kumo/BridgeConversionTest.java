// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.bridge.kumo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.kumo.data.Expr;
import ai.kumo.graph.Column;
import java.util.List;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.junit.jupiter.api.Test;

/** The bridge's conversions for what the end-to-end test cannot build. */
class BridgeConversionTest {

  @Test
  void aColumnTypeTheStockPrototypeLacksIsReadAsText() {
    // A modified prototype may add a constant (an id type, say); this build cannot name it.
    assertEquals("STRING", Answers.readAs("ID"));
    for (Column.Type known : Column.Type.values()) {
      assertEquals(known, Answers.readAs(known));
    }
    assertEquals(ArrowType.Utf8.INSTANCE, Answers.field("id", Column.Type.valueOf(Answers.readAs("ID"))).getType());
  }

  @Test
  void inclusiveAndExclusiveTimeBoundsUseOnlyGreaterThanAndAtMost() {
    Expr time = new Expr.Call(Expr.Function.TIME_MICROS, List.of(new Expr.Field("ts")));
    assertEquals(new Expr.Compare(time, Expr.Op.GT, new Expr.Literal(9L)), QueryProvider.after(time, 10L, true));
    assertEquals(new Expr.Compare(time, Expr.Op.GT, new Expr.Literal(10L)), QueryProvider.after(time, 10L, false));
    assertEquals(new Expr.Compare(time, Expr.Op.LE, new Expr.Literal(10L)), QueryProvider.until(time, 10L, true));
    assertEquals(new Expr.Compare(time, Expr.Op.LE, new Expr.Literal(9L)), QueryProvider.until(time, 10L, false));
  }
}
