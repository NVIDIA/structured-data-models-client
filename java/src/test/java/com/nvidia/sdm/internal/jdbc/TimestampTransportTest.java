// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nvidia.sdm.DatabricksSource;
import com.nvidia.sdm.SourceAccess;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Native JDBC requests lossless transfer for both instants and timezone-free walls. */
class TimestampTransportTest {
  @Test
  void theCoreChoosesInstantAndWallEncodingFromActualWarehouseTypes() {
    FakeJdbc.Database database = new FakeJdbc.Database(text -> List.of());
    var chosen = SourceAccess.sqlChoice(DatabricksSource.of(database.connection()));
    assertEquals("microseconds", chosen.get("timestamp_encoding"));
    assertEquals(List.of(), database.ran, "choosing a source must not mutate its session");
  }
}
