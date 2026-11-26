// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.List;
import java.util.Optional;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;

/**
 * A custom source's own exception reaches the caller on the {@code PROVIDER_ERROR} it fails with,
 * as {@link StructuredDataException#sourceError()}, so its message (which input failed, say) is not
 * lost; the error's message, details and stack trace keep only what the core says, as for every
 * source.
 */
class SourceFailureTest {

  static final IllegalArgumentException FAILURE =
      new IllegalArgumentException("cannot parse date '2025-13-01' in column ts");

  /** The documented in-memory source, failing every scan with {@link #FAILURE}. */
  static final class FailingScans implements Source {
    private final Source tables = new InMemoryProviderExampleTest.InMemorySource();

    @Override
    public Capabilities capabilities() {
      return tables.capabilities();
    }

    @Override
    public ArrowReader answer(
        DataRequest request, Optional<Schema> expected, BufferAllocator allocator) throws Exception {
      if (request instanceof DataRequest.Scan) throw FAILURE;
      return tables.answer(request, expected, allocator);
    }
  }

  @Test
  void theSourcesExceptionIsOnTheProviderErrorAndNotInItsStackTrace() {
    StructuredDataException error =
        assertThrows(
            StructuredDataException.class,
            () -> {
              var schema =
                  com.nvidia.sdm.Schema.fromSource(
                          new FailingScans(),
                          FromSourceOptions.builder()
                              .table("users")
                              .table("orders")
                              .primaryKey("users", "id")
                              .primaryKey("orders", "order_id")
                              .timeColumn("orders", "placed")
                              .build())
                      .withRelationships(
                          List.of(new Relationship("orders", "user_id", "users", "id")));
              PredictiveQuery.parse(
                      "PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.id")
                  .materialize(
                      schema, MaterializeOptions.builder().indices(List.of(2L, 3L)).build());
            });
    assertEquals("PROVIDER_ERROR", error.code());
    assertEquals(IllegalArgumentException.class.getName(), error.details().get("driver_class"));
    assertSame(FAILURE, error.sourceError().orElseThrow());
    assertNull(error.getCause());
    assertFalse(error.getMessage().contains("2025-13-01"));
    assertFalse(error.details().toString().contains("2025-13-01"));
    StringWriter trace = new StringWriter();
    error.printStackTrace(new PrintWriter(trace));
    assertFalse(trace.toString().contains("2025-13-01"));
  }
}
