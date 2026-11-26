// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
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

  /** The documented in-memory source, each answer passed through {@code answered}. */
  static Source answering(
      Capabilities capabilities, Function<DataRequest, ArrowReaderAnswer> answered) {
    Source tables = new InMemoryProviderExampleTest.InMemorySource();
    return new Source() {
      @Override
      public Capabilities capabilities() {
        return capabilities;
      }

      @Override
      public ArrowReader answer(
          DataRequest request, Optional<Schema> expected, BufferAllocator allocator)
          throws Exception {
        return answered
            .apply(request)
            .answer(tables.answer(request, expected, allocator), allocator);
      }
    };
  }

  /** What a source answers with, given the in-memory source's answer. */
  interface ArrowReaderAnswer {
    ArrowReader answer(ArrowReader tables, BufferAllocator allocator) throws Exception;
  }

  /** {@code reader}, failing with {@code failure} once its batches are read. */
  static ArrowReader failingOnRead(
      ArrowReader reader, RuntimeException failure, BufferAllocator allocator) {
    return new ArrowReader(allocator) {
      @Override
      public org.apache.arrow.vector.VectorSchemaRoot getVectorSchemaRoot() throws IOException {
        return reader.getVectorSchemaRoot();
      }

      @Override
      public boolean loadNextBatch() {
        throw failure;
      }

      @Override
      public long bytesRead() {
        return 0;
      }

      @Override
      public void close() throws IOException {
        reader.close();
      }

      @Override
      protected void closeReadSource() throws IOException {
        reader.close();
      }

      @Override
      protected Schema readSchema() throws IOException {
        return reader.getVectorSchemaRoot().getSchema();
      }
    };
  }

  static StructuredDataException materialized(Source source) {
    return assertThrows(
        StructuredDataException.class,
        () -> {
          var schema =
              com.nvidia.sdm.Schema.fromSource(
                      source,
                      FromSourceOptions.builder()
                          .table("users")
                          .table("orders")
                          .primaryKey("users", "id")
                          .primaryKey("orders", "order_id")
                          .timeColumn("orders", "placed")
                          .build())
                  .withRelationships(
                      List.of(new Relationship("orders", "user_id", "users", "id")));
          PredictiveQuery.parse("PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.id")
              .materialize(schema, MaterializeOptions.builder().indices(List.of(2L, 3L)).build());
        });
  }

  @Test
  void theSourcesExceptionIsOnTheProviderErrorAndNotInItsStackTrace() {
    StructuredDataException error = materialized(new FailingScans());
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

  @Test
  void aLazyReadersFailureIsCarriedAsAnAnswersIs() {
    Source source =
        answering(
            new InMemoryProviderExampleTest.InMemorySource().capabilities(),
            request ->
                (tables, allocator) ->
                    request instanceof DataRequest.Scan
                        ? failingOnRead(tables, FAILURE, allocator)
                        : tables);
    StructuredDataException error = materialized(source);
    assertEquals("PROVIDER_ERROR", error.code());
    assertEquals(IllegalArgumentException.class.getName(), error.details().get("driver_class"));
    assertSame(FAILURE, error.sourceError().orElseThrow());
    assertNull(error.getCause());
  }

  @Test
  void anEarlierStepsFailureIsNotCarriedOntoALaterStepsError() {
    // The core reads a failed read of a table's keys as none and carries on; the scan that fails
    // later is the failure the error carries.
    IllegalStateException keys = new IllegalStateException("the keys could not be read");
    AtomicBoolean keysAsked = new AtomicBoolean();
    Source source =
        answering(
            new InMemoryProviderExampleTest.InMemorySource()
                .capabilities()
                .toBuilder()
                .keysOptional(true)
                .build(),
            request ->
                (tables, allocator) -> {
                  if (request instanceof DataRequest.Describe describe
                      && describe.part() == DataRequest.DescribePart.KEYS) {
                    tables.close();
                    keysAsked.set(true);
                    throw keys;
                  }
                  if (request instanceof DataRequest.Scan) {
                    tables.close();
                    throw FAILURE;
                  }
                  return tables;
                });
    StructuredDataException error = materialized(source);
    assertTrue(keysAsked.get());
    assertEquals("PROVIDER_ERROR", error.code());
    assertEquals(IllegalArgumentException.class.getName(), error.details().get("driver_class"));
    assertSame(FAILURE, error.sourceError().orElseThrow());
  }
}
