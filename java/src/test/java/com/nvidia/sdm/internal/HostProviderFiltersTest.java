// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import com.nvidia.sdm.provider.FailureFacts;
import java.math.BigInteger;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.junit.jupiter.api.Test;

/** A provider's value filters: declared on the wire, and read back as typed filters. */
class HostProviderFiltersTest {

  @Test
  void declaredFilterKindsGoOnTheWireAndNoneLeavesTheFieldOut() {
    Capabilities declared =
        Capabilities.builder()
            .filters(DataRequest.FilterKind.IN, DataRequest.FilterKind.COMPARE)
            .build();
    assertEquals(List.of("compare", "in"), HostProviders.capabilities(declared).get("filters"));

    // Capabilities declaring no value filter leave the field out.
    Capabilities none = Capabilities.builder().maxSeed(1).build();
    assertEquals(Set.of(), none.filters());
    assertFalse(HostProviders.capabilities(none).containsKey("filters"));
  }

  @Test
  void valueFiltersAreReadAsTypedFiltersNestedAndAll() {
    Map<String, Object> age =
        Map.of(
            "filter", "compare",
            "column", "age",
            "op", "gt",
            "value", Map.of("t", "i64", "v", "76"));
    Map<String, Object> wire =
        Map.of(
            "request", "scan",
            "table", Map.of("source", List.of("users")),
            "columns", List.of("id"),
            "filters",
                List.of(
                    Map.of("filter", "not_null", "column", "id"),
                    Map.of(
                        "filter",
                        "any_of",
                        "filters",
                        List.of(
                            age,
                            Map.of("filter", "is_null", "column", "age"),
                            Map.of(
                                "filter",
                                "all_of",
                                "filters",
                                List.of(
                                    Map.of(
                                        "filter",
                                        "in",
                                        "column",
                                        "name",
                                        "values",
                                        List.of(Map.of("t", "utf8", "v", "ann")))))))));
    DataRequest.Scan scan =
        (DataRequest.Scan) HostProviders.Requests.read(wire, at -> null);
    assertEquals(
        List.of(
            new DataRequest.NotNull("id"),
            new DataRequest.AnyOf(
                List.of(
                    new DataRequest.Compare(
                        "age",
                        DataRequest.CompareOp.GT,
                        new DataRequest.Value(76L, new ArrowType.Int(64, true))),
                    new DataRequest.IsNull("age"),
                    new DataRequest.AllOf(
                        List.of(new DataRequest.In(
                                "name",
                                List.of(
                                    new DataRequest.Value("ann", ArrowType.Utf8.INSTANCE)))))))),
        scan.filters());
  }

  @Test
  void hasRowsIsReadWithItsTableAndNestedFilters() {
    Map<String, Object> wire =
        Map.of(
            "request", "scan",
            "table", Map.of("source", List.of("users")),
            "columns", List.of("id"),
            "filters",
                List.of(
                    Map.of(
                        "filter", "has_rows",
                        "column", "id",
                        "table", Map.of("source", List.of("orders")),
                        "key", "uid",
                        "filters", List.of(Map.of("filter", "not_null", "column", "amount")),
                        "at_least", 2)));
    DataRequest.Scan scan = (DataRequest.Scan) HostProviders.Requests.read(wire, at -> null);
    DataRequest.HasRows hasRows = (DataRequest.HasRows) scan.filters().get(0);
    assertEquals("id", hasRows.column());
    assertEquals(List.of("orders"), hasRows.table().parts());
    assertEquals("uid", hasRows.key());
    assertEquals(List.of(new DataRequest.NotNull("amount")), hasRows.filters());
    assertEquals(2L, hasRows.atLeast());
  }

  @Test
  void aSourcesDeclaredFiltersReachTheCore() {
    com.nvidia.sdm.Source source =
        new com.nvidia.sdm.Source() {
          @Override
          public Capabilities capabilities() {
            return Capabilities.builder().filters(DataRequest.FilterKind.HAS_ROWS).build();
          }

          @Override
          public org.apache.arrow.vector.ipc.ArrowReader answer(
              DataRequest request,
              Optional<org.apache.arrow.vector.types.pojo.Schema> expected,
              org.apache.arrow.memory.BufferAllocator allocator) {
            throw new UnsupportedOperationException();
          }
        };
    @SuppressWarnings("unchecked")
    Map<String, Object> declared =
        (Map<String, Object>) HostProviders.chosen(source).choice().get("capabilities");
    assertEquals(List.of("has_rows"), declared.get("filters"));
  }

  /** As Python's {@code test_capabilities_refuse_what_no_provider_declares}. */
  @Test
  void capabilitiesRefuseWhatNoProviderDeclares() {
    assertThrows(IllegalArgumentException.class, () -> Capabilities.builder().maxKeysPerRequest(0));
    assertThrows(IllegalArgumentException.class, () -> Capabilities.builder().maxSeed(-1));
    // The core's seed is an unsigned 64-bit integer.
    BigInteger largest = BigInteger.TWO.pow(64).subtract(BigInteger.ONE);
    assertEquals(largest, Capabilities.builder().maxSeed(largest).build().maxSeed());
    assertThrows(
        IllegalArgumentException.class, () -> Capabilities.builder().maxSeed(largest.add(BigInteger.ONE)));
    assertThrows(IllegalArgumentException.class, () -> Capabilities.builder().engine(""));
    assertThrows(IllegalArgumentException.class, () -> new Capabilities.Fallback(0, 1));
    assertEquals(
        new FailureFacts(true, false, Optional.of(FailureFacts.Unusable.MISSING), false, Optional.empty()),
        FailureFacts.builder().notFound().unusable(FailureFacts.Unusable.MISSING).build());
  }
}
