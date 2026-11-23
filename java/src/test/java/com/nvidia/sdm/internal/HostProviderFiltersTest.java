// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.provider.Capabilities;
import com.nvidia.sdm.provider.DataRequest;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;
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

    // The constructor from before value filters declares none, and the wire leaves them out.
    Capabilities older =
        new Capabilities(
            Set.of(),
            OptionalLong.empty(),
            Capabilities.Sampling.SEEDED_STABLE,
            1,
            Optional.empty(),
            Optional.empty());
    assertEquals(Set.of(), older.filters());
    assertFalse(HostProviders.capabilities(older).containsKey("filters"));
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
                    new DataRequest.Compare("age", DataRequest.CompareOp.GT, 76L),
                    new DataRequest.IsNull("age"),
                    new DataRequest.AllOf(
                        List.of(new DataRequest.In("name", List.of("ann"))))))),
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
  void aProviderSourceCarriesItsProviderAndItsDeclaredFiltersReachTheCore() {
    com.nvidia.sdm.provider.DataProvider provider =
        new com.nvidia.sdm.provider.DataProvider() {
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
    com.nvidia.sdm.ProviderSource source = com.nvidia.sdm.ProviderSource.of(provider);
    assertEquals(provider, source.provider());
    assertEquals(new com.nvidia.sdm.ProviderSource(provider), source);
    assertEquals(new com.nvidia.sdm.ProviderSource(provider).hashCode(), source.hashCode());
    assertTrue(source.toString().startsWith("ProviderSource["));
    @SuppressWarnings("unchecked")
    Map<String, Object> declared =
        (Map<String, Object>) HostProviders.chosen(source).choice().get("capabilities");
    assertEquals(List.of("has_rows"), declared.get("filters"));
  }
}
