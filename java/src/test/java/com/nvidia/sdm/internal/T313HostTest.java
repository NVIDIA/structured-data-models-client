// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.DatabricksSource;
import com.nvidia.sdm.ModelEndpointException;
import com.nvidia.sdm.StructuredDataException;
import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import java.lang.reflect.Proxy;
import java.sql.Connection;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * The Java host's Databricks pieces (T3.13): a {@link DatabricksSource} names the dialect and the
 * namespace Python's {@code for_source} names for its {@code DatabricksSource}, the catalog and
 * the schema, each null where the connection's current one is meant; and the core's {@code
 * ENDPOINT_REFUSED} is raised as {@link ModelEndpointException} under the endpoint's status and
 * its own code, none where it has none, as Python raises {@code NimRequestError}.
 */
class T313HostTest {
  private static Connection connection() {
    return (Connection)
        Proxy.newProxyInstance(
            T313HostTest.class.getClassLoader(),
            new Class<?>[] {Connection.class},
            (self, method, args) ->
                switch (method.getName()) {
                  case "toString" -> "connection";
                  case "hashCode" -> 1;
                  case "equals" -> self == args[0];
                  default -> throw new AssertionError(method.getName());
                });
  }

  @Test
  void aDatabricksSourceNamesItsDialectAndNamespace() {
    Connection connection = connection();
    Providers.Chosen both = Providers.of(DatabricksSource.of(connection, "main", "sales"));
    assertEquals(
        Map.of("provider", "sql", "dialect", "databricks", "namespace", Arrays.asList("main", "sales")),
        both.choice());
    assertInstanceOf(JdbcProvider.class, both.provider());
    assertEquals(
        Arrays.asList(null, "sales"),
        Providers.of(DatabricksSource.of(connection, null, "sales")).choice().get("namespace"));
    assertEquals(
        Arrays.asList("main", null),
        Providers.of(DatabricksSource.of(connection, "main", null)).choice().get("namespace"));
    Map<String, Object> current = Providers.of(DatabricksSource.of(connection)).choice();
    assertEquals(Map.of("provider", "sql", "dialect", "databricks"), current);
    assertFalse(current.containsKey("namespace"));
    DatabricksSource source = DatabricksSource.of(connection, "main", null);
    assertEquals(connection, source.connection());
    assertEquals("main", source.catalog().orElseThrow());
    assertTrue(source.schema().isEmpty());
  }

  /** The core's error of {@code sentence} under {@code status} and {@code code}. */
  private static StructuredDataException raised(String sentence, int status, String code) {
    TestOps.required();
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("status", Map.of("t", "u16", "v", Integer.toString(status)));
    fields.put("endpoint", Map.of("t", "utf8", "v", "kumo"));
    if (code != null) {
      fields.put("refused_code", Map.of("t", "utf8", "v", code));
      fields.put("endpoint_code", Map.of("t", "utf8", "v", code));
      fields.put("provider_code", Map.of("t", "utf8", "v", code));
    }
    Map<String, Object> error = new LinkedHashMap<>();
    error.put("kind", "model_endpoint");
    error.put("code", "ENDPOINT_REFUSED");
    error.put("sentence", sentence);
    error.put("message", "m");
    error.put("retryable", false);
    error.put("fields", fields);
    Map<Integer, Object> attachments = Map.of(1, TestOps.json(error));
    return assertThrows(
        StructuredDataException.class,
        () -> Host.run(Map.of("op", "test", "test", "fail", "error", 1), attachments));
  }

  @Test
  void anEndpointsRefusalIsRaisedUnderItsStatusAndItsOwnCode() {
    Object[][] cases = {
      {"serving_callers_fault", 503, "INVALID_REQUEST"},
      {"serving_refused", 400, "INVALID_SCHEMA"},
      {"serving_failed_coded", 404, "RESOURCE_DOES_NOT_EXIST"},
      {"serving_failed", 500, null},
    };
    for (Object[] each : cases) {
      StructuredDataException made = raised((String) each[0], (Integer) each[1], (String) each[2]);
      ModelEndpointException refused = assertInstanceOf(ModelEndpointException.class, made);
      assertEquals(each[1], refused.status(), (String) each[0]);
      assertEquals(each[2], refused.code(), (String) each[0]);
      assertEquals("kumo", refused.details().get("endpoint"));
      assertFalse(refused.details().containsKey("status"));
      assertFalse(refused.details().containsKey("refused_code"));
      assertEquals("model_endpoint", refused.kind());
      assertTrue(refused.getMessage().startsWith("serving endpoint \"kumo\""), refused.getMessage());
    }
    assertNull(raised("serving_failed", 503, null).code());
    assertEquals(
        "serving endpoint \"kumo\" refused the request (INVALID_SCHEMA)",
        raised("serving_refused", 400, "INVALID_SCHEMA").getMessage());
  }
}
