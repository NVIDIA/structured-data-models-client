// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
import java.util.List;
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
  @Test
  void theHostReportsItsSqlWarehouseProviderButNoSparkProvider() {
    assertEquals(java.util.List.of("duckdb", "sqlite", "postgres", "databricks-sql"),
        Host.PROVIDERS);
    assertFalse(Host.PROVIDERS.contains("databricks-spark"));
  }

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
        Map.of("provider", "sql", "dialect", "databricks", "namespace", Arrays.asList("main", "sales"),
            "timestamp_encoding", "microseconds"),
        both.choice());
    assertInstanceOf(JdbcProvider.class, both.provider());
    assertEquals(
        Arrays.asList(null, "sales"),
        Providers.of(DatabricksSource.of(connection, null, "sales")).choice().get("namespace"));
    assertEquals(
        Arrays.asList("main", null),
        Providers.of(DatabricksSource.of(connection, "main", null)).choice().get("namespace"));
    Map<String, Object> current = Providers.of(DatabricksSource.of(connection)).choice();
    assertEquals(Map.of("provider", "sql", "dialect", "databricks",
        "timestamp_encoding", "microseconds"), current);
    assertFalse(current.containsKey("namespace"));
    DatabricksSource source = DatabricksSource.of(connection, "main", null);
    assertEquals(connection, source.connection());
    assertEquals("main", source.catalog().orElseThrow());
    assertTrue(source.schema().isEmpty());
  }

  @Test
  void aMetricViewAddsPlainConfigurationAndKeepsTheOriginalSource() {
    DatabricksSource source = DatabricksSource.of(connection(), "main", "sales");
    DatabricksSource view = source.semanticModel("revenue", true);
    assertEquals(source.connection(), view.connection());
    assertFalse(Providers.of(source).choice().containsKey("view"));
    assertEquals(Map.of("name", "revenue", "trust_expressions", true),
        Providers.of(view).choice().get("view"));
    assertEquals(Map.of("name", "revenue", "trust_expressions", false),
        Providers.of(source.semanticModel("revenue", false)).choice().get("view"));
    assertEquals(Arrays.asList("main", "sales"), Providers.of(view).choice().get("namespace"));
    assertThrows(NullPointerException.class, () -> source.semanticModel(null, false));
  }

  @Test
  void aMetricViewsWarningsAndErrorsCanNameTheSource() {
    DatabricksSource view = DatabricksSource.of(connection(), "main", "sales")
        .semanticModel("revenue", false);
    Map<String, java.util.function.Function<Map<String, Object>, Object>> held = Providers.held(view);
    assertEquals(view, held.get("source").apply(Map.of()));
    assertEquals(view, held.get("semantic_model").apply(Map.of()));
    assertFalse(Providers.held(DatabricksSource.of(connection())).containsKey("semantic_model"));
    String said = Wording.warnings(java.util.List.of(Map.of(
        "code", "VIEW_READ_IN_PART", "sentence", "view_materialization_ignored")), held).get(0).text();
    assertTrue(said.contains("revenue"), said);
    assertTrue(said.contains("has what a graph cannot hold"), said);
    assertFalse(said.contains("cannot fill"), said);
  }

  @Test
  void aDriverDiagnosticDoesNotEscapeTheCoreProviderError() {
    TestOps.required();
    Host.Provider provider = (request, keys, allocator) -> {
      throw new java.sql.SQLException("secret_driver_text", "42501", 17);
    };
    assertEquals("secret_driver_text",
        provider.failure(new java.sql.SQLException("secret_driver_text")).get("driver_message"));
    StructuredDataException error = assertThrows(StructuredDataException.class,
        () -> Host.run(Map.of("op", "test", "test", "fetch", "fetch", 1),
            Map.of(1, TestOps.json(TestOps.fetch(false))), new Host.Call(provider, null, null)));
    assertEquals("PROVIDER_ERROR", error.code());
    assertFalse(error.getMessage().contains("secret_driver_text"));
    assertFalse(error.details().toString().contains("secret_driver_text"));
    assertEquals("42501", error.details().get("sqlstate"));
    assertEquals("17", error.details().get("vendor_code"));
  }

  private static Map<String, Object> fetchFailure(Object diagnostic, String state) {
    Map<String, Object> facts = Map.of(
        "kind", "data_source", "retryable", true, "sqlstate", state,
        "vendor_code", "1003", "error_class", "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        "driver_message", diagnostic);
    return Map.of("message", "fetch_reply", "id", 71L,
        "outcome", Map.of("outcome", "error", "error", facts));
  }

  private static Map<String, Object> withoutDiagnostic(Map<String, Object> reply) {
    Map<?, ?> outcome = (Map<?, ?>) reply.get("outcome");
    Map<?, ?> error = (Map<?, ?>) outcome.get("error");
    Map<String, Object> facts = new LinkedHashMap<>();
    error.forEach((key, value) -> facts.put((String) key, value));
    facts.remove("driver_message");
    return Map.of("message", reply.get("message"), "id", reply.get("id"),
        "outcome", Map.of("outcome", "error", "error", facts));
  }

  @Test
  void optionalDiagnosticsUseTheCompleteEncodedReplyBudget() throws Exception {
    int overhead = Json.write(fetchFailure("", "42703")).length;
    String fits = "x".repeat(Host.CONTROL_CAP - overhead);
    Map<String, Object> exact = fetchFailure(fits, "42703");
    assertEquals(Host.CONTROL_CAP, Json.write(exact).length);
    assertArrayEquals(Json.write(exact), Host.fetchControl(exact));
    Map<String, Object> past = fetchFailure(fits + "x", "42703");
    assertEquals(Host.CONTROL_CAP + 1, Json.write(past).length);
    assertEquals(withoutDiagnostic(past), Json.read(Host.fetchControl(past)));
    assertEquals(fits + "x", ((Map<?, ?>) ((Map<?, ?>) past.get("outcome"))
        .get("error")).get("driver_message"), "caller-owned immutable facts are unchanged");
  }

  @Test
  void escapedAndMultibyteDiagnosticsUseEncodedBytesNotCharacters() throws Exception {
    for (String text : List.of("é".repeat(30_000), "\\\n".repeat(20_000),
        "é".repeat(40_000), "\uD83D\uDE00".repeat(17_000), "x".repeat(100_000))) {
      Map<String, Object> original = fetchFailure(text, "42703");
      Object expected = Json.write(original).length <= Host.CONTROL_CAP
          ? original : withoutDiagnostic(original);
      byte[] encoded = Host.fetchControl(original);
      assertTrue(encoded.length <= Host.CONTROL_CAP);
      assertEquals(expected, Json.read(encoded));
    }
  }

  @Test
  void oversizedMalformedTextStillFailsTheExistingJsonValidation() {
    for (String text : List.of("\uD800", "\uDC00", "x".repeat(100_000) + "\uD800",
        "\uDC00" + "x".repeat(100_000), "\uD83D\uDE00".repeat(40_000) + "\uD800")) {
      Map<String, Object> original = fetchFailure(text, "42703");
      StructuredDataException ordinary = assertThrows(StructuredDataException.class,
          () -> Host.control(original));
      StructuredDataException fetched = assertThrows(StructuredDataException.class,
          () -> Host.fetchControl(original));
      assertEquals(ordinary.code(), fetched.code());
      assertEquals(ordinary.getMessage(), fetched.getMessage());
      assertTrue(fetched.getMessage().contains("unpaired surrogate"));
    }
  }

  @Test
  void onlyTheOptionalStringDiagnosticMayBeOmitted() throws Exception {
    Map<String, Object> numbered = fetchFailure(17L, "42703");
    assertArrayEquals(Host.control(numbered), Host.fetchControl(numbered));
    for (Object value : List.of(new Object(), List.of("x".repeat(100_000)))) {
      assertThrows(StructuredDataException.class,
          () -> Host.fetchControl(fetchFailure(value, "42703")));
    }
    assertThrows(StructuredDataException.class,
        () -> Host.fetchControl(fetchFailure("x".repeat(100_000), "s".repeat(100_000))));
    Map<String, Object> data = Map.of("message", "fetch_reply", "id", 71L,
        "outcome", Map.of("outcome", "data", "data", 1L));
    assertArrayEquals(Host.control(data), Host.fetchControl(data));
    assertThrows(StructuredDataException.class,
        () -> Host.control(fetchFailure("x".repeat(100_000), "42703")),
        "ordinary control messages retain their strict cap");
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
