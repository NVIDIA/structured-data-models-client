// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/** Native query transport, warning ownership, public text retention and error mapping. */
class BindingTest {
    private static final String BASE = "s0.accept.binary-classification";
    private static final Set<String> ACCEPTED = Set.of(
        BASE, "s0.accept.multiclass-classification", "s0.accept.sum-window",
        "s0.boundary.deepest", "s0.boundary.depth-999", "s0.boundary.digits-4300",
        "s0.boundary.escape-lone-surrogate", "s0.boundary.escape-surrogate-pair",
        "s0.boundary.escapes", "s0.boundary.start-past-64kb");
    private static final Map<String, String> REFUSED = Map.ofEntries(
        Map.entry("s0.boundary.digits-4301", "PQL_VALUE_INVALID_LITERAL"),
        Map.entry("s0.boundary.escape-truncated", "PQL_VALUE_INVALID_LITERAL"),
        Map.entry("s0.refuse.date-impossible", "PQL_VALUE_INVALID_LITERAL"),
        Map.entry("s0.refuse.join-event-filter", "PQL_JOIN_EVENT_COLUMN_IN_ENTITY_FILTER"),
        Map.entry("s0.refuse.model-implicit-join", "PQL_MODEL_NO_IMPLICIT_JOIN"),
        Map.entry("s0.refuse.parse-syntax", "PQL_PARSE_SYNTAX"),
        Map.entry("s0.refuse.target-foreign-key", "PQL_TARGET_AGGREGATION_ON_FOREIGN_KEY"),
        Map.entry("s0.refuse.task-forecast-zero", "PQL_TASK_FORECAST_NOT_POSITIVE"),
        Map.entry("s0.refuse.time-past-window", "PQL_TIME_PAST_NOT_ALLOWED"),
        Map.entry("s0.refuse.type-constant", "PQL_TYPE_CONSTANT_MISMATCH"),
        Map.entry("s0.refuse.unknown-column", "PQL_SCHEMA_UNKNOWN_COLUMN"),
        Map.entry("s0.refuse.value-int64", "PQL_VALUE_INT_OUT_OF_RANGE"));
    private static final Set<String> DATE_ERRORS = Set.of(
        "s0.boundary.date-long-field", "s0.boundary.date-packed");

    private static Map<String, Map<String, Object>> cases() throws IOException {
        Map<String, Map<String, Object>> cases = new LinkedHashMap<>();
        try (Stream<Path> paths = Files.list(Names.root().resolve("contract/cases/s0"))) {
            for (Path path : paths.filter(p -> p.toString().endsWith(".json")).sorted().toList()) {
                Map<String, Object> spec = Json.object(Json.read(Files.readString(path)));
                if ("query_attachment_72082".equals(spec.get("input_factory"))) {
                    assertEquals("s0.boundary.start-past-64kb", spec.get("id"));
                    Map<String, Object> input = Json.object(spec.get("input"));
                    assertEquals("<generated-query>", input.get("pql"));
                    String values = String.join(", ", java.util.Collections.nCopies(3000, "10.0, 11.5, 13.0, 14.5"));
                    String query = "PREDICT COUNT(orders.* WHERE orders.price IN (" + values
                        + "), 0, 30, days) FOR EACH users.user_id";
                    assertEquals(72082, query.length());
                    input.put("pql", query);
                    spec.put("input", input);
                    spec.remove("input_factory");
                }
                cases.put((String) spec.get("id"), spec);
            }
        }
        return cases;
    }

    private static Map<String, String> extraTexts() {
        return Map.of(
            "text.nul", "PREDICT users.country = 'a\u0000b' FOR EACH users.user_id",
            "text.non-bmp", "PREDICT users.country = '😀' FOR EACH users.user_id",
            "text.non-ascii", "PREDICT users.country = 'Zürich 東京' FOR EACH users.user_id",
            "text.escape-warning", "PREDICT users.country = 'a\\/' FOR EACH users.user_id");
    }

    private static void assertWarnings(Object run, boolean expected) throws Exception {
        Object taken = Names.call(run, "takeWarnings");
        if (!expected) {
            assertNull(taken);
            return;
        }
        assertTrue(taken != null, "warning channel was lost");
        try {
            byte[] control = (byte[]) Names.read(taken, "control", "getControl");
            Map<String, Object> message = Json.object(Json.read(new String(control, StandardCharsets.UTF_8)));
            if ("attached".equals(message.get("message"))) {
                int id = ((Number) message.get("attachment")).intValue();
                message = Json.object(Json.read(new String((byte[]) Names.call(taken, "bytes", id), StandardCharsets.UTF_8)));
            }
            assertEquals("warnings", message.get("message"));
            List<?> warnings = (List<?>) message.get("warnings");
            assertEquals(1, warnings.size());
            Map<String, Object> warning = Json.object(warnings.get(0));
            assertEquals("INVALID_STRING_ESCAPE", warning.get("code"));
            assertEquals("invalid_string_escape", warning.get("sentence"));
            assertEquals("a string literal uses a deprecated escape", warning.get("message"));
            Map<String, Object> fields = Json.object(warning.get("fields"));
            assertEquals("\\/", Json.object(fields.get("escape")).get("v"));
            for (String field : List.of("start_row", "end_row"))
                assertEquals("1", Json.object(fields.get(field)).get("v"));
            for (String field : List.of("start_col", "end_col"))
                assertEquals("24", Json.object(fields.get(field)).get("v"));
        } finally { Names.call(taken, "close"); }
        assertNull(Names.call(run, "takeWarnings"), "warnings must drain exactly once");
    }

    private static void validate(String name, Map<String, Object> spec, String text,
                                 String diagnostic, boolean dateError) throws Exception {
        Object run = Names.validation(spec, text);
        try {
            if (dateError) {
                try { Names.nextWhole(run); fail("invalid packed date accepted"); }
                catch (Exception raised) {
                    assertTrue(Names.isStructuredDataException(raised));
                    assertEquals("INVALID_REQUEST", Names.code(raised));
                    assertEquals("date_literal_packed", Names.detail(raised, "rule"));
                }
            } else {
                Map<String, Object> message = Json.object(Json.read((String) Names.nextWhole(run)[1]));
                assertEquals("step", message.get("message"));
                assertEquals("done", message.get("step"));
                Map<String, Object> result = Json.object(message.get("result"));
                assertEquals("validate_query", result.get("op"));
                assertEquals(diagnostic == null, result.get("accepted"));
                if (name.equals("text.nul") || name.equals("text.non-bmp") || name.equals("text.non-ascii")) {
                    // These queries already use canonical spelling. Inspect the text
                    // returned through JNI, not only the Java constructor's local copy.
                    Map<String, Object> query = Json.object(result.get("query"));
                    assertArrayEquals(text.toCharArray(), ((String) query.get("text")).toCharArray());
                }
                List<?> diagnostics = (List<?>) result.get("diagnostics");
                List<String> expectedCodes = diagnostic == null ? List.of() : List.of(diagnostic);
                List<String> severities = diagnostic == null ? List.of() : List.of("error");
                if (name.equals("s0.refuse.model-implicit-join")) {
                    // users.age belongs to the entity, so this also warns that its
                    // predicate belongs in FOR EACH rather than the target aggregation.
                    expectedCodes = List.of(diagnostic, "PQL_TARGET_FILTER_ONLY_ENTITY_COLUMNS");
                    severities = List.of("error", "warning");
                }
                assertEquals(expectedCodes, diagnostics.stream().map(d -> Json.object(d).get("code")).toList());
                assertEquals(severities, diagnostics.stream().map(d -> Json.object(d).get("severity")).toList());
            }
        } finally {
            try { assertWarnings(run, name.equals("s0.boundary.escapes") || name.equals("text.escape-warning")); }
            finally { Names.close(run); }
        }
    }

    @TestFactory
    Stream<DynamicTest> queryInputsPreserveValidationAndDiagnostics() throws IOException {
        Map<String, Map<String, Object>> cases = cases();
        Set<String> expected = new java.util.HashSet<>(ACCEPTED);
        expected.addAll(REFUSED.keySet()); expected.addAll(DATE_ERRORS);
        assertEquals(expected, cases.keySet(), "new cases need explicit boundary assertions");
        List<DynamicTest> tests = new ArrayList<>();
        cases.forEach((name, spec) -> tests.add(DynamicTest.dynamicTest(name,
            () -> validate(name, spec, (String) Json.object(spec.get("input")).get("pql"),
                           REFUSED.get(name), DATE_ERRORS.contains(name)))));
        extraTexts().forEach((name, text) -> tests.add(DynamicTest.dynamicTest(name,
            () -> validate(name, cases.get(BASE), text, null, false))));
        return tests.stream();
    }

    @Test
    void largeWarningsUseAttachmentsAndPreserveOrderUntilDrained() throws Exception {
        List<String> escapes = List.of("first", "x".repeat(70_000));
        List<Object> warnings = new ArrayList<>();
        for (String escape : escapes) {
            warnings.add(Map.of("code", "INVALID_STRING_ESCAPE", "sentence", "invalid_string_escape",
                "message", "deprecated escape", "fields", Map.of("escape", Map.of("t", "utf8", "v", escape))));
        }
        Object run = Names.startRun("{\"op\":\"test\",\"test\":\"warn\",\"warnings\":1}",
            Map.of(1, com.nvidia.sdm.internal.Json.write(warnings)));
        try {
            assertEquals("done", Json.object(Json.read((String) Names.nextWhole(run)[1])).get("step"));
            Object taken = Names.call(run, "takeWarnings");
            assertTrue(taken != null);
            try {
                byte[] control = (byte[]) Names.read(taken, "control", "getControl");
                Map<String, Object> envelope = Json.object(Json.read(new String(control, StandardCharsets.UTF_8)));
                assertEquals("attached", envelope.get("message"));
                byte[] bytes = (byte[]) Names.call(taken, "bytes", ((Number) envelope.get("attachment")).intValue());
                Map<String, Object> message = Json.object(Json.read(new String(bytes, StandardCharsets.UTF_8)));
                assertEquals("warnings", message.get("message"));
                List<?> received = (List<?>) message.get("warnings");
                assertEquals(escapes, received.stream().map(w -> Json.object(
                    Json.object(Json.object(w).get("fields")).get("escape")).get("v")).toList());
            } finally { Names.call(taken, "close"); }
            assertNull(Names.call(run, "takeWarnings"));
        } finally { Names.close(run); }
    }

    /** Authored code/kind/sentence inputs cover each supported native error mapping. */
    @TestFactory
    Stream<DynamicTest> everyErrorCodeReachesJavaWithItsCode() {
        String[][] errors = {
            new String[]{"ENDPOINT_REFUSED", "model_endpoint", "serving_refused"},
            new String[]{"INTERNAL_ERROR", "internal", "result_merge_empty"},
            new String[]{"INVALID_CONFIGURATION", "invalid_request", "invalid_configuration"},
            new String[]{"INVALID_MESSAGE", "protocol", "invalid_message"},
            new String[]{"INVALID_REQUEST", "invalid_request", "direct_empty"},
            new String[]{"INVALID_RESPONSE", "model_endpoint", "tabular_missing_row_index"},
            new String[]{"INVALID_SCHEMA", "invalid_request", "wire_reserved_one"},
            new String[]{"MISSING_EXTRA", "unsupported", "missing_extra"},
            new String[]{"PROVIDER_ERROR", "not_found", "provider_failed"},
            new String[]{"PROVIDER_SCHEMA_MISMATCH", "data_source", "provider_schema_mismatch"},
            new String[]{"REPLY_MISSING", "protocol", "reply_missing"},
            new String[]{"RUN_BUSY", "protocol", "run_busy"},
            new String[]{"RUN_CANCELLED", "cancelled", "run_cancelled"},
            new String[]{"RUN_FINISHED", "protocol", "run_finished"},
            new String[]{"RUN_POISONED", "protocol", "run_poisoned"},
            new String[]{"SESSIONS_UNSUPPORTED", "unsupported", "sessions_unsupported"},
            new String[]{"TRANSPORT_ERROR", "model_endpoint", "transport_failed"},
            new String[]{"UNKNOWN_MODEL", "not_found", "unknown_model"},
            new String[]{"UNKNOWN_REPLY", "protocol", "unknown_reply"},
            new String[]{"UNSUPPORTED_REQUEST", "unsupported", "direct_relational_frames"},
            new String[]{"UNSUPPORTED_TYPE", "unsupported", "unsupported_type"},
            new String[]{"VALUE_OUT_OF_RANGE", "unsupported", "lag_past_times"},
            new String[]{"WRONG_REPLY", "protocol", "wrong_reply"}
        };
        return Stream.of(errors).map(error -> DynamicTest.dynamicTest(error[0], () -> {
            byte[] input = com.nvidia.sdm.internal.Json.write(Map.of(
                    "kind", error[1], "code", error[0], "sentence", error[2],
                    "message", "native-free-text-sentinel", "retryable", false));
            Object run = Names.startRun(
                    "{\"op\":\"test\",\"test\":\"fail\",\"error\":1}", Map.of(1, input));
            try {
                Names.next(run);
                fail("the fail operation did not fail: " + error[0]);
            } catch (Exception raised) {
                assertTrue(Names.isStructuredDataException(raised));
                assertEquals(error[0], Names.code(raised));
                assertTrue(raised.getMessage() != null && !raised.getMessage().isBlank());
                assertTrue(!raised.getMessage().contains("native-free-text-sentinel"));
                if (error[0].equals("INVALID_REQUEST")) {
                    assertEquals("context has no rows, so there is nothing for the model to learn the answer from", raised.getMessage());
                }
                if (error[0].equals("RUN_BUSY")) {
                    assertEquals("the run was already in a call; a run takes one call at a time", raised.getMessage());
                }
            } finally {
                Names.close(run);
            }
        }));
    }

    @TestFactory
    Stream<DynamicTest> parsePreservesTextAndRefusesOnlyBlankInputs() throws IOException {
        Map<String, String> texts = new LinkedHashMap<>(extraTexts());
        cases().forEach((name, spec) -> texts.put(name, (String) Json.object(spec.get("input")).get("pql")));
        texts.put("java-lone-surrogate", "PREDICT users.country = '\ud800' FOR EACH users.user_id");
        List<DynamicTest> tests = new ArrayList<>();
        texts.forEach((name, text) -> tests.add(DynamicTest.dynamicTest(name, () -> {
            Object parsed = Names.call(Names.type(Names.QUERY), "parse", text);
            assertArrayEquals(text.toCharArray(), ((String) Names.read(parsed, "text", "getText")).toCharArray());
        })));
        for (String blank : List.of("", "   ", "\t\n ")) {
            tests.add(DynamicTest.dynamicTest("blank-" + blank.length(), () -> {
                try { Names.call(Names.type(Names.QUERY), "parse", blank); fail("blank text accepted"); }
                catch (Exception raised) {
                    assertTrue(Names.isStructuredDataException(raised));
                    assertEquals("INVALID_REQUEST", Names.code(raised));
                }
            }));
        }
        return tests.stream();
    }
}
