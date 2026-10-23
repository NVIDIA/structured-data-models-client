// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * T3.11's Java API and binding on every S0 case, held to what the reference
 * and Python's binding over the same core answer
 * ({@code contract/t311-java-vectors.json}, recorded by
 * {@code contract/tools/t311_java_record.py}).
 *
 * <p>Exit 1 (every S0 case on the Java host equal to the reference) is
 * {@code contract/tests/test_t311_java.py}'s, through the Java runner and
 * the comparator; these hold the pieces it rests on, per case, where a
 * failure names the case: the public {@code PredictiveQuery.parse} as the
 * reference's constructor, and the binding's answer to each S0 case's
 * validation as Python's binding gives it, byte for byte, which carries the
 * brief's text axis across JNI (non-ASCII, a NUL, a character outside the
 * BMP, escaped lone surrogates, a query past 64 KB). And the brief's
 * error axis: every contract error code reaching Java as the one
 * {@code StructuredDataException} with its code (choice 10).
 *
 * <p>These run on the binding's test-ops build, as sdm-py's binding tests
 * do (the {@code fail} operation is test-only), with the two runtime flags.
 */
class T311BindingTest {

    private static Map<String, Object> section(String name) throws IOException {
        return T311Json.object(T311Names.vectors().get(name));
    }

    private static String sha256(byte[] data) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data));
    }

    /** Every S0 case's id, from its file in contract/cases/s0/. */
    private static List<String> s0Cases() throws IOException {
        try (Stream<Path> files = Files.list(T311Names.root().resolve("contract/cases/s0"))) {
            return files.map(file -> file.getFileName().toString())
                    .filter(name -> name.endsWith(".json"))
                    .map(name -> name.substring(0, name.length() - ".json".length()))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    /** The vectors cover every S0 case: a case added since they were
     * recorded fails here until the recorder runs again. */
    @Test
    void theVectorsCoverEveryS0Case() throws IOException {
        List<String> cases = s0Cases();
        assertTrue(cases.size() >= 24, "contract/cases/s0 holds " + cases.size() + " cases");
        TreeSet<String> missing = new TreeSet<>(cases);
        missing.removeAll(T311Json.object(section("binding").get("validate")).keySet());
        assertEquals(new TreeSet<String>(), missing, "S0 cases with no binding vector");
        missing = new TreeSet<>(cases);
        missing.removeAll(T311Json.object(section("reference").get("constructor")).keySet());
        assertEquals(new TreeSet<String>(), missing, "S0 cases with no reference vector");
        assertEquals(
                T311Json.read(T311Names.FACTS),
                section("binding").get("facts"),
                "the facts the Java tests send are the recorder's");
    }

    /** Brief, input axes (text across JNI) and exit 1's footing: each S0
     * case's validate_query, and the texts no case holds, answered through
     * the Java binding with the same control bytes Python's binding
     * gives. */
    @TestFactory
    Stream<DynamicTest> eachS0CaseValidatesAsThroughPythonsBinding() throws IOException {
        Map<String, Object> validate = T311Json.object(section("binding").get("validate"));
        List<DynamicTest> tests = new ArrayList<>();
        for (Map.Entry<String, Object> each : validate.entrySet()) {
            Map<String, Object> vector = T311Json.object(each.getValue());
            tests.add(DynamicTest.dynamicTest(each.getKey(), () -> {
                Object run = T311Names.startRun(
                        (String) vector.get("operation"),
                        T311Names.attachments(T311Json.object(vector.get("attachments"))));
                try {
                    String error = (String) vector.get("error");
                    if (error == null) {
                        Object[] whole = T311Names.nextWhole(run);
                        byte[] control = (byte[]) whole[0];
                        Map<String, Object> result = T311Json.object(T311Json.object(
                                T311Json.read((String) whole[1])).get("result"));
                        assertEquals(vector.get("accepted"), result.get("accepted"), each.getKey());
                        assertEquals(vector.get("digest"), sha256(control),
                                each.getKey() + ": the done step's bytes differ from Python's binding's");
                    } else {
                        try {
                            T311Names.next(run);
                            fail(each.getKey() + ": no error; Python's binding raised " + error);
                        } catch (Exception raised) {
                            assertTrue(T311Names.isStructuredDataException(raised),
                                    each.getKey() + ": raised " + raised.getClass().getName());
                            assertEquals(error, T311Names.code(raised), each.getKey());
                        }
                    }
                } finally {
                    T311Names.close(run);
                }
            }));
        }
        return tests.stream();
    }

    /** Brief, input axes (errors) and choice 10: every contract error code
     * reaches Java as StructuredDataException carrying that code and saying
     * the core's sentence. */
    @TestFactory
    Stream<DynamicTest> everyErrorCodeReachesJavaWithItsCode() throws IOException {
        Map<String, Object> errors = T311Json.object(section("binding").get("errors"));
        assertTrue(errors.size() >= 22, "the vectors hold " + errors.size() + " codes");
        List<DynamicTest> tests = new ArrayList<>();
        for (Map.Entry<String, Object> each : errors.entrySet()) {
            Map<String, Object> vector = T311Json.object(each.getValue());
            tests.add(DynamicTest.dynamicTest(each.getKey(), () -> {
                Object run = T311Names.startRun(
                        "{\"op\":\"test\",\"test\":\"fail\",\"error\":1}",
                        Map.of(1, ((String) vector.get("error")).getBytes(StandardCharsets.UTF_8)));
                try {
                    T311Names.next(run);
                    fail(each.getKey() + ": the fail operation did not fail");
                } catch (Exception raised) {
                    assertTrue(T311Names.isStructuredDataException(raised),
                            each.getKey() + ": raised " + raised.getClass().getName());
                    assertEquals(vector.get("code"), T311Names.code(raised));
                    // The core's sentence, said from its shared template, never
                    // the core's own free text (T3.17).
                    String template = (String) vector.get("template");
                    assertNotEquals("t311", raised.getMessage(), each.getKey());
                    if (!template.contains("{")) {
                        assertEquals(template, raised.getMessage(), each.getKey());
                    }
                } finally {
                    T311Names.close(run);
                }
            }));
        }
        return tests.stream();
    }

    /** Brief, "Open questions" item 1: PredictiveQuery.parse is the
     * reference's constructor in Java's idiom. It takes every text the
     * constructor takes, every S0 query among them, and holds it exactly
     * (a lone surrogate, a NUL, a character outside the BMP, 72 KB); it
     * refuses the blank texts the constructor refuses, with its code. */
    @TestFactory
    Stream<DynamicTest> parseTakesWhatThePythonConstructorTakes() throws IOException {
        Map<String, Object> reference = section("reference");
        Map<String, Object> answers = T311Json.object(reference.get("constructor"));
        Map<String, Object> validate = T311Json.object(section("binding").get("validate"));
        Map<String, Object> blanks = T311Json.object(reference.get("texts"));
        List<DynamicTest> tests = new ArrayList<>();
        for (Map.Entry<String, Object> each : answers.entrySet()) {
            String name = each.getKey();
            Map<String, Object> answer = T311Json.object(each.getValue());
            String text = (String) (blanks.containsKey(name)
                    ? blanks.get(name)
                    : T311Json.object(validate.get(name)).get("text"));
            tests.add(DynamicTest.dynamicTest(name, () -> {
                Class<?> query = T311Names.type(T311Names.QUERY);
                if (Boolean.TRUE.equals(answer.get("taken"))) {
                    Object parsed = T311Names.call(query, "parse", text);
                    Object kept = T311Names.read(parsed, "text", "getText");
                    assertArrayEquals(text.toCharArray(), ((String) kept).toCharArray(),
                            name + ": parse did not keep the text");
                } else {
                    try {
                        T311Names.call(query, "parse", text);
                        fail(name + ": parse took a text the reference refuses");
                    } catch (Exception raised) {
                        assertTrue(T311Names.isStructuredDataException(raised),
                                name + ": raised " + raised.getClass().getName());
                        assertEquals(answer.get("code"), T311Names.code(raised), name);
                    }
                }
            }));
        }
        return tests.stream();
    }
}
