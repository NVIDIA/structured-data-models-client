// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The names T3.11's tests reach the Java host by, in one place, and the
 * helpers that reach them.
 *
 * <p>The public names are choice 10's (design, "Architecture choices",
 * 10): {@code com.nvidia.sdm.PredictiveQuery.parse(String)}, holding its
 * text as Python's {@code PredictiveQuery.text} does, and one
 * {@code com.nvidia.sdm.StructuredDataException} carrying the contract's
 * error code, as Python's {@code StructuredDataError.code} does, and its
 * details, as {@code StructuredDataError.details} does.
 *
 * <p>ASSUMED until the implementer relays its names (the test author then
 * updates this file, and only this file): the binding's low-level run, the
 * Java twin of {@code sdm_client._native.Run}, is
 * {@code com.nvidia.sdm.internal.Run}, with {@code static Run start(byte[]
 * control, Map<Integer, ?> attachments)} (a bytes attachment a
 * {@code byte[]}), {@code next()} giving the step (its control bytes, or an
 * object whose {@code control()} gives them), and {@code close()}, the
 * explicit release; a core error reaches Java from it as the one
 * {@code StructuredDataException}, whose code is read by {@code code()} or
 * {@code getCode()} and details by {@code details()} or
 * {@code getDetails()}. The names are reached by reflection, so a name
 * that differs fails the tests that need it, naming itself, and never the
 * build.
 */
final class T311Names {
    static final String QUERY = "com.nvidia.sdm.PredictiveQuery";
    static final String EXCEPTION = "com.nvidia.sdm.StructuredDataException";
    /** ASSUMED: the binding's low-level run. */
    static final String RUN = "com.nvidia.sdm.internal.Run";

    /** The host facts every start carries: the recorder's FACTS. */
    static final String FACTS =
            "{\"now\":{\"t\":\"ts\",\"unit\":\"us\",\"zone\":\"UTC\",\"v\":\"1767225600000000\"},"
                    + "\"local_zone\":null,\"zone_roots\":[],"
                    + "\"platform\":{\"os\":\"linux\",\"strftime_pads_year\":false}}";

    /** The two runtime flags of choice 10, which every Java test runs with. */
    static final String[] FLAGS = {
        "--enable-native-access=ALL-UNNAMED", "--add-opens=java.base/java.nio=ALL-UNNAMED",
    };

    private T311Names() {}

    /** The repository's root: the system property {@code sdm.repo}, else the
     * nearest directory above the working one holding {@code contract/}. */
    static Path root() {
        String given = System.getProperty("sdm.repo");
        if (given != null && !given.isEmpty()) {
            return Paths.get(given).toAbsolutePath();
        }
        for (Path at = Paths.get("").toAbsolutePath(); at != null; at = at.getParent()) {
            if (Files.isRegularFile(at.resolve("contract/t311-java-vectors.json"))) {
                return at;
            }
        }
        throw new IllegalStateException(
                "no repository root above " + Paths.get("").toAbsolutePath()
                        + ": set -Dsdm.repo");
    }

    /** contract/t311-java-vectors.json, read. */
    static Map<String, Object> vectors() throws IOException {
        Path file = root().resolve("contract/t311-java-vectors.json");
        return T311Json.object(T311Json.read(Files.readString(file, StandardCharsets.UTF_8)));
    }

    /** The contract version, read from the crate (lesson 11: never
     * hard-coded). */
    static long contractVersion() throws IOException {
        String source = Files.readString(
                root().resolve("rust/crates/sdm-contract/src/lib.rs"), StandardCharsets.UTF_8);
        Matcher found = Pattern.compile("pub const CONTRACT_VERSION: u32 = (\\d+);").matcher(source);
        if (!found.find()) {
            throw new IllegalStateException("sdm-contract names no CONTRACT_VERSION");
        }
        return Long.parseLong(found.group(1));
    }

    /** A start control message for {@code operation}, a JSON object's text. */
    static byte[] start(String operation) throws IOException {
        return ("{\"message\":\"start\",\"contract\":" + contractVersion() + ",\"facts\":" + FACTS
                        + ",\"operation\":" + operation + "}")
                .getBytes(StandardCharsets.UTF_8);
    }

    /** A vector's attachments: id to the bytes recorded (base64). */
    static Map<Integer, Object> attachments(Map<String, Object> recorded) {
        Map<Integer, Object> found = new LinkedHashMap<>();
        for (Map.Entry<String, Object> each : recorded.entrySet()) {
            found.put(Integer.valueOf(each.getKey()), Base64.getDecoder().decode((String) each.getValue()));
        }
        return found;
    }

    static Class<?> type(String name) {
        try {
            return Class.forName(name);
        } catch (ClassNotFoundException missing) {
            throw new AssertionError("T3.11: no class " + name + " (T311Names)", missing);
        }
    }

    /** Calls {@code name} on {@code target} (a class for a static method),
     * the first public method of that name taking {@code arguments}; an
     * exception the method throws is thrown as itself. */
    static Object call(Object target, String name, Object... arguments) throws Exception {
        Class<?> type = target instanceof Class ? (Class<?>) target : target.getClass();
        boolean statics = target instanceof Class;
        for (Method method : type.getMethods()) {
            if (!method.getName().equals(name)
                    || method.getParameterCount() != arguments.length
                    || Modifier.isStatic(method.getModifiers()) != statics) {
                continue;
            }
            try {
                return method.invoke(statics ? null : target, arguments);
            } catch (InvocationTargetException thrown) {
                Throwable cause = thrown.getCause();
                if (cause instanceof Exception) {
                    throw (Exception) cause;
                }
                if (cause instanceof Error) {
                    throw (Error) cause;
                }
                throw thrown;
            } catch (IllegalArgumentException mismatched) {
                continue;
            }
        }
        throw new AssertionError("T3.11: " + type.getName() + " has no "
                + (statics ? "static " : "") + "method " + name + " taking "
                + arguments.length + " argument(s) (T311Names)");
    }

    /** The first of {@code names} that {@code target} answers with no
     * arguments. */
    static Object read(Object target, String... names) throws Exception {
        for (String name : names) {
            Method method;
            try {
                method = target.getClass().getMethod(name);
            } catch (NoSuchMethodException absent) {
                continue;
            }
            return call(target, method.getName());
        }
        throw new AssertionError("T3.11: " + target.getClass().getName() + " answers none of "
                + String.join(", ", names) + " (T311Names)");
    }

    /** The contract's error code {@code error} carries. */
    static String code(Throwable error) throws Exception {
        Object code = read(error, "code", "getCode");
        return code == null ? null : code.toString();
    }

    /** The detail {@code name} of {@code error}, or null. */
    static Object detail(Throwable error, String name) throws Exception {
        Object details = read(error, "details", "getDetails");
        return details instanceof Map ? ((Map<?, ?>) details).get(name) : null;
    }

    /** Starts a run of {@code operation} with {@code attachments}. */
    static Object startRun(String operation, Map<Integer, Object> attachments) throws Exception {
        return call(type(RUN), "start", start(operation), attachments);
    }

    /** The control bytes of the run's next step. */
    static byte[] next(Object run) throws Exception {
        Object step = call(run, "next");
        if (step instanceof byte[]) {
            return (byte[]) step;
        }
        Object control = read(step, "control", "getControl");
        if (!(control instanceof byte[])) {
            throw new AssertionError("T3.11: a step's control is not bytes (T311Names)");
        }
        return (byte[]) control;
    }

    /** The run's next step as the whole message the core wrote, as JSON
     * text: an {@code attached} one, past the control cap, read from the
     * bytes attachment it names (decision H7); and its control bytes. */
    static Object[] nextWhole(Object run) throws Exception {
        Object step = call(run, "next");
        byte[] control = (byte[]) read(step, "control", "getControl");
        String text = new String(control, java.nio.charset.StandardCharsets.UTF_8);
        Map<String, Object> message = T311Json.object(T311Json.read(text));
        if ("attached".equals(message.get("message"))) {
            int id = ((Number) message.get("attachment")).intValue();
            byte[] carried = (byte[]) call(step, "bytes", id);
            text = new String(carried, java.nio.charset.StandardCharsets.UTF_8);
        }
        return new Object[] {control, text};
    }

    /** Releases the run. */
    static void close(Object run) throws Exception {
        call(run, "close");
    }

    static boolean isStructuredDataException(Throwable error) {
        return type(EXCEPTION).isInstance(error);
    }
}
