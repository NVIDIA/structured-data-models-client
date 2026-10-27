// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A JSON reader for T3.11's tests alone, so they name no JSON library the
 * build might not have: objects as {@link LinkedHashMap}, arrays as
 * {@link ArrayList}, strings with every escape read as JSON says (a lone
 * {@code \\uD83D} stays a lone UTF-16 unit, as a Java string holds it),
 * numbers as {@link Long} or {@link Double}, and {@code true},
 * {@code false} and {@code null}.
 */
final class T311Json {
    private final String text;
    private int at;

    private T311Json(String text) {
        this.text = text;
    }

    /** The value {@code text} holds, the whole of it. */
    static Object read(String text) {
        T311Json reader = new T311Json(text);
        Object value = reader.value();
        reader.spaces();
        if (reader.at != text.length()) {
            throw reader.wrong("text after the value");
        }
        return value;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> object(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    static List<Object> array(Object value) {
        return (List<Object>) value;
    }

    private IllegalArgumentException wrong(String what) {
        return new IllegalArgumentException("JSON: " + what + " at " + at);
    }

    private void spaces() {
        while (at < text.length() && " \t\r\n".indexOf(text.charAt(at)) >= 0) {
            at++;
        }
    }

    private Object value() {
        spaces();
        if (at >= text.length()) {
            throw wrong("no value");
        }
        char first = text.charAt(at);
        switch (first) {
            case '{':
                return objectValue();
            case '[':
                return arrayValue();
            case '"':
                return string();
            case 't':
                return word("true", Boolean.TRUE);
            case 'f':
                return word("false", Boolean.FALSE);
            case 'n':
                return word("null", null);
            default:
                return number();
        }
    }

    private Object word(String word, Object value) {
        if (!text.startsWith(word, at)) {
            throw wrong("not " + word);
        }
        at += word.length();
        return value;
    }

    private Map<String, Object> objectValue() {
        Map<String, Object> found = new LinkedHashMap<>();
        at++;
        spaces();
        if (text.charAt(at) == '}') {
            at++;
            return found;
        }
        while (true) {
            spaces();
            String key = string();
            spaces();
            if (text.charAt(at++) != ':') {
                throw wrong("no colon");
            }
            found.put(key, value());
            spaces();
            char next = text.charAt(at++);
            if (next == '}') {
                return found;
            }
            if (next != ',') {
                throw wrong("no comma in an object");
            }
        }
    }

    private List<Object> arrayValue() {
        List<Object> found = new ArrayList<>();
        at++;
        spaces();
        if (text.charAt(at) == ']') {
            at++;
            return found;
        }
        while (true) {
            found.add(value());
            spaces();
            char next = text.charAt(at++);
            if (next == ']') {
                return found;
            }
            if (next != ',') {
                throw wrong("no comma in an array");
            }
        }
    }

    private String string() {
        if (text.charAt(at) != '"') {
            throw wrong("not a string");
        }
        at++;
        StringBuilder found = new StringBuilder();
        while (true) {
            char c = text.charAt(at++);
            if (c == '"') {
                return found.toString();
            }
            if (c != '\\') {
                found.append(c);
                continue;
            }
            char escape = text.charAt(at++);
            switch (escape) {
                case '"':
                case '\\':
                case '/':
                    found.append(escape);
                    break;
                case 'b':
                    found.append('\b');
                    break;
                case 'f':
                    found.append('\f');
                    break;
                case 'n':
                    found.append('\n');
                    break;
                case 'r':
                    found.append('\r');
                    break;
                case 't':
                    found.append('\t');
                    break;
                case 'u':
                    found.append((char) Integer.parseInt(text.substring(at, at + 4), 16));
                    at += 4;
                    break;
                default:
                    throw wrong("an unknown escape");
            }
        }
    }

    private Object number() {
        int start = at;
        while (at < text.length() && "+-0123456789.eE".indexOf(text.charAt(at)) >= 0) {
            at++;
        }
        String digits = text.substring(start, at);
        if (digits.isEmpty()) {
            throw wrong("not a value");
        }
        if (digits.contains(".") || digits.contains("e") || digits.contains("E")) {
            return Double.parseDouble(digits);
        }
        return Long.parseLong(digits);
    }
}
