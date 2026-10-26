// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The JSON the control messages and their input-sized attachments are written in, read and
 * written with no dependency: an object is a {@code Map<String, Object>} in its members' order, an
 * array a {@code List<Object>}, a string a {@code String}, a number a {@code Long}, a {@code
 * BigInteger} past it, or a {@code BigDecimal} when it has a fraction or an exponent, a boolean a
 * {@code Boolean}, and null {@code null}.
 *
 * <p>Writing refuses what JSON cannot hold, as the Python host's encoder does: text with an
 * unpaired surrogate (UTF-8 has no encoding for one), a number that is not finite, a value of
 * another type. Reading refuses anything but one JSON value, and nesting past {@link #MAX_DEPTH},
 * by name and never by a stack overflow. Not part of the API.
 */
public final class Json {
  /** The deepest nesting read or written. */
  public static final int MAX_DEPTH = 512;

  private Json() {}

  /** Text nested past {@link #MAX_DEPTH}, which this reader does not read. */
  public static final class TooDeep extends JsonException {
    private static final long serialVersionUID = 1L;

    TooDeep(String message) {
      super(message);
    }
  }

  /** Why a value could not be written or text could not be read; it quotes none of either. */
  public static class JsonException extends Exception {
    private static final long serialVersionUID = 1L;

    JsonException(String message) {
      super(message);
    }
  }

  /** {@code value} as UTF-8 JSON. */
  public static byte[] write(Object value) throws JsonException {
    StringBuilder out = new StringBuilder();
    write(value, out, 0, false);
    return out.toString().getBytes(StandardCharsets.UTF_8);
  }

  /** Option attachment numbers are admitted by the core, separately from ordinary JSON. */
  public static byte[] writeOptions(Object value) throws JsonException {
    StringBuilder out = new StringBuilder();
    write(value, out, 0, true);
    return out.toString().getBytes(StandardCharsets.UTF_8);
  }

  private static void write(Object value, StringBuilder out, int depth, boolean optionValues)
      throws JsonException {
    if (depth > MAX_DEPTH) {
      throw new JsonException("a value is nested past " + MAX_DEPTH + " levels");
    }
    if (value == null) {
      out.append("null");
    } else if (value instanceof String text) {
      string(text, out);
    } else if (value instanceof Boolean flag) {
      out.append(flag ? "true" : "false");
    } else if (value instanceof Long
        || value instanceof Integer
        || value instanceof Short
        || value instanceof Byte
        || value instanceof BigInteger) {
      out.append(value);
    } else if (value instanceof BigDecimal decimal) {
      out.append(decimal.toString());
    } else if (value instanceof Double || value instanceof Float) {
      double number = ((Number) value).doubleValue();
      if (!optionValues && !Double.isFinite(number)) {
        throw new JsonException("a number is not finite");
      }
      out.append(value);
    } else if (value instanceof Map<?, ?> map) {
      out.append('{');
      boolean first = true;
      for (Map.Entry<?, ?> member : map.entrySet()) {
        if (!(member.getKey() instanceof String key)) {
          throw new JsonException("an object's key is not text");
        }
        if (!first) {
          out.append(',');
        }
        first = false;
        string(key, out);
        out.append(':');
        write(member.getValue(), out, depth + 1, optionValues);
      }
      out.append('}');
    } else if (value instanceof List<?> list) {
      out.append('[');
      for (int i = 0; i < list.size(); i++) {
        if (i > 0) {
          out.append(',');
        }
        write(list.get(i), out, depth + 1, optionValues);
      }
      out.append(']');
    } else {
      throw new JsonException("a value of a type JSON does not hold: " + value.getClass().getName());
    }
  }

  private static void string(String text, StringBuilder out) throws JsonException {
    out.append('"');
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      if (Character.isHighSurrogate(c)) {
        if (i + 1 < text.length() && Character.isLowSurrogate(text.charAt(i + 1))) {
          out.append(c).append(text.charAt(++i));
          continue;
        }
        throw new JsonException("text holds an unpaired surrogate");
      }
      if (Character.isLowSurrogate(c)) {
        throw new JsonException("text holds an unpaired surrogate");
      }
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        case '\b' -> out.append("\\b");
        case '\f' -> out.append("\\f");
        default -> {
          if (c < 0x20) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    out.append('"');
  }

  /** The one JSON value UTF-8 {@code bytes} hold. */
  public static Object read(byte[] bytes) throws JsonException {
    String text;
    try {
      text =
          StandardCharsets.UTF_8
              .newDecoder()
              .decode(java.nio.ByteBuffer.wrap(bytes))
              .toString();
    } catch (java.nio.charset.CharacterCodingException e) {
      throw new JsonException("not UTF-8");
    }
    Reader reader = new Reader(text);
    reader.space();
    Object value = reader.value(0);
    reader.space();
    if (reader.at < text.length()) {
      throw new JsonException("text follows the value at " + reader.at);
    }
    return value;
  }

  private static final class Reader {
    private final String text;
    private int at;

    Reader(String text) {
      this.text = text;
    }

    void space() {
      while (at < text.length()) {
        char c = text.charAt(at);
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
          return;
        }
        at++;
      }
    }

    JsonException failed(String what) {
      return new JsonException(what + " at " + at);
    }

    char peek() throws JsonException {
      if (at >= text.length()) {
        throw failed("the text ends early");
      }
      return text.charAt(at);
    }

    void expect(String word) throws JsonException {
      if (!text.startsWith(word, at)) {
        throw failed("not a JSON value");
      }
      at += word.length();
    }

    Object value(int depth) throws JsonException {
      if (depth > MAX_DEPTH) {
        throw new TooDeep("nesting past " + MAX_DEPTH + " levels at " + at);
      }
      char c = peek();
      switch (c) {
        case '{':
          return object(depth);
        case '[':
          return array(depth);
        case '"':
          return string();
        case 't':
          expect("true");
          return Boolean.TRUE;
        case 'f':
          expect("false");
          return Boolean.FALSE;
        case 'n':
          expect("null");
          return null;
        default:
          if (c == '-' || (c >= '0' && c <= '9')) {
            return number();
          }
          throw failed("not a JSON value");
      }
    }

    Map<String, Object> object(int depth) throws JsonException {
      at++;
      Map<String, Object> members = new LinkedHashMap<>();
      space();
      if (peek() == '}') {
        at++;
        return Collections.unmodifiableMap(members);
      }
      while (true) {
        space();
        if (peek() != '"') {
          throw failed("an object's key is not text");
        }
        String key = string();
        space();
        if (peek() != ':') {
          throw failed("a member has no ':'");
        }
        at++;
        space();
        if (members.containsKey(key)) {
          throw failed("an object repeats a key");
        }
        members.put(key, value(depth + 1));
        space();
        char next = peek();
        at++;
        if (next == '}') {
          return Collections.unmodifiableMap(members);
        }
        if (next != ',') {
          throw failed("an object's members are not separated by ','");
        }
      }
    }

    List<Object> array(int depth) throws JsonException {
      at++;
      List<Object> items = new ArrayList<>();
      space();
      if (peek() == ']') {
        at++;
        return Collections.unmodifiableList(items);
      }
      while (true) {
        space();
        items.add(value(depth + 1));
        space();
        char next = peek();
        at++;
        if (next == ']') {
          return Collections.unmodifiableList(items);
        }
        if (next != ',') {
          throw failed("an array's items are not separated by ','");
        }
      }
    }

    String string() throws JsonException {
      at++;
      StringBuilder out = new StringBuilder();
      while (true) {
        char c = peek();
        at++;
        if (c == '"') {
          return out.toString();
        }
        if (c < 0x20) {
          throw failed("text holds a control character");
        }
        if (c != '\\') {
          out.append(c);
          continue;
        }
        char escape = peek();
        at++;
        switch (escape) {
          case '"' -> out.append('"');
          case '\\' -> out.append('\\');
          case '/' -> out.append('/');
          case 'b' -> out.append('\b');
          case 'f' -> out.append('\f');
          case 'n' -> out.append('\n');
          case 'r' -> out.append('\r');
          case 't' -> out.append('\t');
          case 'u' -> {
            if (at + 4 > text.length()) {
              throw failed("a \\u escape ends early");
            }
            try {
              out.append((char) Integer.parseInt(text.substring(at, at + 4), 16));
            } catch (NumberFormatException e) {
              throw failed("a \\u escape is not hexadecimal");
            }
            at += 4;
          }
          default -> throw failed("an unknown escape");
        }
      }
    }

    Object number() throws JsonException {
      int start = at;
      if (peek() == '-') {
        at++;
      }
      boolean integral = true;
      while (at < text.length()) {
        char c = text.charAt(at);
        if (c >= '0' && c <= '9') {
          at++;
        } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
          integral = false;
          at++;
        } else {
          break;
        }
      }
      String digits = text.substring(start, at);
      try {
        if (integral) {
          BigInteger number = new BigInteger(digits);
          return number.bitLength() < 64 ? (Object) number.longValue() : number;
        }
        return new BigDecimal(digits);
      } catch (NumberFormatException e) {
        throw failed("not a JSON number");
      }
    }
  }
}
