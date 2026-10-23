// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Text written as the reference's Python driver path writes it, where the Arrow that reaches the
 * core holds that text: a JSON value Postgres gave, read by Python's {@code json.loads} and
 * written back by {@code json.dumps(value, separators=(',', ':'))} (ASCII only, a float as
 * Python's {@code repr} of it), as the Python provider hands one over. Not part of the API.
 */
public final class PythonText {
  private PythonText() {}

  /**
   * The JSON text {@code text} as Python's {@code json.loads} reads it and {@code json.dumps(...,
   * separators=(',', ':'))} writes it back, as the Python Postgres provider hands a JSON value
   * over.
   */
  public static String json(String text) {
    return dumps(new Reader(text).document(), ",", ":", true);
  }

  /**
   * The JSON {@code bytes} as Python's {@code json.loads} reads them and {@code json.dumps(...,
   * allow_nan=False)} writes them back with its default separators ({@code ", "} and {@code ": "}),
   * as the Python host sends a request's payload.
   *
   * @throws IllegalArgumentException text Python would not read, or a number that is not finite
   */
  public static String dumpsDefault(byte[] bytes) {
    return dumps(new Reader(new String(bytes, StandardCharsets.UTF_8)).document(), ", ", ": ", false);
  }

  /** A number as written, which Python reads as an {@code int} or, with a fraction or exponent, a {@code float}. */
  private record Num(String text) {
    boolean whole() {
      return text.chars().noneMatch(c -> c == '.' || c == 'e' || c == 'E');
    }
  }

  /** Python's {@code json.loads}, its numbers kept as written. */
  private static final class Reader {
    private final String text;
    private int at;

    Reader(String text) {
      this.text = text;
    }

    Object document() {
      Object value = value(0);
      space();
      if (at != text.length()) {
        throw refused();
      }
      return value;
    }

    private IllegalArgumentException refused() {
      return new IllegalArgumentException("JSON Python would not read");
    }

    private void space() {
      while (at < text.length() && " \t\n\r".indexOf(text.charAt(at)) >= 0) {
        at++;
      }
    }

    private Object value(int depth) {
      if (depth > 900) {
        throw refused();
      }
      space();
      if (at >= text.length()) {
        throw refused();
      }
      char c = text.charAt(at);
      if (c == '{') {
        at++;
        Map<String, Object> object = new LinkedHashMap<>();
        space();
        if (peek('}')) {
          return object;
        }
        while (true) {
          space();
          if (at >= text.length() || text.charAt(at) != '"') {
            throw refused();
          }
          String key = string();
          space();
          expect(':');
          object.put(key, value(depth + 1));
          space();
          if (peek('}')) {
            return object;
          }
          expect(',');
        }
      }
      if (c == '[') {
        at++;
        List<Object> list = new ArrayList<>();
        space();
        if (peek(']')) {
          return list;
        }
        while (true) {
          list.add(value(depth + 1));
          space();
          if (peek(']')) {
            return list;
          }
          expect(',');
        }
      }
      if (c == '"') {
        return string();
      }
      for (String word : new String[] {"true", "false", "null", "NaN", "Infinity", "-Infinity"}) {
        if (text.startsWith(word, at)) {
          at += word.length();
          return switch (word) {
            case "true" -> Boolean.TRUE;
            case "false" -> Boolean.FALSE;
            case "null" -> null;
            default -> new Num(word);
          };
        }
      }
      int start = at;
      if (peek('-')) {
        // The sign is part of the number.
      }
      while (at < text.length() && "0123456789+-.eE".indexOf(text.charAt(at)) >= 0) {
        at++;
      }
      if (at == start) {
        throw refused();
      }
      return new Num(text.substring(start, at));
    }

    private boolean peek(char c) {
      if (at < text.length() && text.charAt(at) == c) {
        at++;
        return true;
      }
      return false;
    }

    private void expect(char c) {
      if (!peek(c)) {
        throw refused();
      }
    }

    private String string() {
      expect('"');
      StringBuilder out = new StringBuilder();
      while (true) {
        if (at >= text.length()) {
          throw refused();
        }
        char c = text.charAt(at++);
        if (c == '"') {
          return out.toString();
        }
        if (c != '\\') {
          out.append(c);
          continue;
        }
        if (at >= text.length()) {
          throw refused();
        }
        char e = text.charAt(at++);
        switch (e) {
          case '"', '\\', '/' -> out.append(e);
          case 'b' -> out.append('\b');
          case 'f' -> out.append('\f');
          case 'n' -> out.append('\n');
          case 'r' -> out.append('\r');
          case 't' -> out.append('\t');
          case 'u' -> {
            if (at + 4 > text.length()) {
              throw refused();
            }
            out.append((char) Integer.parseInt(text.substring(at, at + 4), 16));
            at += 4;
          }
          default -> throw refused();
        }
      }
    }
  }

  private static String dumps(Object value, String comma, String colon, boolean nan) {
    StringBuilder out = new StringBuilder();
    // Iterative, so a deep value is written without the stack.
    Deque<Object> work = new ArrayDeque<>();
    work.push(orNull(value));
    while (!work.isEmpty()) {
      Object next = work.pop();
      if (next instanceof Literal literal) {
        out.append(literal.text());
      } else if (next instanceof Map<?, ?> map) {
        out.append('{');
        Deque<Object> members = new ArrayDeque<>();
        Iterator<? extends Map.Entry<?, ?>> entries = map.entrySet().iterator();
        while (entries.hasNext()) {
          Map.Entry<?, ?> entry = entries.next();
          members.add(new Literal(string(String.valueOf(entry.getKey())) + colon));
          members.add(orNull(entry.getValue()));
          if (entries.hasNext()) {
            members.add(new Literal(comma));
          }
        }
        members.add(new Literal("}"));
        pushAll(work, members);
      } else if (next instanceof List<?> list) {
        out.append('[');
        Deque<Object> items = new ArrayDeque<>();
        for (int at = 0; at < list.size(); at++) {
          items.add(orNull(list.get(at)));
          if (at + 1 < list.size()) {
            items.add(new Literal(comma));
          }
        }
        items.add(new Literal("]"));
        pushAll(work, items);
      } else {
        out.append(scalar(next, nan));
      }
    }
    return out.toString();
  }

  private static void pushAll(Deque<Object> work, Deque<Object> items) {
    Iterator<Object> backwards = items.descendingIterator();
    while (backwards.hasNext()) {
      work.push(backwards.next());
    }
  }

  /** Text already written. */
  private record Literal(String text) {}

  /** A value the work deque can hold: null as its text, which no deque holds. */
  private static Object orNull(Object value) {
    return value == null ? new Literal("null") : value;
  }

  private static String scalar(Object value, boolean nan) {
    if (value == null) {
      return "null";
    }
    if (value instanceof Boolean flag) {
      return flag ? "true" : "false";
    }
    if (value instanceof Long || value instanceof BigInteger || value instanceof Integer) {
      return value.toString();
    }
    if (value instanceof Num number) {
      if (number.whole() && !number.text().equals("Infinity") && !number.text().equals("NaN")
          && !number.text().equals("-Infinity")) {
        return new BigInteger(number.text()).toString();
      }
      double real =
          switch (number.text()) {
            case "NaN" -> Double.NaN;
            case "Infinity" -> Double.POSITIVE_INFINITY;
            case "-Infinity" -> Double.NEGATIVE_INFINITY;
            default -> Double.parseDouble(number.text());
          };
      if (!nan && (Double.isNaN(real) || Double.isInfinite(real))) {
        throw new IllegalArgumentException("a number JSON has no form for");
      }
      return floatJson(real);
    }
    if (value instanceof String text) {
      return string(text);
    }
    throw new IllegalArgumentException("not a JSON value");
  }

  /** A float as {@code json.dumps} writes one: its repr, or Python's names of the non-finite. */
  private static String floatJson(double value) {
    if (Double.isNaN(value)) {
      return "NaN";
    }
    if (Double.isInfinite(value)) {
      return value > 0 ? "Infinity" : "-Infinity";
    }
    return floatRepr(value);
  }

  /**
   * Python's {@code repr} of a float: the shortest digits that read back as it, written plainly
   * from {@code 1e-4} to below {@code 1e16} (with a {@code .0} where it is whole) and with an
   * exponent of at least two digits outside ({@code 1e+16}, {@code 1.5e-05}).
   */
  public static String floatRepr(double value) {
    if (value == 0) {
      return (1 / value < 0) ? "-0.0" : "0.0";
    }
    BigDecimal exact = new BigDecimal(value);
    BigDecimal shortest = exact;
    for (int digits = 1; digits <= 17; digits++) {
      BigDecimal rounded = exact.round(new MathContext(digits, RoundingMode.HALF_EVEN));
      if (rounded.doubleValue() == value) {
        shortest = rounded;
        break;
      }
    }
    BigDecimal stripped = shortest.stripTrailingZeros();
    String digits = stripped.unscaledValue().abs().toString();
    int exponent = digits.length() - 1 - stripped.scale();
    String sign = stripped.signum() < 0 ? "-" : "";
    if (exponent >= -4 && exponent < 16) {
      String plain = stripped.abs().toPlainString();
      return sign + (plain.contains(".") ? plain : plain + ".0");
    }
    String mantissa = digits.length() == 1 ? digits : digits.charAt(0) + "." + digits.substring(1);
    String power = String.format("%02d", Math.abs(exponent));
    return sign + mantissa + "e" + (exponent < 0 ? "-" : "+") + power;
  }

  /** Text as {@code json.dumps} writes it with {@code ensure_ascii}: every other character escaped. */
  static String string(String text) {
    StringBuilder out = new StringBuilder("\"");
    for (int at = 0; at < text.length(); at++) {
      char c = text.charAt(at);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        case '\b' -> out.append("\\b");
        case '\f' -> out.append("\\f");
        default -> {
          if (c < 0x20 || c > 0x7e) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    return out.append('"').toString();
  }
}
