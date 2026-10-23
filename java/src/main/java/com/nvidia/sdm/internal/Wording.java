// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * What the core's errors and warnings say. The core picks each one's sentence from a closed list
 * (T3.17), and every host fills that sentence's one shared template, which the native library
 * exports as data ({@link NativeCore#templates}); this class chooses no sentence. It fills a
 * template with the values the error or warning names, written as Java writes them (the one value
 * formatter), and says a sentence it cannot fill as the defect it is, naming the sentence, as the
 * Python host does. Not part of the API.
 *
 * <p>A template is text with placeholders in braces, {@code {{} and {@code }}} standing for a
 * brace ({@code sdm_contract::wording}). This host fills {@code {name}}, {@code {name[0]}},
 * {@code {name:,}} (a whole number, its thousands grouped) and the conversions {@code !r} (the
 * value as a Java literal), {@code !l} and {@code !a} (an and-list of literals, or of the values
 * as they are), {@code !j} and {@code !c} (joined by commas), {@code !R} (by semicolons) and
 * {@code !d} (by dots). The rest of the grammar (times, values the caller gave as JSON text,
 * paired lists, database names, host-held values and advice) arrives with the Java host's first
 * data operations (T3.12); validation (S0) names none of it. A sentence that relays a host
 * library's own message is said in its shared, neutral words: the Java host relays no such
 * library.
 */
public final class Wording {
  /** The words of a sentence the core gave without what its template names. */
  static final String DEFECT =
      "the core said %s without what it names; this is a defect, not a problem with the request";

  private static volatile Map<String, String> templates;

  private Wording() {}

  /**
   * The words of an error or warning the core gave, as its JSON object: its sentence filled with
   * its fields, then, for an error in a part of a question asked in parts, the part's sentence,
   * after the error's, which ends in a full stop.
   */
  public static String said(Map<String, Object> given) {
    Map<String, Object> fields = Messages.details(given);
    String said = say(String.valueOf(given.get("sentence")), fields);
    if (given.get("part") instanceof String part) {
      said = said.stripTrailing();
      if (!said.isEmpty() && ".!?".indexOf(said.charAt(said.length() - 1)) < 0) {
        said += ".";
      }
      said += " " + say(part, fields);
    }
    return said;
  }

  /** What {@code sentence} says, filled with {@code fields}; the defect, named, if it cannot be. */
  static String say(String sentence, Map<String, Object> fields) {
    String template = table().get(sentence);
    try {
      if (template == null) {
        throw new Unfilled();
      }
      return fill(template, fields);
    } catch (Unfilled e) {
      return String.format(DEFECT, '"' + sentence + '"');
    }
  }

  /** The core's templates, by sentence, read once. */
  @SuppressWarnings("unchecked")
  static Map<String, String> table() {
    Map<String, String> known = templates;
    if (known == null) {
      Map<String, String> read = new LinkedHashMap<>();
      try {
        Map<String, Object> table = (Map<String, Object>) Json.read(NativeCore.templates());
        for (Object entry : (List<Object>) table.get("sentences")) {
          Map<String, Object> sentence = (Map<String, Object>) entry;
          read.put((String) sentence.get("name"), (String) sentence.get("template"));
        }
      } catch (Json.JsonException | ClassCastException e) {
        throw Messages.internal("the native library's table of sentences cannot be read");
      }
      known = Map.copyOf(read);
      templates = known;
    }
    return known;
  }

  /** A placeholder this host does not fill, or a field the error does not carry. */
  static final class Unfilled extends Exception {
    private static final long serialVersionUID = 1L;

    Unfilled() {
      super(null, null, false, false);
    }
  }

  /** {@code template} filled with {@code fields}. */
  static String fill(String template, Map<String, Object> fields) throws Unfilled {
    StringBuilder out = new StringBuilder();
    int n = 0;
    while (n < template.length()) {
      char c = template.charAt(n);
      if ((c == '{' || c == '}') && template.startsWith(String.valueOf(c), n + 1)) {
        out.append(c);
        n += 2;
      } else if (c == '{') {
        int end = template.indexOf('}', n);
        if (end < 0) {
          throw new Unfilled();
        }
        out.append(placeholder(template.substring(n + 1, end), fields));
        n = end + 1;
      } else if (c == '}') {
        throw new Unfilled();
      } else {
        out.append(c);
        n++;
      }
    }
    return out.toString();
  }

  /** One placeholder: {@code name[index]!conversion:spec}, each part but the name optional. */
  private static String placeholder(String spec, Map<String, Object> fields) throws Unfilled {
    String format = "";
    int colon = spec.indexOf(':');
    if (colon >= 0) {
      format = spec.substring(colon + 1);
      spec = spec.substring(0, colon);
    }
    char conversion = 0;
    int bang = spec.indexOf('!');
    if (bang >= 0) {
      if (spec.length() != bang + 2) {
        throw new Unfilled();
      }
      conversion = spec.charAt(bang + 1);
      spec = spec.substring(0, bang);
    }
    Integer index = null;
    int bracket = spec.indexOf('[');
    if (bracket >= 0) {
      if (!spec.endsWith("]")) {
        throw new Unfilled();
      }
      try {
        index = Integer.parseInt(spec.substring(bracket + 1, spec.length() - 1));
      } catch (NumberFormatException e) {
        throw new Unfilled();
      }
      spec = spec.substring(0, bracket);
    }
    if (!fields.containsKey(spec)) {
      throw new Unfilled();
    }
    Object value = fields.get(spec);
    if (index != null) {
      if (!(value instanceof List<?> list) || index >= list.size()) {
        throw new Unfilled();
      }
      value = list.get(index);
    }
    return formatted(converted(value, conversion), format);
  }

  private static Object converted(Object value, char conversion) throws Unfilled {
    return switch (conversion) {
      case 0 -> value;
      case 'r' -> literal(value);
      case 'l' -> andList(items(value, true));
      case 'a' -> andList(items(value, false));
      case 'j' -> String.join(", ", items(value, true));
      case 'c' -> String.join(", ", items(value, false));
      case 'R' -> String.join("; ", items(value, false));
      case 'd' -> String.join(".", items(value, false));
      default -> throw new Unfilled();
    };
  }

  private static String formatted(Object value, String format) throws Unfilled {
    if (format.isEmpty()) {
      return plain(value);
    }
    if (",".equals(format) && (value instanceof Long || value instanceof BigInteger)) {
      return String.format(Locale.ROOT, "%,d", value);
    }
    throw new Unfilled();
  }

  private static List<String> items(Object value, boolean literal) throws Unfilled {
    if (!(value instanceof List<?> list)) {
      throw new Unfilled();
    }
    List<String> out = new ArrayList<>();
    for (Object item : list) {
      out.add(literal ? literal(item) : plain(item));
    }
    return out;
  }

  /** {@code 'a'}, {@code 'a' and 'b'}, {@code 'a', 'b' and 'c'}: the words given, as an and-list. */
  private static String andList(List<String> words) {
    if (words.size() < 3) {
      return String.join(" and ", words);
    }
    return String.join(", ", words.subList(0, words.size() - 1))
        + " and "
        + words.get(words.size() - 1);
  }

  /** A value as it is: text as itself, any other value as its literal. */
  private static String plain(Object value) throws Unfilled {
    return value instanceof String text ? text : literal(value);
  }

  /** A value as Java writes it as a literal: text quoted and escaped, a list in brackets. */
  static String literal(Object value) throws Unfilled {
    if (value == null || value instanceof Boolean || value instanceof Long
        || value instanceof BigInteger) {
      return String.valueOf(value);
    }
    if (value instanceof String text) {
      StringBuilder out = new StringBuilder("\"");
      for (int n = 0; n < text.length(); n++) {
        char c = text.charAt(n);
        switch (c) {
          case '"' -> out.append("\\\"");
          case '\\' -> out.append("\\\\");
          case '\n' -> out.append("\\n");
          case '\r' -> out.append("\\r");
          case '\t' -> out.append("\\t");
          default -> {
            if (c < 0x20 || c == 0x7f || Character.isSurrogate(c) && !paired(text, n)) {
              out.append(String.format("\\u%04x", (int) c));
            } else {
              out.append(c);
            }
          }
        }
      }
      return out.append('"').toString();
    }
    if (value instanceof List<?> list) {
      return "[" + String.join(", ", items(list, true)) + "]";
    }
    // A tagged scalar this host does not read plainly (a time, a decimal): T3.12's.
    throw new Unfilled();
  }

  /** Whether the surrogate at {@code n} is half of a pair. */
  private static boolean paired(String text, int n) {
    char c = text.charAt(n);
    return Character.isHighSurrogate(c)
        ? n + 1 < text.length() && Character.isLowSurrogate(text.charAt(n + 1))
        : n > 0 && Character.isHighSurrogate(text.charAt(n - 1));
  }
}
