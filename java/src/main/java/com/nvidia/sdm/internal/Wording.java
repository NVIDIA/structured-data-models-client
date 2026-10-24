// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * What the core's errors and warnings say. The core picks each one's sentence from a closed list
 * (T3.17), and every host fills that sentence's one shared template, which the native library
 * exports as data ({@link NativeCore#templates}); this class chooses no sentence. It fills a
 * template with the values the error or warning names, written as Java writes them (the one
 * value formatter, {@link #literal}), the values only this host holds ({@code {host.<name>}}),
 * and the advice a template names, spelled in Java ({@link #ADVICE}); it says warnings of one
 * group given one after another as one, as the table's groups say; and it says a sentence it
 * cannot fill as the defect it is, naming the sentence, as the Python host does. Not part of the
 * API.
 *
 * <p>The grammar is {@code sdm_contract::wording}'s: {@code {name}}, {@code {name[0]}}, {@code
 * {name:,}}, the conversions {@code !r} (a literal), {@code !l} and {@code !a} (an and-list of
 * literals, or of the values as they are), {@code !j} and {@code !c} (joined by commas), {@code
 * !R} (by semicolons), {@code !d} (by dots), {@code !p} (a row of values), {@code !t} (a date or
 * time as {@code java.time} writes it), {@code !T} (a list of times of day or text), and for a
 * value the caller gave, held as the JSON text this host wrote it in, {@code !G} (as a literal),
 * {@code !S} (as text), {@code !N} (the name of its Java type) and {@code !I} (the whole number
 * it is); {@code :to(other)}, {@code :place} and {@code :backend}; {@code {host.<name>}} and
 * {@code {advice:<id>(<field>, ...)}}. A sentence that relays a host library's own message (the
 * Python host's pandas or NumPy) is said in its shared, neutral words: the Java host relays no
 * such library.
 */
public final class Wording {
  /** The words of a sentence the core gave without what its template names. */
  static final String DEFECT =
      "the core said %s without what it names; this is a defect, not a problem with the request";

  /**
   * Each advice a template names, spelled as Java is written, from the fields the template gives
   * it, in order (the core's {@code wording::Advice}); a test holds it to the core's list.
   */
  static final Map<String, Function<List<Object>, String>> ADVICE = advice();

  private static volatile Table table;

  private Wording() {}

  private static Map<String, Function<List<Object>, String>> advice() {
    Map<String, Function<List<Object>, String>> spelled = new LinkedHashMap<>();
    spelled.put(
        "append_relationship",
        given ->
            "schema.withRelationship(new Relationship("
                + quoted(first(given.get(0)))
                + ", <foreign key>, <parent table>, <its primary key>))");
    spelled.put("booleans", given -> "true and false");
    spelled.put(
        "build_relationship",
        given -> "new Relationship(sourceTable, sourceColumn, targetTable, targetColumn)");
    spelled.put(
        "choose_statistic",
        given -> "MaterializeOptions.builder().option(\"prediction_statistic\", ...)");
    spelled.put(
        "clear_time_column",
        given -> "FromSourceOptions.builder().noTimeColumn(" + quoted(given.get(0)) + ")");
    spelled.put(
        "context_samples",
        given -> "MaterializeOptions.builder().numContextSamples(" + plainText(given.get(0)) + ")");
    spelled.put("empty_request", given -> "MaterializeOptions.builder().indices(List.of())");
    spelled.put(
        "give_a_hop",
        given -> "MaterializeOptions.builder().option(\"num_neighbors\", List.of(8, 8))");
    spelled.put(
        "keep_one_link",
        given ->
            "schema.withRelationships(schema.relationships().stream().filter(link -> !("
                + "link.sourceTable().equals("
                + quoted(given.get(0))
                + ") && link.targetTable().equals("
                + quoted(given.get(1))
                + ") && List.of("
                + String.join(", ", quotedAll(given.get(2)))
                + ").contains(link.sourceColumn()))).toList())");
    spelled.put("materialize_call", given -> "materialize()");
    spelled.put("missing_time", given -> "null");
    spelled.put(
        "name_candidates",
        given -> "the values of column " + quoted(given.get(0)) + " as the candidates");
    spelled.put(
        "name_primary_key",
        given -> "FromSourceOptions.builder().primaryKey(" + quoted(given.get(0)) + ", ...)");
    spelled.put(
        "name_time_column",
        given ->
            "FromSourceOptions.builder().timeColumn("
                + quoted(given.get(0))
                + ", "
                + quoted(given.get(1))
                + ")");
    spelled.put("nothing", given -> "null");
    spelled.put("optimize_indices", given -> "an index on the key column");
    spelled.put("predict_call", given -> "predict()");
    spelled.put("raise_most_parts", given -> "a higher limit on the parts one question is asked as");
    spelled.put("result_frame", given -> "PredictionResult.read()");
    spelled.put(
        "semantic_model", given -> "source.semanticModel(" + quoted(given.get(0)) + ", false)");
    spelled.put(
        "set_primary_key",
        given ->
            "FromSourceOptions.builder().primaryKey("
                + quoted(given.get(0))
                + ", "
                + quoted(given.get(1))
                + ")");
    spelled.put(
        "skip_key_check",
        given ->
            "Name another key with FromSourceOptions.builder().primaryKey(...), or pass"
                + " checkKeys(false) to skip (one full scan per table).");
    spelled.put(
        "smaller_calls",
        given -> "MaterializeOptions.builder().option(\"rows_per_call\", N)");
    spelled.put("trust_expressions", given -> "semanticModel(name, true)");
    return Map.copyOf(spelled);
  }

  /** The core's table of sentences, groups, advice and databases, read once. */
  record Table(
      Map<String, Map<String, Object>> sentences,
      Map<String, Map<String, Object>> groups,
      Map<String, List<String>> advice,
      Map<String, Map<String, Object>> databases) {}

  @SuppressWarnings("unchecked")
  static Table table() {
    Table known = table;
    if (known == null) {
      try {
        Map<String, Object> read = (Map<String, Object>) Json.read(NativeCore.templates());
        known =
            new Table(
                byName((List<Object>) read.get("sentences")),
                byName((List<Object>) read.get("groups")),
                arguments((List<Object>) read.get("advice")),
                byName((List<Object>) read.get("databases")));
      } catch (Json.JsonException | ClassCastException | NullPointerException e) {
        throw Messages.internal("the native library's table of sentences cannot be read");
      }
      table = known;
    }
    return known;
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Map<String, Object>> byName(List<Object> entries) {
    Map<String, Map<String, Object>> read = new LinkedHashMap<>();
    for (Object entry : entries == null ? List.of() : entries) {
      Map<String, Object> each = (Map<String, Object>) entry;
      read.put(String.valueOf(each.get("name")), each);
    }
    return Map.copyOf(read);
  }

  @SuppressWarnings("unchecked")
  private static Map<String, List<String>> arguments(List<Object> entries) {
    Map<String, List<String>> read = new LinkedHashMap<>();
    for (Object entry : entries) {
      Map<String, Object> each = (Map<String, Object>) entry;
      read.put(String.valueOf(each.get("name")), List.copyOf((List<String>) each.get("arguments")));
    }
    return Map.copyOf(read);
  }

  /** The advice the core's table names: every one must have a Java spelling here. */
  public static List<String> adviceNamed() {
    return List.copyOf(table().advice().keySet());
  }

  /** The advice this host spells. */
  public static List<String> adviceSpelled() {
    return List.copyOf(ADVICE.keySet());
  }

  // --- saying an error or a warning -----------------------------------------------------

  /** The words of an error or warning the core gave, with no host-held value. */
  public static String said(Map<String, Object> given) {
    return said(given, Map.of());
  }

  /**
   * The words of an error or warning the core gave, as its JSON object: its sentence filled with
   * its fields and the values {@code held}, then, for an error in a part of a question asked in
   * parts, the part's sentence, after the error's, which ends in a full stop.
   */
  public static String said(
      Map<String, Object> given, Map<String, Function<Map<String, Object>, Object>> held) {
    Map<String, Object> fields = fields(given);
    String said = say(String.valueOf(given.get("sentence")), fields, held);
    if (given.get("part") instanceof String part) {
      said = said.stripTrailing();
      if (!said.isEmpty() && ".!?".indexOf(said.charAt(said.length() - 1)) < 0) {
        said += ".";
      }
      said += " " + say(part, fields, held);
    }
    return said;
  }

  /** A run of warnings said as one: their code, words, and the warnings it says. */
  public record Said(String code, String text, List<Map<String, Object>> warnings) {}

  /**
   * {@code warnings}, in order, each said, warnings of one group given one after another with
   * the same values of the fields the group names said together as one (the table's groups).
   */
  @SuppressWarnings("unchecked")
  public static List<Said> warnings(
      List<Map<String, Object>> warnings, Map<String, Function<Map<String, Object>, Object>> held) {
    Table known = table();
    List<Object> keys = new ArrayList<>();
    List<List<Map<String, Object>>> runs = new ArrayList<>();
    for (Map<String, Object> warning : warnings) {
      Map<String, Object> sentence = known.sentences().get(String.valueOf(warning.get("sentence")));
      Object group = sentence == null ? null : sentence.get("group");
      Object key = null;
      if (group instanceof String name && known.groups().containsKey(name)) {
        List<Object> agreed = new ArrayList<>();
        agreed.add(name);
        Map<String, Object> fields =
            warning.get("fields") instanceof Map<?, ?> map ? (Map<String, Object>) map : Map.of();
        for (Object field : (List<Object>) known.groups().get(name).get("agree")) {
          agreed.add(fields.get(String.valueOf(field)));
        }
        key = agreed;
      }
      if (key != null && !keys.isEmpty() && key.equals(keys.get(keys.size() - 1))) {
        runs.get(runs.size() - 1).add(warning);
      } else {
        keys.add(key);
        List<Map<String, Object>> run = new ArrayList<>();
        run.add(warning);
        runs.add(run);
      }
    }
    List<Said> said = new ArrayList<>();
    for (int at = 0; at < runs.size(); at++) {
      List<Map<String, Object>> run = runs.get(at);
      Map<String, Object> first = run.get(0);
      String code = String.valueOf(first.get("code"));
      if (keys.get(at) == null) {
        said.add(new Said(code, said(first, held), List.copyOf(run)));
        continue;
      }
      Map<String, Object> words =
          known.groups().get(String.valueOf(((List<Object>) keys.get(at)).get(0)));
      List<String> each = new ArrayList<>();
      for (Map<String, Object> warning : run) {
        each.add(said(warning, held));
      }
      String opening;
      try {
        opening = fill(String.valueOf(words.get("opening")), fields(first), held);
      } catch (Unfilled e) {
        opening = "";
      }
      said.add(
          new Said(
              code,
              opening
                  + String.join(String.valueOf(words.get("between")), each)
                  + words.get("closing"),
              List.copyOf(run)));
    }
    return said;
  }

  /** What {@code sentence} says, filled with {@code fields}; the defect, named, if it cannot be. */
  static String say(String sentence, Map<String, Object> fields) {
    return say(sentence, fields, Map.of());
  }

  /** Each sentence's template, by name. */
  static Map<String, String> templates() {
    Map<String, String> read = new LinkedHashMap<>();
    table().sentences().forEach((name, entry) -> read.put(name, (String) entry.get("template")));
    return read;
  }

  /** What {@code sentence} says, filled; the defect, named, if it cannot be. */
  static String say(
      String sentence,
      Map<String, Object> fields,
      Map<String, Function<Map<String, Object>, Object>> held) {
    Map<String, Object> entry = table().sentences().get(sentence);
    try {
      if (entry == null || !(entry.get("template") instanceof String template)) {
        throw new Unfilled();
      }
      return fill(template, fields, held);
    } catch (Unfilled | RuntimeException e) {
      return String.format(DEFECT, '"' + sentence + '"');
    }
  }

  /** An error's or warning's fields as Java values: every scalar decoded. */
  static Map<String, Object> fields(Map<String, Object> given) {
    Map<String, Object> fields = new LinkedHashMap<>();
    if (given.get("fields") instanceof Map<?, ?> map) {
      for (Map.Entry<?, ?> field : map.entrySet()) {
        fields.put(String.valueOf(field.getKey()), value(field.getValue()));
      }
    }
    return fields;
  }

  /** A field's value as Java: a scalar's value, or a list of them. */
  static Object value(Object field) {
    if (field instanceof List<?> list) {
      List<Object> out = new ArrayList<>();
      for (Object item : list) {
        out.add(scalar(item));
      }
      return out;
    }
    return scalar(field);
  }

  /**
   * A contract scalar as Java holds it: a whole number a {@code Long} or {@code BigInteger}, a
   * float a {@code Double}, a decimal a {@code BigDecimal}, text a {@code String}, binary a {@code
   * byte[]}, a date a {@link LocalDate}, a timestamp a {@link LocalDateTime} (or, with a zone, a
   * {@link java.time.ZonedDateTime}), a time of day a {@link LocalTime}, a duration a {@link
   * Duration}.
   */
  static Object scalar(Object value) {
    if (!(value instanceof Map<?, ?> tagged) || !(tagged.get("t") instanceof String tag)) {
      return value;
    }
    Object v = tagged.get("v");
    return switch (tag) {
      case "null" -> null;
      case "bool", "utf8" -> v;
      case "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64" -> {
        BigInteger number = new BigInteger((String) v);
        yield number.bitLength() < 64 ? (Object) number.longValue() : number;
      }
      case "f32", "f64" -> floating((String) v);
      case "d128", "d256" ->
          new BigDecimal(new BigInteger((String) v), ((Number) tagged.get("s")).intValue());
      case "binary" -> Base64.getDecoder().decode((String) v);
      case "date32" -> LocalDate.ofEpochDay(Long.parseLong((String) v));
      case "ts" -> timestamp(Long.parseLong((String) v), (String) tagged.get("unit"),
          (String) tagged.get("zone"));
      case "time64" -> LocalTime.ofNanoOfDay(nanos(Long.parseLong((String) v),
          (String) tagged.get("unit")));
      case "duration" -> Duration.ofNanos(nanos(Long.parseLong((String) v),
          (String) tagged.get("unit")));
      default -> value;
    };
  }

  private static Double floating(String text) {
    return switch (text) {
      case "NaN" -> Double.NaN;
      case "inf" -> Double.POSITIVE_INFINITY;
      case "-inf" -> Double.NEGATIVE_INFINITY;
      default -> Double.valueOf(text);
    };
  }

  private static long nanos(long count, String unit) {
    return "us".equals(unit) ? Math.multiplyExact(count, 1_000L) : count;
  }

  private static Object timestamp(long count, String unit, String zone) {
    long seconds = Math.floorDiv(count, "us".equals(unit) ? 1_000_000L : 1_000_000_000L);
    long fraction = Math.floorMod(count, "us".equals(unit) ? 1_000_000L : 1_000_000_000L);
    Instant instant =
        Instant.ofEpochSecond(seconds, "us".equals(unit) ? fraction * 1_000L : fraction);
    if (zone == null) {
      return LocalDateTime.ofInstant(instant, ZoneOffset.UTC);
    }
    try {
      return instant.atZone(ZoneId.of(zone));
    } catch (RuntimeException e) {
      return instant.atZone(ZoneOffset.UTC);
    }
  }

  // --- filling a template -------------------------------------------------------------

  /** A placeholder this host does not fill, or a field the error does not carry. */
  static final class Unfilled extends Exception {
    private static final long serialVersionUID = 1L;

    Unfilled() {
      super(null, null, false, false);
    }
  }

  /** {@code template} filled with {@code fields}, with no host-held value. */
  static String fill(String template, Map<String, Object> fields) throws Unfilled {
    return fill(template, fields, Map.of());
  }

  /** {@code template} filled with {@code fields} and the values {@code held}. */
  static String fill(
      String template,
      Map<String, Object> fields,
      Map<String, Function<Map<String, Object>, Object>> held)
      throws Unfilled {
    StringBuilder out = new StringBuilder();
    int n = 0;
    while (n < template.length()) {
      char c = template.charAt(n);
      if ((c == '{' || c == '}') && template.startsWith(String.valueOf(c), n + 1)) {
        out.append(c);
        n += 2;
      } else if (c == '{') {
        int end = closing(template, n);
        out.append(placeholder(template.substring(n + 1, end), fields, held));
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

  /** Where the placeholder opened at {@code start} closes (an advice's arguments hold none). */
  private static int closing(String template, int start) throws Unfilled {
    int end = template.indexOf('}', start);
    if (end < 0) {
      throw new Unfilled();
    }
    return end;
  }

  /** One placeholder: {@code name[index]!conversion:spec}, each part but the name optional. */
  private static String placeholder(
      String spec,
      Map<String, Object> fields,
      Map<String, Function<Map<String, Object>, Object>> held)
      throws Unfilled {
    if (spec.startsWith("advice:")) {
      return advise(spec.substring("advice:".length()), fields);
    }
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
    Object value;
    if (spec.startsWith("host.")) {
      Function<Map<String, Object>, Object> read = held.get(spec.substring("host.".length()));
      if (read == null) {
        throw new Unfilled();
      }
      try {
        value = read.apply(fields);
      } catch (RuntimeException e) {
        throw new Unfilled();
      }
    } else {
      if (!fields.containsKey(spec)) {
        throw new Unfilled();
      }
      value = fields.get(spec);
    }
    if (index != null) {
      if (!(value instanceof List<?> list) || index < 0 || index >= list.size()) {
        throw new Unfilled();
      }
      value = list.get(index);
    }
    return formatted(converted(value, conversion), format, fields);
  }

  /** {@code {advice:name(field, ...)}}: the advice spelled in Java, given those fields. */
  private static String advise(String spec, Map<String, Object> fields) throws Unfilled {
    int open = spec.indexOf('(');
    if (open < 0 || !spec.endsWith(")")) {
      throw new Unfilled();
    }
    Function<List<Object>, String> spelled = ADVICE.get(spec.substring(0, open));
    if (spelled == null) {
      throw new Unfilled();
    }
    List<Object> given = new ArrayList<>();
    for (String name : spec.substring(open + 1, spec.length() - 1).split(",")) {
      String field = name.strip();
      if (field.isEmpty()) {
        continue;
      }
      if (!fields.containsKey(field)) {
        throw new Unfilled();
      }
      given.add(fields.get(field));
    }
    try {
      return spelled.apply(given);
    } catch (RuntimeException e) {
      throw new Unfilled();
    }
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
      case 'p' -> literal(value);
      case 't' -> time(value);
      case 'T' -> "[" + String.join(", ", timesOfDay(value)) + "]";
      case 'G' -> literal(given(value));
      case 'S' -> plain(given(value));
      case 'N' -> typeName(given(value));
      case 'I' -> whole(given(value));
      default -> throw new Unfilled();
    };
  }

  private static String formatted(Object value, String format, Map<String, Object> fields)
      throws Unfilled {
    if (format.isEmpty()) {
      return plain(value);
    }
    if (",".equals(format)
        && (value instanceof Long || value instanceof BigInteger || value instanceof Integer)) {
      return String.format(Locale.ROOT, "%,d", value);
    }
    if (format.startsWith("to(") && format.endsWith(")")) {
      Object others = fields.get(format.substring(3, format.length() - 1));
      if (!(value instanceof List<?> ones) || !(others instanceof List<?> twos)
          || ones.size() != twos.size()) {
        throw new Unfilled();
      }
      List<String> pairs = new ArrayList<>();
      for (int at = 0; at < ones.size(); at++) {
        pairs.add(literal(ones.get(at)) + " to " + literal(twos.get(at)));
      }
      return String.join(", ", pairs);
    }
    if ("place".equals(format) || "backend".equals(format)) {
      Map<String, Object> names = table().databases().get(String.valueOf(value));
      if ("backend".equals(format)) {
        return names == null ? String.valueOf(value) : String.valueOf(names.get("backend"));
      }
      return names == null ? value + " database" : String.valueOf(names.get("place"));
    }
    throw new Unfilled();
  }

  /** A value the caller gave, held as the JSON text this host wrote it in, read back. */
  private static Object given(Object value) throws Unfilled {
    if (!(value instanceof String text)) {
      throw new Unfilled();
    }
    try {
      return Json.read(text.getBytes(StandardCharsets.UTF_8));
    } catch (Json.JsonException e) {
      throw new Unfilled();
    }
  }

  /** The Java type a value the caller gave was, as JSON read it back. */
  private static String typeName(Object value) {
    if (value == null) {
      return "null";
    }
    if (value instanceof BigDecimal) {
      return "Double";
    }
    if (value instanceof Map<?, ?>) {
      return "Map";
    }
    if (value instanceof List<?>) {
      return "List";
    }
    return value.getClass().getSimpleName();
  }

  private static Object whole(Object value) throws Unfilled {
    if (value instanceof Long || value instanceof BigInteger) {
      return value;
    }
    if (value instanceof BigDecimal decimal) {
      try {
        return decimal.toBigIntegerExact();
      } catch (ArithmeticException e) {
        throw new Unfilled();
      }
    }
    throw new Unfilled();
  }

  /** A date or time as {@code java.time} writes it; text a time column holds, read as one. */
  private static String time(Object value) throws Unfilled {
    if (value instanceof String text) {
      for (String candidate : new String[] {text, text.replace(' ', 'T')}) {
        try {
          return LocalDateTime.parse(candidate).format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        } catch (DateTimeParseException e) {
          // Try the next reading.
        }
      }
      try {
        return LocalDate.parse(text).atStartOfDay().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
      } catch (DateTimeParseException e) {
        return text;
      }
    }
    if (value instanceof LocalDate date) {
      return date.atStartOfDay().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
    }
    if (value instanceof LocalDateTime local) {
      return local.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
    }
    if (value instanceof java.time.ZonedDateTime zoned) {
      return zoned.format(DateTimeFormatter.ISO_ZONED_DATE_TIME);
    }
    if (value instanceof LocalTime clock) {
      return clock.format(DateTimeFormatter.ISO_LOCAL_TIME);
    }
    throw new Unfilled();
  }

  private static List<String> timesOfDay(Object value) throws Unfilled {
    if (!(value instanceof List<?> list)) {
      throw new Unfilled();
    }
    List<String> out = new ArrayList<>();
    for (Object item : list) {
      out.add(item instanceof String text ? literal(text) : time(item));
    }
    return out;
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

  /** {@code a}, {@code a and b}, {@code a, b and c}: the words given, as an and-list. */
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

  private static String plainText(Object value) {
    try {
      return plain(value);
    } catch (Unfilled e) {
      return String.valueOf(value);
    }
  }

  private static Object first(Object value) {
    return value instanceof List<?> list && !list.isEmpty() ? list.get(0) : value;
  }

  private static String quoted(Object value) {
    try {
      return literal(value);
    } catch (Unfilled e) {
      return String.valueOf(value);
    }
  }

  private static List<String> quotedAll(Object value) {
    List<String> out = new ArrayList<>();
    if (value instanceof List<?> list) {
      for (Object item : list) {
        out.add(quoted(item));
      }
    } else {
      out.add(quoted(value));
    }
    return out;
  }

  /**
   * A value as Java writes it as a literal: text quoted and escaped, a number, a boolean and null
   * as Java prints them, a list in brackets, a date or time as {@code java.time} writes it, and
   * any other object as its {@code toString()} (a source, a value only the host holds).
   */
  static String literal(Object value) throws Unfilled {
    if (value == null || value instanceof Boolean || value instanceof Long
        || value instanceof Integer || value instanceof BigInteger) {
      return String.valueOf(value);
    }
    if (value instanceof Double number) {
      return Double.toString(number);
    }
    if (value instanceof BigDecimal decimal) {
      return decimal.toString();
    }
    if (value instanceof String text) {
      return stringLiteral(text);
    }
    if (value instanceof List<?> list) {
      return "[" + String.join(", ", items(list, true)) + "]";
    }
    if (value instanceof Map<?, ?> map) {
      List<String> members = new ArrayList<>();
      for (Map.Entry<?, ?> entry : map.entrySet()) {
        members.add(literal(entry.getKey()) + "=" + literal(entry.getValue()));
      }
      return "{" + String.join(", ", members) + "}";
    }
    if (value instanceof byte[] bytes) {
      return "0x" + java.util.HexFormat.of().formatHex(bytes);
    }
    if (value instanceof LocalDate || value instanceof LocalDateTime
        || value instanceof java.time.ZonedDateTime || value instanceof LocalTime) {
      return time(value);
    }
    if (value instanceof Duration duration) {
      return duration.toString();
    }
    return Objects.toString(value);
  }

  private static String stringLiteral(String text) {
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

  /** Whether the surrogate at {@code n} is half of a pair. */
  private static boolean paired(String text, int n) {
    char c = text.charAt(n);
    return Character.isHighSurrogate(c)
        ? n + 1 < text.length() && Character.isLowSurrogate(text.charAt(n + 1))
        : n > 0 && Character.isHighSurrogate(text.charAt(n - 1));
  }
}
