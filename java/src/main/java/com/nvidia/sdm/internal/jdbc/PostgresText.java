// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.sql.SQLDataException;
import java.time.DateTimeException;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Values Postgres writes as text, read as the reference's driver (psycopg 3, its text loaders)
 * reads them, and written as the reference's provider writes a value it hands over as text
 * (Python's {@code str} of what psycopg read). A value psycopg refuses is refused here, by what it
 * is ({@link SQLDataException}), never read as another value. Not part of the API.
 */
final class PostgresText {
  private PostgresText() {}

  /** A value psycopg refuses; its message names what it is, never the value. */
  static SQLDataException refused(String what) {
    return new SQLDataException("the reference's driver cannot read this value: " + what);
  }

  // --- dates and times -----------------------------------------------------------------

  private static final Pattern DATE = Pattern.compile("(\\d{4})-(\\d{2})-(\\d{2})");

  /** A {@code date} (ISO), as psycopg reads one: years 1 to 9999, else refused. */
  static LocalDate date(String text) throws SQLDataException {
    Matcher found = DATE.matcher(text);
    if (!found.matches()) {
      throw refused("a date past year 9999, before year 1, or infinite");
    }
    try {
      return LocalDate.of(
          Integer.parseInt(found.group(1)),
          Integer.parseInt(found.group(2)),
          Integer.parseInt(found.group(3)));
    } catch (DateTimeException e) {
      throw refused("a date before year 1");
    }
  }

  private static final Pattern TIMESTAMP =
      Pattern.compile(
          "(\\d+)[^a-zA-Z0-9](\\d+)[^a-zA-Z0-9](\\d+)(?:T|[^a-zA-Z0-9])"
              + "(\\d+)[^a-zA-Z0-9](\\d+)[^a-zA-Z0-9](\\d+)(?:\\.(\\d+))?");

  private static final Pattern OFFSET = Pattern.compile("([-+])(\\d+)(?::(\\d+))?(?::(\\d+))?");

  /** A {@code timestamp} (ISO), as psycopg reads one: years 1 to 9999, else refused. */
  static LocalDateTime timestamp(String text) throws SQLDataException {
    Matcher found = TIMESTAMP.matcher(text);
    if (!found.matches()) {
      throw refused("a timestamp past year 9999, before year 1, or infinite");
    }
    return wall(found);
  }

  /** A {@code timestamptz}, as the server wrote it in the session's zone: its wall and offset. */
  record Zoned(LocalDateTime wall, int offsetSeconds) {
    long micros() {
      return PostgresText.micros(wall.toInstant(ZoneOffset.ofTotalSeconds(offsetSeconds)));
    }
  }

  /** A {@code timestamptz} (ISO), as psycopg reads one: years 1 to 9999, else refused. */
  static Zoned zoned(String text) throws SQLDataException {
    Matcher found =
        Pattern.compile(TIMESTAMP.pattern() + OFFSET.pattern()).matcher(text);
    if (!found.matches()) {
      throw refused("a timestamp past year 9999, before year 1, or infinite");
    }
    return new Zoned(wall(found), offset(found, 8));
  }

  private static LocalDateTime wall(Matcher found) throws SQLDataException {
    try {
      int year = Integer.parseInt(found.group(1));
      if (year < 1 || year > 9999) {
        throw refused("a timestamp past year 9999 or before year 1");
      }
      return LocalDateTime.of(
          year,
          Integer.parseInt(found.group(2)),
          Integer.parseInt(found.group(3)),
          Integer.parseInt(found.group(4)),
          Integer.parseInt(found.group(5)),
          Integer.parseInt(found.group(6)),
          fraction(found.group(7)) * 1_000);
    } catch (DateTimeException | NumberFormatException e) {
      throw refused("a timestamp outside a datetime's range");
    }
  }

  /** An offset's seconds, from the groups at {@code at} (sign, hours, minutes, seconds). */
  private static int offset(Matcher found, int at) {
    int seconds = 3600 * Integer.parseInt(found.group(at + 1));
    if (found.group(at + 2) != null) {
      seconds += 60 * Integer.parseInt(found.group(at + 2));
    }
    if (found.group(at + 3) != null) {
      seconds += Integer.parseInt(found.group(at + 3));
    }
    return "-".equals(found.group(at)) ? -seconds : seconds;
  }

  /** A fraction's digits as microseconds, as psycopg pads them. */
  private static int fraction(String digits) {
    if (digits == null) {
      return 0;
    }
    String six = digits.length() >= 6 ? digits.substring(0, 6) : (digits + "000000").substring(0, 6);
    return Integer.parseInt(six);
  }

  private static final Pattern TIME = Pattern.compile("(\\d+):(\\d+):(\\d+)(?:\\.(\\d+))?");

  /** A {@code time}, as psycopg reads one: 24:00:00 is no time of day, and refused. */
  static LocalTime time(String text) throws SQLDataException {
    Matcher found = TIME.matcher(text);
    if (!found.lookingAt()) {
      throw refused("a time of day it cannot parse");
    }
    return timeOf(found);
  }

  private static LocalTime timeOf(Matcher found) throws SQLDataException {
    try {
      return LocalTime.of(
          Integer.parseInt(found.group(1)),
          Integer.parseInt(found.group(2)),
          Integer.parseInt(found.group(3)),
          fraction(found.group(4)) * 1_000);
    } catch (DateTimeException | NumberFormatException e) {
      throw refused("a time of 24:00:00 or past it");
    }
  }

  /** A {@code timetz}: its time of day and offset. */
  record ZonedTime(LocalTime time, int offsetSeconds) {}

  /** A {@code timetz}, as psycopg reads one: 24:00:00 is no time of day, and refused. */
  static ZonedTime zonedTime(String text) throws SQLDataException {
    Matcher found = Pattern.compile(TIME.pattern() + OFFSET.pattern()).matcher(text);
    if (!found.matches()) {
      throw refused("a time with a zone it cannot parse");
    }
    return new ZonedTime(timeOf(found), offset(found, 5));
  }

  static long micros(java.time.Instant instant) {
    return Math.addExact(
        Math.multiplyExact(instant.getEpochSecond(), 1_000_000L), instant.getNano() / 1_000L);
  }

  // --- intervals -----------------------------------------------------------------------

  /** psycopg's reading of an interval in the {@code postgres} style. */
  private static final Pattern INTERVAL =
      Pattern.compile(
          "(?:([-+]?\\d+)\\s+years?\\s*)?"
              + "(?:([-+]?\\d+)\\s+mons?\\s*)?"
              + "(?:([-+]?\\d+)\\s+days?\\s*)?"
              + "(?:([-+])?(\\d+):(\\d+):(\\d+(?:\\.\\d+)?))?");

  /** The most days a {@code timedelta} holds, either way. */
  private static final int MOST_DAYS = 999_999_999;

  /** An interval as psycopg's {@code timedelta} of it: its days, and the microseconds after. */
  record Delta(int days, long micros) {}

  /**
   * An interval as psycopg's {@code timedelta} of it: a year as 365 days and a month as 30, summed
   * as psycopg's C loader sums them (32-bit, so a sum past it wraps), the time of day as seconds,
   * rounded to the microsecond as a timedelta rounds them, then normalized as a timedelta is; one
   * past a timedelta's days refused, as psycopg refuses it. Only the {@code postgres} style is
   * read, as psycopg reads only it.
   */
  static Delta interval(String text, String style) throws SQLDataException {
    if (!"postgres".equals(style)) {
      throw refused("an interval written in a style other than postgres");
    }
    Matcher found = INTERVAL.matcher(text);
    if (!found.matches()) {
      throw refused("an interval it cannot parse");
    }
    int days = 0;
    try {
      if (found.group(1) != null) {
        days += 365 * Integer.parseInt(found.group(1));
      }
      if (found.group(2) != null) {
        days += 30 * Integer.parseInt(found.group(2));
      }
      if (found.group(3) != null) {
        days += Integer.parseInt(found.group(3));
      }
    } catch (NumberFormatException e) {
      throw refused("an interval past a timedelta");
    }
    long micros = 0;
    if (found.group(5) != null) {
      double seconds =
          3600.0 * Long.parseLong(found.group(5))
              + 60.0 * Long.parseLong(found.group(6))
              + Double.parseDouble(found.group(7));
      if ("-".equals(found.group(4))) {
        seconds = -seconds;
      }
      micros =
          new BigDecimal(seconds)
              .movePointRight(6)
              .setScale(0, RoundingMode.HALF_EVEN)
              .longValueExact();
    }
    long total = (long) days + Math.floorDiv(micros, 86_400_000_000L);
    if (Math.abs(days) > MOST_DAYS || Math.abs(total) > MOST_DAYS) {
      throw refused("an interval past a timedelta");
    }
    return new Delta((int) total, Math.floorMod(micros, 86_400_000_000L));
  }

  // --- Python's text of a value --------------------------------------------------------

  /** Python's {@code str} of a {@code date}. */
  static String pyDate(LocalDate date) {
    return String.format(
        java.util.Locale.ROOT, "%04d-%02d-%02d", date.getYear(), date.getMonthValue(),
        date.getDayOfMonth());
  }

  /** Python's {@code str} of a {@code time}: its microseconds only where there are some. */
  static String pyTime(LocalTime time) {
    String text =
        String.format(
            java.util.Locale.ROOT, "%02d:%02d:%02d", time.getHour(), time.getMinute(),
            time.getSecond());
    int micros = time.getNano() / 1_000;
    return micros == 0 ? text : text + String.format(java.util.Locale.ROOT, ".%06d", micros);
  }

  /** Python's {@code str} of a naive {@code datetime}. */
  static String pyDatetime(LocalDateTime wall) {
    return pyDate(wall.toLocalDate()) + " " + pyTime(wall.toLocalTime());
  }

  /** Python's text of a UTC offset: {@code +HH:MM}, and {@code :SS} where it has seconds. */
  static String pyOffset(int seconds) {
    int magnitude = Math.abs(seconds);
    String text =
        String.format(
            java.util.Locale.ROOT, "%s%02d:%02d", seconds < 0 ? "-" : "+", magnitude / 3600,
            magnitude / 60 % 60);
    return magnitude % 60 == 0
        ? text
        : text + String.format(java.util.Locale.ROOT, ":%02d", magnitude % 60);
  }

  /** Python's {@code str} of a {@code timedelta}: {@code 1 day, 2:00:00}, {@code 0:00:00.500000}. */
  static String pyTimedelta(Delta delta) {
    long seconds = delta.micros() / 1_000_000L;
    long micros = delta.micros() % 1_000_000L;
    String text =
        String.format(
            java.util.Locale.ROOT, "%d:%02d:%02d", seconds / 3600, seconds / 60 % 60, seconds % 60);
    if (delta.days() != 0) {
      text = delta.days() + (Math.abs(delta.days()) == 1 ? " day, " : " days, ") + text;
    }
    return micros == 0 ? text : text + String.format(java.util.Locale.ROOT, ".%06d", micros);
  }

  // --- ranges --------------------------------------------------------------------------

  /** psycopg's reading of a range ({@code load_range_text}). */
  private static final Pattern RANGE =
      Pattern.compile(
          "([(\\[])(?:\"((?:[^\"]|\"\")*)\"|([^\",]+))?,(?:\"((?:[^\"]|\"\")*)\"|([^\"\\)\\]]+))?([)\\]])");

  private static final Pattern UNDOUBLE = Pattern.compile("([\"\\\\])\\1");

  /** How a range's bound is read and written: Python's {@code str} of what psycopg read. */
  @FunctionalInterface
  interface Bound {
    String text(String given, ConversionTable.Session session) throws java.sql.SQLException;
  }

  /**
   * Python's {@code str} of psycopg's {@code Range}: its bounds' brackets around each bound's
   * {@code str} (an unbounded side {@code None}), {@code empty} for an empty range.
   */
  static String pyRange(String text, Bound bound, ConversionTable.Session session)
      throws java.sql.SQLException {
    List<String> read = new ArrayList<>();
    int end = range(text, 0, bound, session, read);
    if (end != text.length()) {
      throw refused("a range it cannot parse");
    }
    return read.get(0);
  }

  /** One range read from {@code at}: its Python text added to {@code out}; where it ends. */
  private static int range(
      String text, int at, Bound bound, ConversionTable.Session session, List<String> out)
      throws java.sql.SQLException {
    if (text.startsWith("empty", at)) {
      out.add("empty");
      return at + 5;
    }
    Matcher found = RANGE.matcher(text);
    found.region(at, text.length());
    if (!found.lookingAt()) {
      throw refused("a range it cannot parse");
    }
    String lower = side(found.group(2), found.group(3), bound, session);
    String upper = side(found.group(4), found.group(5), bound, session);
    out.add(found.group(1) + lower + ", " + upper + found.group(6));
    return found.end();
  }

  private static String side(
      String quoted, String plain, Bound bound, ConversionTable.Session session)
      throws java.sql.SQLException {
    if (quoted != null) {
      return bound.text(UNDOUBLE.matcher(quoted).replaceAll("$1"), session);
    }
    return plain == null ? "None" : bound.text(plain, session);
  }

  /** Python's {@code str} of psycopg's {@code Multirange}: its ranges', braced. */
  static String pyMultirange(String text, Bound bound, ConversionTable.Session session)
      throws java.sql.SQLException {
    if (!text.startsWith("{")) {
      throw refused("a multirange it cannot parse");
    }
    if ("{}".equals(text)) {
      return "{}";
    }
    List<String> read = new ArrayList<>();
    int at = 1;
    while (true) {
      at = range(text, at, bound, session, read);
      if (at >= text.length()) {
        throw refused("a multirange it cannot parse");
      }
      char next = text.charAt(at);
      if (next == ',') {
        at++;
      } else if (next == '}' && at == text.length() - 1) {
        return "{" + String.join(", ", read) + "}";
      } else {
        throw refused("a multirange it cannot parse");
      }
    }
  }
}
