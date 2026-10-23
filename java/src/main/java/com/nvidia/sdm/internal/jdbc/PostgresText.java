// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Values Postgres writes as text, read as the reference's driver (psycopg) reads them, where the
 * Arrow that reaches the core holds what that reading gives. Not part of the API.
 */
final class PostgresText {
  private PostgresText() {}

  /** psycopg's reading of an interval in the {@code postgres} style. */
  private static final Pattern INTERVAL =
      Pattern.compile(
          "(?:([-+]?\\d+)\\s+years?\\s*)?"
              + "(?:([-+]?\\d+)\\s+mons?\\s*)?"
              + "(?:([-+]?\\d+)\\s+days?\\s*)?"
              + "(?:([-+])?(\\d+):(\\d+):(\\d+(?:\\.\\d+)?))?");

  /**
   * An interval as psycopg's {@code timedelta} of it, in microseconds: a year as 365 days, a
   * month as 30, then the days, and the time of day as seconds (a float in psycopg, rounded to
   * the microsecond as a timedelta rounds it).
   */
  static long intervalMicros(String text) {
    Matcher found = INTERVAL.matcher(text);
    if (!found.lookingAt()) {
      throw new IllegalArgumentException("an interval the reference's driver cannot read");
    }
    long days = 0;
    if (found.group(1) != null) {
      days += 365L * Long.parseLong(found.group(1));
    }
    if (found.group(2) != null) {
      days += 30L * Long.parseLong(found.group(2));
    }
    if (found.group(3) != null) {
      days += Long.parseLong(found.group(3));
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
    return Math.addExact(Math.multiplyExact(days, 86_400_000_000L), micros);
  }

  private static final DateTimeFormatter TIMESTAMP =
      new DateTimeFormatterBuilder()
          .appendPattern("uuuu-MM-dd HH:mm:ss")
          .optionalStart()
          .appendFraction(ChronoField.NANO_OF_SECOND, 0, 9, true)
          .optionalEnd()
          .toFormatter();

  /** A {@code timestamp} element of an array, as the server writes it. */
  static LocalDateTime timestamp(String text) {
    return LocalDateTime.parse(text.replace('T', ' '), TIMESTAMP);
  }
}
