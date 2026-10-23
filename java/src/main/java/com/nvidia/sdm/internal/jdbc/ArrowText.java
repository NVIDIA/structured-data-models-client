// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;

/**
 * A value as Arrow writes it as text (pyarrow's cast to {@code string}), as the Python Postgres
 * provider writes each cell of a key set it binds: an integer as its digits, a boolean as {@code
 * true} or {@code false}, a float as its shortest digits (plainly from {@code 1e-6} to below
 * {@code 1e10}, else with an exponent: {@code 1e+16}, {@code 1e-7}), a decimal with its scale's
 * digits, a date as {@code 2024-01-02}, a timestamp as {@code 2024-01-02 03:04:05.000000} with
 * its unit's digits, and {@code Z} after a zoned one. Not part of the API.
 */
final class ArrowText {
  private ArrowText() {}

  static String of(Object value, Field field) {
    if (value instanceof String text) {
      return text;
    }
    if (value instanceof Boolean flag) {
      return flag ? "true" : "false";
    }
    if (value instanceof Long || value instanceof java.math.BigInteger) {
      return value.toString();
    }
    if (value instanceof Double real) {
      return floating(real);
    }
    if (value instanceof BigDecimal decimal) {
      return decimal.toPlainString();
    }
    if (value instanceof LocalDate date) {
      return date.toString();
    }
    if (value instanceof KeySets.Stamp stamp) {
      ArrowType.Timestamp type = stamp.type();
      String pattern =
          switch (type.getUnit()) {
            case SECOND -> "uuuu-MM-dd HH:mm:ss";
            case MILLISECOND -> "uuuu-MM-dd HH:mm:ss.SSS";
            case MICROSECOND -> "uuuu-MM-dd HH:mm:ss.SSSSSS";
            case NANOSECOND -> "uuuu-MM-dd HH:mm:ss.SSSSSSSSS";
          };
      String text =
          LocalDateTime.ofInstant(stamp.instant(), ZoneOffset.UTC)
              .format(DateTimeFormatter.ofPattern(pattern));
      return type.getTimezone() == null ? text : text + "Z";
    }
    throw new IllegalArgumentException("a key set cell of " + field.getType() + " has no text here");
  }

  /** A float32 as Arrow's cast to text writes it: the shortest digits that read back as it. */
  static String floating32(float value) {
    if (Float.isNaN(value) || Float.isInfinite(value) || value == 0) {
      return floating(value);
    }
    BigDecimal exact = new BigDecimal(value);
    for (int digits = 1; digits <= 9; digits++) {
      BigDecimal rounded = exact.round(new MathContext(digits, RoundingMode.HALF_EVEN));
      if (rounded.floatValue() == value) {
        return written(rounded, value < 0);
      }
    }
    return floating(value);
  }

  /** A float as Arrow's cast to text writes it. */
  static String floating(double value) {
    if (Double.isNaN(value)) {
      return "nan";
    }
    if (Double.isInfinite(value)) {
      return value > 0 ? "inf" : "-inf";
    }
    if (value == 0) {
      return (1 / value < 0) ? "-0" : "0";
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
    return written(shortest, value < 0);
  }

  /** Shortest digits as Arrow writes them: plainly from 1e-6 to below 1e10, else with an exponent. */
  private static String written(BigDecimal shortest, boolean negative) {
    BigDecimal stripped = shortest.stripTrailingZeros();
    String digits = stripped.unscaledValue().abs().toString();
    int exponent = digits.length() - 1 - stripped.scale();
    String sign = negative ? "-" : "";
    if (exponent >= -6 && exponent < 10) {
      return sign + stripped.abs().toPlainString();
    }
    String mantissa = digits.length() == 1 ? digits : digits.charAt(0) + "." + digits.substring(1);
    return sign + mantissa + "e" + (exponent < 0 ? "-" : "+") + Math.abs(exponent);
  }
}
