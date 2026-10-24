// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal.jdbc;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

/**
 * A float as Arrow writes it as text (pyarrow's cast to {@code string}): its shortest digits,
 * plainly from {@code 1e-6} to below {@code 1e10}, else with an exponent ({@code 1e+16}, {@code
 * 1e-7}). Not part of the API.
 */
final class ArrowText {
  private ArrowText() {}

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
