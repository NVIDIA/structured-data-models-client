// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.ReadOptions.CsvType;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.table.Table;

/** Lossless CSV inference after DuckDB parses CSV quoting, compression and missing tokens. */
public final class CsvTypes {
  private CsvTypes() {}
  private static final BigInteger MAX_UNSIGNED = BigInteger.ONE.shiftLeft(64).subtract(BigInteger.ONE);

  public static Table infer(Table input, Map<String, CsvType> overrides, BufferAllocator allocator) {
    List<FieldVector> result = new ArrayList<>();
    java.util.Set<String> found = new java.util.HashSet<>();
    try {
      for (int column = 0; column < input.getVectorCount(); column++) {
        try (FieldVector raw = input.getVectorCopy(column)) {
          found.add(raw.getName());
          CsvType type = overrides.getOrDefault(raw.getName(), infer(raw));
          FieldVector output = switch (type) {
            case STRING -> new VarCharVector(raw.getName(), allocator);
            case INT64 -> new BigIntVector(raw.getName(), allocator);
            case UINT64 -> new UInt8Vector(raw.getName(), allocator);
            case FLOAT64 -> new Float8Vector(raw.getName(), allocator);
            case BOOLEAN -> new BitVector(raw.getName(), allocator);
          };
          result.add(output);
          output.setInitialCapacity(raw.getValueCount());
          output.allocateNew();
          for (int row = 0; row < raw.getValueCount(); row++) {
            if (raw.isNull(row)) { output.setNull(row); continue; }
            String text = raw.getObject(row).toString();
            if (output instanceof VarCharVector v) v.setSafe(row, text.getBytes(StandardCharsets.UTF_8));
            else if (output instanceof BigIntVector v) v.setSafe(row, new BigInteger(text.strip()).longValueExact());
            else if (output instanceof UInt8Vector v) {
              BigInteger integer = new BigInteger(text.strip());
              if (integer.signum() < 0 || integer.compareTo(MAX_UNSIGNED) > 0) throw new IllegalArgumentException("CSV value exceeds uint64");
              v.setSafe(row, integer.longValue());
            } else if (output instanceof Float8Vector v) v.setSafe(row, floating(text));
            else if (output instanceof BitVector v) {
              if (!text.equalsIgnoreCase("true") && !text.equalsIgnoreCase("false")) throw new IllegalArgumentException("CSV value is not boolean");
              v.setSafe(row, text.equalsIgnoreCase("true") ? 1 : 0);
            }
          }
          output.setValueCount(raw.getValueCount());
        }
      }
      if (!found.containsAll(overrides.keySet())) throw new IllegalArgumentException("CSV type override names a missing column");
      return new Table(result, Math.toIntExact(input.getRowCount()));
    } finally {
      result.forEach(FieldVector::close);
    }
  }

  private static CsvType infer(FieldVector raw) {
    boolean integers = true, floats = true, booleans = true, seen = false, negative = false;
    boolean unsigned = false, tooWide = false, floatWouldRound = false;
    for (int row = 0; row < raw.getValueCount(); row++) {
      if (raw.isNull(row)) continue;
      seen = true;
      String text = raw.getObject(row).toString();
      booleans &= text.equalsIgnoreCase("true") || text.equalsIgnoreCase("false");
      try {
        if (!text.strip().matches("[+-]?[0-9]+")) throw new NumberFormatException();
        BigInteger integer = new BigInteger(text.strip());
        negative |= integer.signum() < 0;
        unsigned |= integer.compareTo(BigInteger.valueOf(Long.MAX_VALUE)) > 0;
        tooWide |= integer.compareTo(BigInteger.valueOf(Long.MIN_VALUE)) < 0 || integer.compareTo(MAX_UNSIGNED) > 0;
        floatWouldRound |= integer.abs().compareTo(BigInteger.ONE.shiftLeft(53)) > 0;
      } catch (NumberFormatException error) { integers = false; }
      try { floating(text); } catch (NumberFormatException error) { floats = false; }
    }
    if (!seen) return CsvType.FLOAT64;
    if (integers) return tooWide || (unsigned && negative) ? CsvType.STRING : unsigned ? CsvType.UINT64 : CsvType.INT64;
    if (booleans) return CsvType.BOOLEAN;
    if (floats && !floatWouldRound) return CsvType.FLOAT64;
    return CsvType.STRING;
  }

  private static double floating(String value) {
    return switch (value.strip().toLowerCase(java.util.Locale.ROOT)) {
      case "inf", "+inf", "infinity", "+infinity" -> Double.POSITIVE_INFINITY;
      case "-inf", "-infinity" -> Double.NEGATIVE_INFINITY;
      default -> {
        String decimal = value.strip();
        if (!decimal.matches("[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?")) {
          throw new NumberFormatException("not a decimal CSV number");
        }
        double number = Double.parseDouble(decimal);
        if (!Double.isFinite(number)) throw new NumberFormatException("finite CSV number overflows float64");
        yield number;
      }
    };
  }
}
