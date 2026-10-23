// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reading the control messages the core writes: a whole message, an {@code attached} one replaced
 * by the message its attachment carries (decision H7), and an error's fields as plain Java values.
 * It needs nothing but the JDK, so a core error is read wherever the library is loaded. Not part
 * of the API.
 */
public final class Messages {
  private Messages() {}

  /**
   * A control message the core wrote, whole: an {@code attached} one is replaced by the message
   * its bytes attachment carries (decision H7).
   */
  @SuppressWarnings("unchecked")
  public static Map<String, Object> read(NativeMessage message) {
    try {
      Map<String, Object> read = (Map<String, Object>) Json.read(message.control());
      if (!"attached".equals(read.get("message"))) {
        return read;
      }
      Object id = read.get("attachment");
      byte[] carried = id instanceof Number number ? message.bytes(number.intValue()) : null;
      if (carried == null) {
        throw internal("an attached message names a bytes attachment that is not with it");
      }
      Map<String, Object> whole = (Map<String, Object>) Json.read(carried);
      if ("attached".equals(whole.get("message"))) {
        throw internal("an attached message carries another");
      }
      return whole;
    } catch (Json.JsonException | ClassCastException e) {
      throw internal("the core wrote a message this host cannot read");
    }
  }

  /**
   * The {@code error} of the core's {@code error} control message, as JSON values; an {@code
   * INTERNAL_ERROR} saying so when the message cannot be read.
   */
  @SuppressWarnings("unchecked")
  static Map<String, Object> fields(NativeMessage message) {
    try {
      Object error = read(message).get("error");
      if (error instanceof Map<?, ?>) {
        return (Map<String, Object>) error;
      }
    } catch (StructuredDataException | ClassCastException e) {
      // Read as below.
    }
    Map<String, Object> unread = new LinkedHashMap<>();
    unread.put("kind", "internal");
    unread.put("code", "INTERNAL_ERROR");
    // No sentence: this host's own words, of a message it cannot read.
    unread.put("message", "the core wrote an error this host cannot read");
    return unread;
  }

  /** An error's fields as plain Java values, by name. */
  static Map<String, Object> details(Map<String, Object> error) {
    Map<String, Object> details = new LinkedHashMap<>();
    if (error.get("fields") instanceof Map<?, ?> map) {
      for (Map.Entry<?, ?> field : map.entrySet()) {
        details.put(String.valueOf(field.getKey()), detail(field.getValue()));
      }
    }
    return details;
  }

  /**
   * A field's value as plain Java: a scalar's ({@link Wording#scalar}: a whole number a {@code
   * Long} or {@code BigInteger}, a float a {@code Double}, a decimal a {@code BigDecimal}, a date
   * or time a {@code java.time} value), or a list of them.
   */
  public static Object detail(Object value) {
    if (value instanceof List<?> list) {
      List<Object> out = new ArrayList<>();
      for (Object item : list) {
        out.add(Wording.scalar(item));
      }
      return Collections.unmodifiableList(out);
    }
    return Wording.scalar(value);
  }

  static StructuredDataException internal(String message) {
    return new StructuredDataException(message, "INTERNAL_ERROR", Map.of(), null);
  }
}
