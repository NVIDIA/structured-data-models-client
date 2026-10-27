// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Every error this client raises, as Python's {@code StructuredDataError} is (design choice 10):
 * {@link #code} is a short stable identifier from the contract's error codes, such as {@code
 * INVALID_REQUEST} or {@code PROVIDER_ERROR}, to branch on rather than on the message; {@link
 * #details} holds the structured fields the core gave with it, by name; {@link #kind} is the
 * contract's error kind, and {@link #retryable} whether the same call may succeed if tried again.
 */
public class StructuredDataException extends RuntimeException {
  private static final long serialVersionUID = 1L;

  private final String code;
  private final transient Map<String, Object> details;
  private final String kind;
  private final boolean retryable;

  /** An error of {@code code} with {@code details}, raised by this client itself. */
  public StructuredDataException(
      String message, String code, Map<String, ?> details, Throwable cause) {
    this(message, code, details, null, false, cause);
  }

  /** An error the core gave: its code, fields, kind and retryability. */
  public StructuredDataException(
      String message,
      String code,
      Map<String, ?> details,
      String kind,
      boolean retryable,
      Throwable cause) {
    super(message, cause);
    this.code = code;
    this.details =
        details == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(details));
    this.kind = kind;
    this.retryable = retryable;
  }

  /** The contract's error code, such as {@code INVALID_REQUEST}. */
  public String code() {
    return code;
  }

  /** The structured context the error carried, by name; empty when there is none. */
  public Map<String, Object> details() {
    return details;
  }

  /** The contract's error kind, such as {@code invalid_request}; null when this client raised it. */
  public String kind() {
    return kind;
  }

  /** Whether the same call may succeed if tried again. */
  public boolean retryable() {
    return retryable;
  }

  /** {@code [CODE] message}, as Python's {@code str()} of the error. */
  @Override
  public String toString() {
    String name = getClass().getName();
    return code == null ? name + ": " + getMessage() : name + ": [" + code + "] " + getMessage();
  }
}
