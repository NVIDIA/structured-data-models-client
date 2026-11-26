// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OptionalDataException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import com.nvidia.sdm.internal.ArrowData;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

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
  private transient Map<String, Object> details;
  private transient byte[] answeredRows;
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
    Object answered = this.details.get("answered");
    this.answeredRows = answered instanceof byte[] ipc ? ipc.clone() : null;
    this.kind = kind;
    this.retryable = retryable;
  }

  /** The contract's error code, such as {@code INVALID_REQUEST}. */
  public String code() {
    return code;
  }

  /** The structured context the error carried, by name; empty when there is none. */
  public Map<String, Object> details() {
    if (answeredRows == null) return details;
    Map<String, Object> copy = new LinkedHashMap<>(details);
    copy.put("answered", answeredRows.clone());
    return Collections.unmodifiableMap(copy);
  }

  /**
   * Rows received before a later model call failed, in the normal prediction layout.
   * The caller closes the reader; each call opens an independent snapshot.
   */
  public Optional<ArrowReader> answered(BufferAllocator allocator) {
    return answeredRows == null ? Optional.empty() : Optional.of(ArrowData.reader(answeredRows, allocator));
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

  /**
   * Writes the details after the default form, which leaves them out since javac does not take a
   * {@code Map} for serializable, so a deserialized error keeps them, as a pickled Python error
   * does.
   */
  private void writeObject(ObjectOutputStream out) throws IOException {
    out.defaultWriteObject();
    out.writeObject(new LinkedHashMap<>(details));
  }

  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    in.defaultReadObject();
    Object read;
    try {
      read = in.readObject();
    } catch (OptionalDataException written) {
      // Written by a version that left the details out.
      read = Map.of();
    }
    if (!(read instanceof Map<?, ?> map)) {
      throw new InvalidObjectException("an error's details are not a map");
    }
    Map<String, Object> restored = new LinkedHashMap<>();
    map.forEach((key, value) -> restored.put(String.valueOf(key), value));
    details = Collections.unmodifiableMap(restored);
    Object answered = details.get("answered");
    answeredRows = answered instanceof byte[] ipc ? ipc.clone() : null;
  }
}
