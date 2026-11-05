// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.util.Map;

/**
 * The core's error, as the native library throws it from a {@link Run}: a {@link
 * StructuredDataException} saying the core's sentence ({@link Wording}), with the core's code,
 * fields, kind and retryability, holding the
 * {@code error} control message the core wrote and its attachments ({@link #message}), which the
 * host reads further (what to drop, what was answered) and closes. Not part of the API.
 */
public final class NativeError extends StructuredDataException {
  private static final long serialVersionUID = 1L;

  /** The message; transient, since an exception holding native handles is not serialized. */
  private final transient NativeMessage message;

  /** Made by the native library only. */
  NativeError(NativeMessage message) {
    this(message, Messages.fields(message));
  }

  private NativeError(NativeMessage message, Map<String, Object> error) {
    super(
        error.containsKey("sentence")
            ? Wording.said(error)
            : String.valueOf(error.get("message")),
        String.valueOf(error.get("code")),
        Messages.details(error),
        error.get("kind") == null ? null : String.valueOf(error.get("kind")),
        Boolean.TRUE.equals(error.get("retryable")),
        null);
    this.message = message;
  }

  /** The {@code error} control message and its attachments. */
  public NativeMessage message() {
    return message;
  }
}
