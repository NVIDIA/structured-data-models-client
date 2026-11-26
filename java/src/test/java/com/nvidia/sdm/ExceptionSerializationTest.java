// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * An error serialized, as a JVM hands one to another (RMI, Spark), is the same error: its class,
 * message, code, details, kind and retryability, as Python's errors are when pickled.
 */
class ExceptionSerializationTest {
  private static <T> T serialized(T error) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
      out.writeObject(error);
    }
    try (ObjectInputStream in =
        new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
      @SuppressWarnings("unchecked")
      T read = (T) in.readObject();
      return read;
    }
  }

  private static void assertSame(StructuredDataException copied, StructuredDataException error) {
    assertNotSame(error, copied);
    assertEquals(error.getClass(), copied.getClass());
    assertEquals(error.toString(), copied.toString());
    assertEquals(error.getMessage(), copied.getMessage());
    assertEquals(error.code(), copied.code());
    assertEquals(error.kind(), copied.kind());
    assertEquals(error.retryable(), copied.retryable());
  }

  @Test
  void anErrorKeepsItsDetails() throws Exception {
    StructuredDataException error =
        new StructuredDataException(
            "the request was refused",
            "INVALID_REQUEST",
            Map.of("rule", "r", "rows", List.of(1L, 2L)),
            "invalid_request",
            false,
            null);
    StructuredDataException copied = serialized(error);
    assertSame(copied, error);
    assertEquals(error.details(), copied.details());
  }

  @Test
  void anErrorKeepsTheRowsAnsweredBeforeIt() throws Exception {
    byte[] answered = {1, 2, 3};
    StructuredDataException error =
        new StructuredDataException(
            "a later call failed", "TRANSPORT_ERROR", Map.of("answered", answered), null);
    StructuredDataException copied = serialized(error);
    assertSame(copied, error);
    assertArrayEquals(answered, (byte[]) copied.details().get("answered"));
  }

  @Test
  void anEndpointRefusalKeepsItsStatusAndDetails() throws Exception {
    ModelEndpointException error =
        ModelEndpointException.of(
            422,
            ("{\"detail\":\"Request validation failed.\",\"code\":\"INVALID_REQUEST\","
                    + "\"invalid_params\":[{\"name\":\"predict.age\","
                    + "\"reason\":\"not a number\"}]}")
                .getBytes(StandardCharsets.UTF_8));
    ModelEndpointException copied = serialized(error);
    assertSame(copied, error);
    assertEquals(422, copied.status());
    assertEquals(error.details(), copied.details());
    assertFalse(copied.details().isEmpty());
  }
}
