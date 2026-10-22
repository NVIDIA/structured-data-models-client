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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The run loop: one core operation, from its {@code start} to its result, as the Python host's
 * {@code sdm_client._rust.run} drives it. It starts a run with this host's facts, asks it for
 * steps, takes the run's warnings in the order given, and raises the run's error as {@link
 * StructuredDataException}, its code and fields the core's. A step this host cannot answer (a
 * fetch or a model call, before the Java host has providers) cancels the run and is refused as
 * {@code INTERNAL_ERROR}. Not part of the API.
 */
public final class Host {
  /**
   * The databases the Java host has a provider for, and so the Java legs its cases run on: none
   * before T3.12; S0's validation reads no data and needs none.
   */
  public static final java.util.List<String> PROVIDERS = java.util.List.of();

  /** The largest control message, in bytes, in either direction. */
  public static final int CONTROL_CAP = 64 * 1024;

  /** Where the core's warnings are written, at {@code WARN}, as they are given. */
  private static final Logger WARNINGS = LoggerFactory.getLogger("com.nvidia.sdm.warnings");

  private Host() {}

  /** A finished run: its result, the warnings it gave in order, and the run, to take from. */
  public record Finished(Map<String, Object> result, List<Map<String, Object>> warnings, Run run)
      implements AutoCloseable {
    @Override
    public void close() {
      run.close();
    }
  }

  /** The contract version of the library loaded: Java reads it from the crate, never its own. */
  public static int contract() {
    return ((Number) NativeCore.load().get("contract")).intValue();
  }

  /**
   * An input-sized field as a bytes attachment of its JSON; what JSON cannot hold (text with an
   * unpaired surrogate, a number that is not finite) is refused as the Python host's transport
   * refuses it, {@code INVALID_REQUEST}.
   */
  public static byte[] attachment(Object value) {
    try {
      return Json.write(value);
    } catch (Json.JsonException e) {
      throw new StructuredDataException(
          "a value cannot be sent to the core: " + e.getMessage(),
          "INVALID_REQUEST",
          Map.of(),
          null);
    }
  }

  /** Runs {@code operation} to its result, with the attachments its fields name. */
  public static Finished run(Map<String, Object> operation, Map<Integer, ?> attachments) {
    Map<String, Object> start = new LinkedHashMap<>();
    start.put("message", "start");
    start.put("contract", contract());
    start.put("facts", HostFacts.now());
    start.put("operation", operation);
    byte[] message = control(start);
    NativeCore.setLogLevel(NativeLog.level());
    Run run;
    try {
      run = Run.start(message, attachments);
    } catch (NativeError refused) {
      throw error(refused);
    }
    List<Map<String, Object>> warnings = new ArrayList<>();
    try {
      while (true) {
        NativeMessage step;
        try {
          step = run.next();
        } catch (NativeError failed) {
          warn(run, warnings);
          throw error(failed);
        }
        Map<String, Object> read;
        try (step) {
          warn(run, warnings);
          read = Messages.read(step);
        }
        if (!"step".equals(read.get("message"))) {
          throw internal("the core gave a message that is not a step");
        }
        if ("done".equals(read.get("step"))) {
          @SuppressWarnings("unchecked")
          Map<String, Object> result = (Map<String, Object>) read.get("result");
          return new Finished(result, Collections.unmodifiableList(warnings), run);
        }
        try {
          run.cancel();
        } catch (NativeError ignored) {
          // The run ends here either way.
        }
        throw internal(
            "the core asked for a " + read.get("step") + " step this call cannot answer");
      }
    } catch (RuntimeException e) {
      run.close();
      throw e;
    }
  }

  /** A control message the host sends, within the control cap. */
  static byte[] control(Map<String, Object> message) {
    byte[] bytes;
    try {
      bytes = Json.write(message);
    } catch (Json.JsonException e) {
      throw internal("a control message cannot be written: " + e.getMessage());
    }
    if (bytes.length > CONTROL_CAP) {
      throw internal("a control message of " + bytes.length + " bytes is past the control cap");
    }
    return bytes;
  }

  /** Takes the run's warnings, writes each to SLF4J and keeps it, in order. */
  @SuppressWarnings("unchecked")
  private static void warn(Run run, List<Map<String, Object>> into) {
    NativeMessage taken;
    try {
      taken = run.takeWarnings();
    } catch (NativeError e) {
      return;
    }
    if (taken == null) {
      return;
    }
    try (taken) {
      Object warnings = Messages.read(taken).get("warnings");
      if (warnings instanceof List<?> list) {
        for (Object item : list) {
          Map<String, Object> warning = (Map<String, Object>) item;
          into.add(warning);
          WARNINGS.warn("[{}] {}", warning.get("code"), warning.get("message"));
        }
      }
    }
  }

  /**
   * The {@link StructuredDataException} a public call raises for the core's error: the same code,
   * fields, kind and retryability, holding nothing native, the error's message closed.
   */
  public static StructuredDataException error(NativeError raised) {
    raised.message().close();
    return new StructuredDataException(
        raised.getMessage(),
        raised.code(),
        raised.details(),
        raised.kind(),
        raised.retryable(),
        null);
  }

  static StructuredDataException internal(String message) {
    return Messages.internal(message);
  }
}
